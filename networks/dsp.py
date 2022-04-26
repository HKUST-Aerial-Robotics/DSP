from typing import Any, Dict, List, Tuple, Union, Optional
import time
import math
import numpy as np
#
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from networks.layers import Conv1d, Res1d, Linear, LinearRes
from networks.network_utils import prob_traj_output
from networks.transformer_utils import MultiHeadAttnEncoderLayer
from utils.utils import gpu, to_long, get_tensor_memory
#
from torch_scatter import scatter_max, scatter_mean, scatter_add


def gather_actor(actors: List[Tensor], pad_flags: List[Tensor], dts: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    for i in range(batch_size):
        # actors[i] (N_a, 20, 2)
        # pad_flags[i] (N_a, 20)
        # dts[i] (20,)
        # steps (20,)

        vel = torch.zeros_like(actors[i]).to(actors[i].device)
        vel[:, 1:, :] = actors[i][:, 1:, :] - actors[i][:, :-1, :]
        steps = (torch.arange(len(dts[i])) / len(dts[i])).type(torch.float32).to(dts[i].device)  # position enc
        steps = steps.repeat((len(actors[i]), 1))
        dt = dts[i].repeat((len(actors[i]), 1))
        actors[i] = torch.cat([actors[i], vel, pad_flags[i].unsqueeze(2), dt.unsqueeze(2), steps.unsqueeze(2)], dim=2)
    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)  # [N_a, 7, 20], N_a is agent number in a batch

    actor_idcs = []  # e.g. [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])]
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    # print('batch_size: {}, num_actors: {}, actors: {}'.format(batch_size, num_actors, actors.shape))
    # print('actor_idcs: ', actor_idcs)
    actor_ctrs = [actors[idcs, 0:2, -1] for idcs in actor_idcs]
    return actors, actor_idcs, actor_ctrs


def gather_graph_da(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]['num_nodes']).to(graphs[i]["ctrs"].device)
        node_idcs.append(idcs)
        count = count + graphs[i]['num_nodes']

    graph_out = dict()
    graph_out["counts"] = counts
    graph_out["idcs"] = node_idcs
    graph_out["ctrs"] = [x["ctrs"] for x in graphs]
    graph_out["feats"] = torch.cat([x["feats"] for x in graphs], 0).type(torch.float32)

    scales = list(graphs[0]['ms_edges'].keys())
    graph_out['ms_edges'] = []
    for i, s in enumerate(scales):
        graph_out['ms_edges'].append(dict())
        for k in ['u', 'v']:
            graph_out['ms_edges'][i][k] = torch.cat(
                [graphs[j]['ms_edges'][s][k] + counts[j] for j in range(batch_size)], 0
            )
    # print('batch_size: {}, feats: {}'.format(batch_size, graph_out["feats"].shape))
    # print('ctrs: ', [ctrs.shape for ctrs in graph_out["ctrs"]])
    # print('node_idcs: ', graph_out["idcs"])
    # print('counts: ', graph_out["counts"])
    return graph_out


def gather_graph_ls(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(graphs[i]["feats"].device)
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph_out = dict()
    graph_out["counts"] = counts
    graph_out["idcs"] = node_idcs
    graph_out["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph_out[key] = torch.cat([x[key] for x in graphs], 0).type(torch.float32)

    for k1 in ["pre", "suc"]:
        graph_out[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph_out[k1].append(dict())
            for k2 in ["u", "v"]:
                graph_out[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph_out[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph_out["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph_out[k1][k2] = torch.cat(temp)
    # print('batch_size: {}, feats: {}'.format(batch_size, graph_out["feats"].shape))
    # print('ctrs: ', [ctrs.shape for ctrs in graph_out["ctrs"]])
    # print('node_idcs: ', graph_out["idcs"])
    # print('counts: ', graph_out["counts"])
    return graph_out


def gather_pairs(graph_u, graph_v, pairs):
    batch_size = len(pairs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        num_nodes = len(pairs[i])
        idcs = torch.arange(count, count + num_nodes).to(pairs[i].device)
        node_idcs.append(idcs)
        count = count + num_nodes

    offset_u = graph_u['counts']
    offset_v = graph_v['counts']

    pairs_new = []
    for i in range(batch_size):
        g_tmp = torch.zeros_like(pairs[i])
        g_tmp[:, 0] = pairs[i][:, 0] + offset_u[i]
        g_tmp[:, 1] = pairs[i][:, 1] + offset_v[i]
        pairs_new.append(g_tmp.clone())
    pairs_new = torch.cat(pairs_new)

    pairs_out = dict()
    pairs_out["counts"] = counts
    pairs_out["idcs"] = node_idcs
    pairs_out["pairs"] = pairs_new

    # print('batch size: ', batch_size, ', counts: ', counts)
    # print('u idcs: ', graph_u['idcs'])
    # print('v idcs: ', graph_v['idcs'])
    # print('pairs in: ', [pair.shape for pair in pairs])
    # print('pairs in: ', pairs)
    # print('node_idcs: ', node_idcs)
    # print('pairs new: ', [pairs_out["pairs"][idcs] for idcs in pairs_out["idcs"]])
    return pairs_out


class DAInput(nn.Module):
    """
        The DA input network: propagates information over DA layer
    """

    def __init__(self, config, n_blk=2):
        super(DAInput, self).__init__()
        n_map = config["n_da"]
        norm = "GN"
        ng = 1
        self.n_scales = config['n_scales_da']
        self.n_blk = n_blk

        self.input = LinearRes(22, n_map, norm=norm, ng=ng)

        fuse = []
        for i in range(self.n_blk):
            aggre = []
            for i in range(self.n_scales):
                aggre.append(GraphAggregationBlock(n_map, n_map, dropout=config["p_dropout"]))
            fuse.append(nn.ModuleList(aggre))
        self.fuse = nn.ModuleList(fuse)

    def forward(self, graph):
        feat = self.input(graph["feats"])  # [N_node, n_da]

        for j in range(self.n_blk):
            for i in range(self.n_scales):
                feat = self.fuse[j][i](
                    feat,
                    feat,
                    graph['ms_edges'][i]['u'],
                    graph['ms_edges'][i]['v']
                )
        return feat


class DA2DA(nn.Module):
    """
        The DA input network: propagates information over DA layer
    """

    def __init__(self, config, n_blk=2):
        super(DA2DA, self).__init__()
        n_map = config["n_da"]
        norm = "GN"
        ng = 1
        self.n_scales = config['n_scales_da']
        self.n_blk = n_blk

        fuse = []
        for i in range(self.n_blk):
            aggre = []
            for i in range(self.n_scales):
                aggre.append(GraphAggregationBlock(n_map, n_map, dropout=config["p_dropout"]))
            fuse.append(nn.ModuleList(aggre))
        self.fuse = nn.ModuleList(fuse)

    def forward(self, feat, graph):
        for j in range(self.n_blk):
            for i in range(self.n_scales):
                feat = self.fuse[j][i](
                    feat,
                    feat,
                    graph['ms_edges'][i]['u'],
                    graph['ms_edges'][i]['v']
                )
        return feat


class ActorInput(nn.Module):
    """
        Actor feature extractor with Conv1D
    """

    def __init__(self, config):
        super(ActorInput, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 7
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        # print('actors: ', actors.shape)
        out = actors  # [N_a, 4, 20]

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)
        # [N_a, 32, 20]
        # [N_a, 64, 10]
        # [N_a, 128, 5]

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]  # [N_a, 128]
        return out


class LSInput(nn.Module):
    """
        Map Grpah feature extractor with LaneConv
    """

    def __init__(self, config, n_blk=2):
        super(LSInput, self).__init__()
        self.config = config
        n_map = config["n_ls"]
        norm = "GN"
        ng = 1
        self.n_blk = n_blk

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["n_scales_ls"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(self.n_blk):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(math.gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return temp.new().resize_(0)
        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)  # [N_node, 128]
        feat += self.seg(graph["feats"])  # [N_node, 128]
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(self.n_blk):
            temp = self.fuse["ctr"][i](feat)  # XW_0
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class Actor2LS(nn.Module):
    """
        Actor to LS:  fuses real-time traffic information from actor nodes to LS layer
    """

    def __init__(self, config, n_blk=2):
        super(Actor2LS, self).__init__()
        self.config = config
        n_map = config["n_ls"]
        norm = "GN"
        ng = 1
        self.n_blk = n_blk

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 4, n_map, norm=norm, ng=ng)
        att = []
        for i in range(self.n_blk):
            att.append(Att(n_map, config["n_actor"], dropout=config["p_dropout"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat: Tensor, graph: Dict[str, Union[List[Tensor], Tensor, List[Dict[str, Tensor]], Dict[str, Tensor]]], actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                graph["turn"],
                graph["control"].unsqueeze(1),
                graph["intersect"].unsqueeze(1),
            ),
            1,
        )  # [N_node, 4]
        feat = self.meta(torch.cat((feat, meta), 1))  # [N_node, 128]
        for i in range(self.n_blk):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["a2ls_dist"],
            )
        return feat  # [N_node, 128]


class LS2LS(nn.Module):
    """
        The LS to LS block: propagates information over LS layer
    """

    def __init__(self, config, n_blk=2):
        super(LS2LS, self).__init__()
        self.config = config
        n_map = config["n_ls"]
        norm = "GN"
        ng = 1
        self.n_blk = n_blk

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["n_scales_ls"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(self.n_blk):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(math.gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """fuse map"""
        res = feat
        for i in range(self.n_blk):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
        return feat  # [N_node, 128]


class LS2Actor(nn.Module):
    """
        The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """

    def __init__(self, config, n_blk=2):
        super(LS2Actor, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_ls"]

        att = []
        for i in range(n_blk):
            att.append(Att(n_actor, n_map, dropout=config["p_dropout"]))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors  # [N_actor, 128]


class Actor2Actor(nn.Module):
    """
        The actor to actor block performs interactions among actors.
    """

    def __init__(self, config, n_blk=2):
        super(Actor2Actor, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_ls"]

        att = []
        for i in range(n_blk):  # 2 A2A blocks
            att.append(Att(n_actor, n_actor, dropout=config["p_dropout"]))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class CrossLayerAggregator(nn.Module):
    """
    """

    def __init__(self, n_agt, n_ctx, n_blk, dropout=0.0):
        super(CrossLayerAggregator, self).__init__()
        self.n_blk = n_blk

        aggre = []
        for i in range(n_blk):
            aggre.append(GraphAtt(n_agt, n_ctx, n_attn=np.max([n_agt, n_ctx]), dropout=dropout))
        self.aggre = nn.ModuleList(aggre)

    def forward(self, agt: Tensor, ctx: Tensor, u: Tensor, v: Tensor):
        for i in range(self.n_blk):
            agt = self.aggre[i](agt, ctx, u, v)
        return agt


class Att(nn.Module):
    """
        Attention block to pass context nodes information to target nodes
        This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """

    def __init__(self, n_agt: int, n_ctx: int, dropout=0.0) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(math.gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)  # elements within a range
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)  # [N,]
        wi = torch.cat(wi, 0)  # [N,]

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs.index_select(0, hi) - ctx_ctrs.index_select(0, wi)  # [N, 2]
        dist = self.dist(dist)  # [N, 128]

        query = self.query(agts).index_select(0, hi)  # [N, 128]

        ctx = ctx.index_select(0, wi)  # [N, 128]
        ctx = torch.cat((dist, query, ctx), 1)  # [N, 384]
        ctx = self.ctx(ctx)  # [N, 128]
        ctx = self.dropout(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts = self.dropout(agts)

        agts += res
        agts = self.relu(agts)
        return agts


class GraphAtt(nn.Module):
    """
        Graph attention block
    """

    def __init__(self, n_agt: int, n_ctx: int, n_attn=128, dropout=0.0) -> None:
        super(GraphAtt, self).__init__()
        norm = "GN"
        ng = 1

        self.w_q = nn.Linear(n_agt, n_attn, bias=False)
        self.w_k = nn.Linear(n_ctx, n_attn, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.agt = Linear(n_agt*2, n_agt, norm=norm, ng=ng, act=False)
        self.ctx = Linear(n_ctx, n_agt, norm=norm, ng=ng, act=False)

        self.norm = nn.GroupNorm(math.gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, agt: Tensor, ctx: Tensor, u: Tensor, v: Tensor):
        '''
            Aggregate features from ctx to agts
            u -> v
        '''
        res = agt

        q = self.w_q(agt).index_select(0, v)
        k = self.w_k(ctx).index_select(0, u)
        c = self.leaky_relu(torch.sum(q * k, dim=1))  # dot-product score
        # making logits <= 0 so that e^logit <= 1, this will improve the numerical stability
        c = (c - c.max()).exp()  # attention score for each edge

        # calculate attention weight
        c_sum = torch.zeros(len(agt), dtype=c.dtype, device=c.device)
        c_sum.index_add_(0, v, c)
        c = c / (c_sum.index_select(0, v) + 1e-16)
        c = self.dropout(c)

        ctx = self.ctx(ctx).index_select(0, u) * c.unsqueeze(1)

        agg = torch.zeros_like(agt)
        agg.index_add_(0, v, ctx)
        agt = self.agt(torch.cat([agt, agg], dim=1))
        agt = self.norm(agt)
        agt = self.relu(agt)

        agt = self.linear(agt)
        agt = self.dropout(agt)

        agt += res
        agt = self.relu(agt)

        return agt


class GraphAggregationBlock(nn.Module):
    """
        Graph aggregation block
    """

    def __init__(self, n_agt: int, n_ctx: int, dropout=0.0) -> None:
        super(GraphAggregationBlock, self).__init__()
        norm = "GN"
        ng = 1

        self.fc_1 = Linear(n_ctx, n_agt, norm=norm, ng=ng)
        self.fc_2 = Linear(n_agt*2, n_agt, norm=norm, ng=ng)

        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, agt: Tensor, ctx: Tensor, u: Tensor, v: Tensor):
        '''
            Aggregate features from ctx to agt (direction: u -> v)
            w/o edge feature, more efficient
        '''
        res = agt
        ctx = self.fc_1(ctx).index_select(0, u)
        agg, _ = scatter_max(ctx, v, dim=0, dim_size=len(agt))  # max-pooling

        agt = self.fc_2(torch.cat([agt, agg], dim=1))
        agt = self.linear(agt)
        agt = self.dropout(agt)

        agt += res
        agt = self.relu(agt)
        return agt


class GodaClassifierDaOnly(nn.Module):
    def __init__(self, config):
        super(GodaClassifierDaOnly, self).__init__()
        n_map = config["n_da"]
        norm = "GN"
        ng = 1

        self.fc_1 = Linear(n_map, 16, norm=norm, ng=ng, act=True)
        self.fc_2 = Linear(16, 8, norm=norm, ng=ng, act=True)
        self.fc_3 = nn.Linear(8, 1, bias=False)

        # self.dropout = nn.Dropout(p=config["p_dropout"])

    def forward(self, feat):
        feat = self.fc_1(feat)
        feat = self.fc_2(feat)
        # feat = self.dropout(feat)

        feat = self.fc_3(feat)
        out = torch.sigmoid_(feat).view(-1)
        return out


class Conv1dAggreBlock(nn.Module):
    """
        Aaggregation block using max-pooling
    """

    def __init__(self, n_feat: int, dropout: float = 0.0) -> None:
        super(Conv1dAggreBlock, self).__init__()
        norm = "GN"
        ng = 1
        self.n_feat = n_feat

        self.conv_1 = Conv1d(n_feat, n_feat, kernel_size=1, norm=norm, ng=ng)
        self.conv_2 = Conv1d(n_feat*2, n_feat, kernel_size=1, norm=norm, ng=ng)

        self.aggre_func = F.adaptive_avg_pool1d

        self.conv_3 = Conv1d(n_feat, n_feat, kernel_size=1, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feats):
        '''
            feats: (batch, c, N)
        '''
        res = feats
        feats = self.conv_1(feats)
        feats_mp, _ = feats.max(dim=1)  # global max-pooling

        feats_mp = feats_mp.unsqueeze(1).repeat((1, self.n_feat, 1))
        feats = torch.cat([feats, feats_mp], dim=1)
        feats = self.conv_2(feats)
        feats = self.dropout(feats)

        feats = self.conv_3(feats)
        feats += res
        feats = self.relu(feats)

        return feats


class GoalDecoder(nn.Module):
    def __init__(self, config, n_feat=32, n_pts=200):
        super(GoalDecoder, self).__init__()
        norm = "GN"
        ng = 1

        self.aggre_1 = Conv1dAggreBlock(n_feat=n_feat, dropout=config['p_dropout'])
        self.conv_1 = Conv1d(n_feat, 8, kernel_size=1, norm=norm, ng=ng)

        self.aggre_2 = Conv1dAggreBlock(n_feat=8, dropout=config['p_dropout'])
        self.conv_2 = Conv1d(8, 4, kernel_size=1, norm=norm, ng=ng)

        self.conv_3 = Conv1d(4, 1, kernel_size=1, norm=norm, ng=ng, act=False)

        self.dropout = nn.Dropout(p=config["p_dropout"])

    def forward(self, feat, coord):
        '''
            feat:   (batch, N, n_feat)
            coord:  (batch, N, 2)
        '''
        feat = feat.transpose(1, 2)

        feat = self.aggre_1(feat)
        feat = self.conv_1(feat)
        feat = self.dropout(feat)

        feat = self.aggre_2(feat)
        feat = self.conv_2(feat)
        feat = self.dropout(feat)

        feat = self.conv_3(feat)

        weights = F.softmax(feat, dim=-1).transpose(1, 2)  # weights, (batch, N, 1)
        goal = torch.sum(coord * weights, dim=1)

        return goal.unsqueeze(1), weights  # (batch, 1, 2)


class GoalGenerator(nn.Module):
    def __init__(self, config, n_blk=2):
        super(GoalGenerator, self).__init__()
        n_mode = config['n_mode']
        n_feat = 32
        norm = "GN"
        ng = 1
        self.n_blk = n_blk

        self.conv_1 = Conv1d(35, n_feat, kernel_size=1, norm=norm, ng=ng)

        self.aggre = nn.ModuleList([
            MultiHeadAttnEncoderLayer(d_x=n_feat, d_k=n_feat, d_v=n_feat, n_head=2,
                                      d_inner=n_feat, dropout=config['p_dropout'])
            for _ in range(self.n_blk)])

        self.multihead_decoder = nn.ModuleList([
            GoalDecoder(config=config, n_feat=n_feat, n_pts=200) for _ in range(n_mode)
        ])

    def forward(self, score, coord, goda_feat):
        feat = torch.cat([coord, score.unsqueeze(2), goda_feat], dim=2).transpose(1, 2)  # (batch, 35, N)
        feat = self.conv_1(feat)
        feat = feat.transpose(1, 2)  # (batch, N, n_feat)

        for enc_layer in self.aggre:
            feat, _ = enc_layer(feat, self_attn_mask=None)  # (batch, N, n_feat)

        goals = []
        weights = []
        for decoder in self.multihead_decoder:
            goals_mode, weights_mode = decoder(feat, coord)
            goals.append(goals_mode)
            weights.append(weights_mode)
        goals = torch.cat(goals, dim=1)  # (batch, n_mode, 2)

        return goals


class TrajCompletor(nn.Module):
    def __init__(self, config, prob_output=True):
        super(TrajCompletor, self).__init__()
        self.prob_output = prob_output
        norm = "GN"
        ng = 1

        self.fc_1 = LinearRes(130, 128, norm=norm, ng=ng)
        # self.fc_2 = LinearRes(128, 128, norm=norm, ng=ng)
        self.dropout = nn.Dropout(p=config["p_dropout"])

        if self.prob_output:
            self.fc_d = nn.Linear(128, 30*5, bias=False)
        else:
            self.fc_d = nn.Linear(128, 30*2, bias=False)

    def forward(self, traj_enc, goal):
        '''
            traj_enc:   (batch, 128)
            goal:       (batch, n_mode, 2)
        '''
        n_batch = goal.shape[0]
        n_mode = goal.shape[1]
        x = torch.cat([traj_enc.unsqueeze(1).repeat((1, n_mode, 1)), goal], dim=2)

        x = x.reshape(-1, 130)
        x = self.fc_1(x)
        # x = self.fc_2(x)
        x = self.dropout(x)

        if self.prob_output:
            traj_pred = self.fc_d(x).reshape(n_batch, n_mode, 30, 5)
            traj_pred = prob_traj_output(traj_pred)
        else:
            traj_pred = self.fc_d(x).reshape(n_batch, n_mode, 30, 2)

        return traj_pred


class DSP(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(DSP, self).__init__()

        self.device = device
        self.cfg = cfg

        # ~ for encoding freespace layer
        self.da_input = DAInput(cfg, n_blk=2)

        # ~ for encoding lane segment layer
        self.actor_input = ActorInput(cfg)
        self.ls_input = LSInput(cfg, n_blk=2)
        self.actor2ls = Actor2LS(cfg, n_blk=2)
        self.ls2ls = LS2LS(cfg, n_blk=2)

        # ~ for DA -> LS
        self.aggre_da2ls = CrossLayerAggregator(n_agt=cfg['n_ls'], n_ctx=cfg['n_da'], n_blk=2, dropout=cfg['p_dropout'])
        self.ls2ls_f = LS2LS(cfg, n_blk=4)

        # ~ for LS -> DA
        self.aggre_ls2da = CrossLayerAggregator(n_agt=cfg['n_da'], n_ctx=cfg['n_ls'], n_blk=2, dropout=cfg['p_dropout'])
        self.da2da = DA2DA(cfg, n_blk=2)

        # ~ fuse actors
        self.ls2actor = LS2Actor(cfg, n_blk=2)
        self.actor2actor = Actor2Actor(cfg, n_blk=2)

        # ~ decoders
        self.goda_classifier = GodaClassifierDaOnly(cfg)
        self.traj_completor = TrajCompletor(cfg)

        # ~ final goal generation
        self.goal_generator = GoalGenerator(cfg, n_blk=2)

    def forward(self, data):
        ''' 'SEQ_ID', 'CITY_NAME', 'ORIG', 'ROT', 'DT' '''
        ''' 'TRAJS_OBS', 'TRAJS_FUT' '''
        ''' 'MLG_DA', 'MLG_LS', 'MLG_DA2LS' '''
        ''' 'GODA_IDCS', 'GOAL_GT' '''
        # start = time.time()
        # ~ send to device and gather batched data
        # * Actors
        actors, actor_idcs, actor_ctrs = gather_actor(gpu(data['TRAJS_OBS'], device=self.device),
                                                      gpu(data['PAD_OBS'], device=self.device),
                                                      gpu(data['DT'], device=self.device))
        # * DA
        graph_da = gather_graph_da(to_long(gpu(data['MLG_DA'], device=self.device)))
        # * LS
        graph_ls = gather_graph_ls(to_long(gpu(data['MLG_LS'], device=self.device)))
        # * DA-LS
        pairs_da2ls = gather_pairs(graph_da, graph_ls, to_long(gpu(data['MLG_DA2LS'], device=self.device)))

        # ~ encode da layer
        feat_da = self.encode_freespace_layer(graph_da)  # (N_{DA}, n_da)
        # ~ encode ls layer
        feat_ls, actors = self.encode_lane_segment_layer(actors, actor_idcs, actor_ctrs, graph_ls)  # (N_{LS}, n_ls)

        # ~ da -> ls
        feat_ls = self.fuse_da_to_ls(feat_da, feat_ls, pairs_da2ls, graph_ls)
        # ~ ls -> da
        feat_da = self.fuse_ls_to_da(feat_ls, feat_da, pairs_da2ls, graph_da)

        # ~ ls -> actors
        actors = self.ls2actor(actors, actor_idcs, actor_ctrs, feat_ls, graph_ls['idcs'], graph_ls['ctrs'])
        actors = self.actor2actor(actors, actor_idcs, actor_ctrs)

        # ~ decoding
        goda_cls, traj_pred, goal_pred, score_pred = self.decode_goal_and_traj(
            data['SEQ_ID'], feat_da, graph_da, actors, actor_idcs, data['GOAL_GT'])

        out = {}
        out['goda_cls'] = goda_cls
        out['traj_pred'] = traj_pred
        out['goal_pred'] = goal_pred
        out['score_pred'] = score_pred

        return out

    def encode_freespace_layer(self, graph):
        feat = self.da_input(graph)
        return feat

    def encode_lane_segment_layer(self, actors, actor_idcs, actor_ctrs, graph):
        actors = self.actor_input(actors)  # [N_a, 128]
        feats = self.ls_input(graph)  # node: [N_node, 128]
        feats = self.actor2ls(feats, graph, actors, actor_idcs, actor_ctrs)
        feats = self.ls2ls(feats, graph)
        return feats, actors

    def fuse_da_to_ls(self, feat_da, feat_ls, pairs_da2ls, graph_ls):
        edge = pairs_da2ls['pairs']
        feat = self.aggre_da2ls(feat_ls, feat_da, edge[:, 0], edge[:, 1])
        feat = self.ls2ls_f(feat, graph_ls)
        return feat

    def fuse_ls_to_da(self, feat_ls, feat_da, pairs_da2ls, graph_da):
        edge = pairs_da2ls['pairs']
        feat = self.aggre_ls2da(feat_da, feat_ls, edge[:, 1], edge[:, 0])
        feat = self.da2da(feat, graph_da)
        return feat

    def decode_goal_and_traj(self, seq_ids, feat_da, graph_da, actors, actor_idcs, goal_gt):
        batch_size = len(graph_da['ctrs'])
        agent_idcs = torch.LongTensor([idcs[0] for idcs in actor_idcs])
        agent_feat = actors[agent_idcs]

        goda_cls = self.goda_classifier(feat_da)

        DEFAULT_GODA_NUM = 200
        goda_score = []
        goda_coord = []
        goda_feat = []
        for i in range(batch_size):
            scores = goda_cls[graph_da['idcs'][i]]
            _, idcs = torch.sort(scores, descending=True)
            assert len(idcs) > DEFAULT_GODA_NUM, 'Invalid goda number'

            idcs = idcs[:DEFAULT_GODA_NUM]
            goda_score.append(scores[idcs])
            goda_coord.append(graph_da['ctrs'][i][idcs])
            goda_feat.append(feat_da[graph_da['idcs'][i]][idcs])

        goda_score = torch.stack(goda_score, dim=0)
        goda_coord = torch.stack(goda_coord, dim=0)
        goda_feat = torch.stack(goda_feat, dim=0)

        goal_pred = self.goal_generator(goda_score, goda_coord, goda_feat)  # (batch, n_mode, 2)

        '''
            train: use GT goal
            test/val: use generated goals
        '''
        if self.training:
            goal_mock = gpu(torch.stack(goal_gt), device=self.device).unsqueeze(1)
            traj_pred = self.traj_completor(agent_feat, goal_mock)
            traj_pred = traj_pred.repeat((1, self.cfg['n_mode'], 1, 1))  # [batch, n_mode, len_pred, 2]
            score_pred = torch.ones(traj_pred.shape[0], traj_pred.shape[1])
        else:
            # sum/mean before sampling
            edges = graph_da['ms_edges'][0]
            goda_cls_tmp = goda_cls.clone()
            goda_cls_tmp = scatter_add(goda_cls_tmp.index_select(0, edges['u']), edges['v'], out=goda_cls_tmp, dim=0)

            traj_pred = self.traj_completor(agent_feat, goal_pred)
            # ! tmp, overwrite
            traj_pred[:, :, -1, 0:2] = goal_pred

            # score_pred = torch.ones(traj_pred.shape[0], traj_pred.shape[1])  # uniform distribution
            # ~ score from heatmap
            score_pred = []
            for b in range(batch_size):
                goals = graph_da['ctrs'][b]
                goal_scores = goda_cls_tmp[graph_da['idcs'][b]]
                score_tmp = []
                for g in goal_pred[b]:
                    dist = torch.sqrt(torch.sum((goals - g.unsqueeze(0))**2, dim=1))
                    min_idx = torch.argmin(dist)
                    score_tmp.append(goal_scores[min_idx])
                score_pred.append(score_tmp)
            score_pred = torch.Tensor(score_pred).to(self.device)

        return goda_cls, traj_pred, goal_pred, score_pred

    def post_process(self, out):
        post_out = {}

        prob_pred = F.softmax(out['score_pred'] * 1.0, dim=1)

        post_out['goda_cls'] = out['goda_cls']
        post_out['traj_pred'] = out['traj_pred']
        post_out['goal_pred'] = out['goal_pred']
        post_out['prob_pred'] = prob_pred

        return post_out
