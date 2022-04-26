import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.vis_utils import ArgoMapVisualizer


class VisualizerDsp():
    def __init__(self):
        self.map_vis = ArgoMapVisualizer()

    def draw_once(self, post_out, data, show_map=False, test_mode=False):

        batch_size = len(data['SEQ_ID'])

        seq_id = data['SEQ_ID'][0]
        city_name = data['CITY_NAME'][0]
        orig = data['ORIG'][0].cpu().detach().numpy()
        rot = data['ROT'][0].cpu().detach().numpy()
        trajs_obs = data['TRAJS_OBS'][0].cpu().detach().numpy()
        trajs_fut = data['TRAJS_FUT'][0].cpu().detach().numpy()
        graph_da = data['MLG_DA'][0]
        graph_ls = data['MLG_LS'][0]
        pairs_da2ls = data['MLG_DA2LS'][0]
        goda_idcs = data['GODA_IDCS'][0]
        goda_label = data['GODA_LABEL'][0]

        goda_cls = post_out['goda_cls'][:graph_da['num_nodes']].cpu().detach().numpy()
        traj_pred = post_out['traj_pred'][0][:, :, :2].cpu().detach().numpy()

        goal_pred = post_out['goal_pred'][0].cpu().detach().numpy()
        prob_pred = post_out['prob_pred'][0].cpu().detach().numpy()

        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        plt.axis('off')
        ax.set_title('seq_id: {}-{}'.format(seq_id, city_name))

        if show_map:
            self.map_vis.show_surrounding_elements(ax, city_name, orig)
        else:
            rot = np.eye(2)
            orig = np.zeros(2)

        # * vis heatmap
        # nodes_da = graph_da['ctrs'].cpu().detach().numpy().dot(rot.T) + orig
        # cmap = cm.get_cmap('RdPu')
        # print('min: {}, max: {}'.format(np.min(goda_cls), np.max(goda_cls)))
        # _range = np.max(goda_cls) - np.min(goda_cls)
        # goda_cls = (goda_cls - np.min(goda_cls)) / _range
        # ax.scatter(nodes_da[:, 0], nodes_da[:, 1], color=cmap(goda_cls), alpha=0.5, marker='o', edgecolors='none')

        # if not test_mode:
        #     ax.plot(nodes_da[goda_idcs, 0], nodes_da[goda_idcs, 1],
        #             marker='^', markersize=10, color='m', ls='none', alpha=0.6)

        # goda_label = goda_label.cpu().detach().numpy()
        # goda_label[goda_label<0.85] = 0.0
        # cmap2 = cm.get_cmap('Reds')
        # ax.scatter(nodes_da[:, 0], nodes_da[:, 1], color=cmap2(
        #     goda_label), alpha=0.5, marker='o', s=25, facecolors='none')

        # goal_pred = goal_pred.dot(rot.T) + orig
        # for k, goal in enumerate(goal_pred):
        #     ax.plot(goal[0], goal[1], marker='*', color='g', markersize=12, alpha=0.75, zorder=30)
        #     ax.text(goal[0], goal[1], '{:.3f}'.format(prob_pred[k]))

        # # * draw freespace layer
        # ax.scatter(nodes_da[:, 0], nodes_da[:, 1], marker='.', color='deepskyblue', alpha=0.1)
        # # show freespace layer
        # edges_da = graph_da['ms_edges'][1]
        # edges_da_u = edges_da['u'][np.where(edges_da['u'] == 125)]
        # edges_da_v = edges_da['v'][np.where(edges_da['u'] == 125)]
        # for u, v in zip(edges_da_u, edges_da_v):
        #     ax.plot([nodes_da[u][0], nodes_da[v][0]], [nodes_da[u][1], nodes_da[v][1]], alpha=0.3)

        # # * draw lane segment layer
        # nodes_ls = graph_ls['ctrs'].cpu().detach().numpy().dot(rot.T) + orig
        # ax.scatter(nodes_ls[:, 0], nodes_ls[:, 1], color='grey', s=2, alpha=0.5)

        # feats_graph = graph_ls['feats'].cpu().detach().numpy().dot(rot.T)
        # for j in range(feats_graph.shape[0]):
        #     vec = feats_graph[j]
        #     pt0 = nodes_ls[j] - vec / 2
        #     pt1 = nodes_ls[j] + vec / 2
        #     ax.arrow(pt0[0], pt0[1], (pt1-pt0)[0], (pt1-pt0)[1], edgecolor=None, color='grey', alpha=0.3, width=0.1)

        # left_u = graph_ls['left']['u']
        # left_v = graph_ls['left']['v']
        # for u, v in zip(left_u, left_v):
        #     x = nodes_ls[u]
        #     dx = nodes_ls[v] - nodes_ls[u]
        #     ax.arrow(x[0], x[1], dx[0], dx[1], edgecolor=None, color='green', alpha=0.3)

        # right_u = graph_ls['right']['u']
        # right_v = graph_ls['right']['v']
        # for u, v in zip(right_u, right_v):
        #     x = nodes_ls[u]
        #     dx = nodes_ls[v] - nodes_ls[u]
        #     ax.arrow(x[0], x[1], dx[0], dx[1], edgecolor=None, color='red', alpha=0.3)

        # # * draw freespace to lane-segment graph
        # pairs = pairs_da2ls[pairs_da2ls[:, 1] == 500]
        # for pair in pairs:
        #     u, v = pair
        #     pt0 = nodes_ls[v]
        #     pt1 = nodes_da[u]
        #     ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], color='cyan', alpha=0.3)

        # * plot trajs
        trajs_obs = trajs_obs.dot(rot.T) + orig
        ax.plot(trajs_obs[1, -1, 0], trajs_obs[1, -1, 1], marker='s',
                color='gold', markersize=8, alpha=0.5)  # plot AV
        for i, traj in enumerate(trajs_obs):
            zorder = 10
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'gold'
            else:
                clr = 'orange'
            ax.plot(traj[:, 0], traj[:, 1], marker='.', alpha=0.5, color=clr, zorder=zorder)
            # ax.scatter(traj[:, 0], traj[:, 1], s=list(traj[:, 2] * 50 + 1), color='b')

        if not test_mode:
            trajs_fut = trajs_fut.dot(rot.T) + orig
            for i, traj in enumerate(trajs_fut):
                zorder = 10
                if i == 0:
                    clr = 'm'
                    zorder = 20
                elif i == 1:
                    clr = 'gold'
                else:
                    clr = 'orange'
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, color=clr, linewidth=3, marker='.', zorder=zorder)
                ax.plot(traj[-1, 0], traj[-1, 1], alpha=0.5, color=clr, marker='o', zorder=zorder, markersize=12)

        for i, traj in enumerate(traj_pred):
            traj = traj.dot(rot.T) + orig
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, color='g', linewidth=3, marker='.', zorder=15)
            ax.plot(traj[-1, 0], traj[-1, 1], marker='*', color='g', markersize=12, alpha=0.75, zorder=30)
            ax.text(traj[-1, 0], traj[-1, 1], '{:.2f}'.format(prob_pred[i]), zorder=15)

        if not test_mode:
            x_max = np.max([trajs_obs[0, :, 0].max(), trajs_fut[0, :, 0].max()]) + 30
            x_min = np.min([trajs_obs[0, :, 0].min(), trajs_fut[0, :, 0].min()]) - 30
            y_max = np.max([trajs_obs[0, :, 1].max(), trajs_fut[0, :, 1].max()]) + 30
            y_min = np.min([trajs_obs[0, :, 1].min(), trajs_fut[0, :, 1].min()]) - 30

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        plt.show()
