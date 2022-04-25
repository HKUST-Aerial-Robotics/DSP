from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon

from argoverse.map_representation.map_api import ArgoverseMap


class ArgoMapVisualizer:
    def __init__(self):
        self.argo_map = ArgoverseMap()

    def show_lanes(self, ax, city_name, lane_ids, clr='g', alpha=0.2, show_lane_ids=False):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for idx in lane_ids:
            lane_cl = seq_lane_props[idx].centerline
            ax.plot(lane_cl[:, 0], lane_cl[:, 1], color=clr, alpha=alpha, linewidth=5)

            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

    def show_map_with_lanes(self,
                            ax,
                            city_name,
                            position,
                            lane_ids,
                            map_size=np.array([150.0, 150.0]),
                            show_freespace=True,
                            show_lane_ids=False):
        x_min = position[0] - map_size[0] / 2
        x_max = position[0] + map_size[0] / 2
        y_min = position[1] - map_size[1] / 2
        y_max = position[1] + map_size[1] / 2

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for idx in lane_ids:
            lane_cl = seq_lane_props[idx].centerline
            lane_polygon = self.argo_map.get_lane_segment_polygon(
                idx, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color='gray',
                        alpha=0.1,
                        edgecolor=None))

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]
            ax.arrow(pt[0],
                     pt[1],
                     vec[0],
                     vec[1],
                     alpha=0.5,
                     color='grey',
                     width=0.1,
                     zorder=1)
            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

        if show_freespace:
            drivable_area = self.argo_map.get_da_contours(city_name)
            surrounding_contours = []
            for contour in drivable_area:
                if (np.min(contour[:, 0]) < x_max
                        and np.min(contour[:, 1]) < y_max
                        and np.max(contour[:, 0]) > x_min
                        and np.max(contour[:, 1]) > y_min):
                    surrounding_contours.append(contour)

            for contour in surrounding_contours:
                ax.add_patch(
                    Polygon(contour[:, 0:2],
                            color='darkgray',
                            alpha=0.1,
                            edgecolor=None))

    def show_surrounding_elements(self,
                                  ax,
                                  city_name,
                                  position,
                                  map_size=np.array([150.0, 150.0]),
                                  show_freespace=True,
                                  show_lane_ids=False):
        x_min = position[0] - map_size[0] / 2
        x_max = position[0] + map_size[0] / 2
        y_min = position[1] - map_size[1] / 2
        y_max = position[1] + map_size[1] / 2

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]
        surrounding_lanes = {}
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max
                    and np.max(lane_cl[:, 0]) > x_min
                    and np.max(lane_cl[:, 1]) > y_min):
                surrounding_lanes[lane_id] = lane_cl

        for idx, lane_cl in surrounding_lanes.items():
            lane_polygon = self.argo_map.get_lane_segment_polygon(
                idx, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color='gray',
                        alpha=0.1,
                        edgecolor=None))

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]
            vec = vec / np.linalg.norm(vec) * 1.0
            ax.arrow(pt[0],
                     pt[1],
                     vec[0],
                     vec[1],
                     alpha=0.5,
                     color='grey',
                     width=0.1,
                     zorder=1)
            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

        if show_freespace:
            drivable_area = self.argo_map.get_da_contours(city_name)
            surrounding_contours = []
            for contour in drivable_area:
                if (np.min(contour[:, 0]) < x_max
                        and np.min(contour[:, 1]) < y_max
                        and np.max(contour[:, 0]) > x_min
                        and np.max(contour[:, 1]) > y_min):
                    surrounding_contours.append(contour)

            for contour in surrounding_contours:
                ax.add_patch(
                    Polygon(contour[:, 0:2],
                            color='darkgray',
                            alpha=0.1,
                            edgecolor=None))

    def show_all_map(self, ax, city_name, show_freespace=True):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]

            under_control = self.argo_map.lane_has_traffic_control_measure(
                lane_id, city_name)

            in_intersection = self.argo_map.lane_is_in_intersection(
                lane_id, city_name)

            turn_dir = self.argo_map.get_lane_turn_direction(
                lane_id, city_name)

            cl_clr = 'grey'

            if in_intersection:
                cl_clr = 'orange'

            if turn_dir == 'LEFT':
                cl_clr = 'blue'
            elif turn_dir == 'RIGHT':
                cl_clr = 'green'

            ax.arrow(pt[0],
                     pt[1],
                     vec[0],
                     vec[1],
                     alpha=0.5,
                     color=cl_clr,
                     width=0.1,
                     zorder=1)

            if under_control:
                p_vec = vec / np.linalg.norm(vec) * 1.5
                pt1 = pt + np.array([-p_vec[1], p_vec[0]])
                pt2 = pt + np.array([p_vec[1], -p_vec[0]])
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color='tomato',
                        alpha=0.5,
                        linewidth=2)

            lane_polygon = self.argo_map.get_lane_segment_polygon(
                lane_id, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color=cl_clr,
                        alpha=0.1,
                        edgecolor=None))

        if show_freespace:
            drivable_area = self.argo_map.get_da_contours(city_name)
            surrounding_contours = []
            for contour in drivable_area:
                surrounding_contours.append(contour)

            for contour in surrounding_contours:
                ax.add_patch(
                    Polygon(contour[:, 0:2],
                            color='darkgray',
                            alpha=0.1,
                            edgecolor=None))
