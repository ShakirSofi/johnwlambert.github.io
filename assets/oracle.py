#!/usr/bin/python3

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import argoverse
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.bfs import remove_duplicate_paths
from argoverse.utils.manhattan_search import compute_point_cloud_bbox
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch

DFS_MAX_DEPTH = 9
MAX_LANE_ORIENTATION_DIFFERENCE = np.pi / 4
OBS_NUM_WAYPTS = 20


def get_traj_and_city_name_from_csv(csv_fpath: str) -> Tuple[np.ndarray, str]:
    """
        Args:
        -   csv_fpath

        Returns:
        -   traj
        -   city_name
    """
    with open(csv_fpath) as csvfile:
        data_reader = csv.DictReader(csvfile)

        traj = np.zeros((50, 2))
        timestamps = np.zeros(50)
        counter = 0
        for i, row in enumerate(data_reader):
            if row["OBJECT_TYPE"] != "AGENT":
                continue
            city_name = row["CITY_NAME"]
            timestamps[counter] = row["TIMESTAMP"]
            if counter > 1:
                assert timestamps[counter] > timestamps[counter - 1]
            traj[counter, 0] = row["X"]
            traj[counter, 1] = row["Y"]
            counter += 1
    return traj, city_name


def build_city_lane_graphs(am: ArgoverseMap) -> Mapping[str, Mapping[int, List[int]]]:
    """
        Args:
        -   am

        Returns:
        -   city_graph_dict
    """
    city_lane_centerlines_dict = am.build_centerline_index()

    city_graph_dict = {}
    for city_name in ["MIA", "PIT"]:
        city_graph = {}

        for lane_id, segment in city_lane_centerlines_dict[city_name].items():
            # allow left/right lane changes
            if segment.l_neighbor_id:
                if lanes_travel_same_direction(
                    lane_id, segment.l_neighbor_id, am, city_name
                ):
                    city_graph.setdefault(str(lane_id), []).append(
                        str(segment.l_neighbor_id)
                    )

            if segment.r_neighbor_id:
                if lanes_travel_same_direction(
                    lane_id, segment.r_neighbor_id, am, city_name
                ):
                    city_graph.setdefault(str(lane_id), []).append(
                        str(segment.r_neighbor_id)
                    )

            if segment.predecessors:
                for pred_id in segment.predecessors:
                    city_graph.setdefault(str(pred_id), []).append(str(lane_id))

            if segment.successors:
                for succ_id in segment.successors:
                    city_graph.setdefault(str(lane_id), []).append(str(succ_id))

        for k, v in city_graph.items():
            city_graph[k] = list(set(v))
            city_graph[k].sort()

        city_graph_dict[city_name] = city_graph
    return city_graph_dict


def get_line_orientation(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """
        Args:
        -   

        Returns:
        -   theta, angle representing lane orientation
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return np.arctan2(y2 - y1, x2 - x1)


def lanes_travel_same_direction(lane_id_1, lane_id_2, am, city_name):
    """
        Args:
        -   lane_id_1
        -   lane_id_2
        -   am
        -   city_name

        Returns:
        -   boolean representing 
    """
    centerline1 = am.get_lane_segment_centerline(lane_id_1, city_name)[:, :2]
    centerline2 = am.get_lane_segment_centerline(lane_id_2, city_name)[:, :2]

    theta_1 = get_line_orientation(centerline1[0], centerline1[-1])
    theta_2 = get_line_orientation(centerline2[0], centerline2[-1])

    return np.absolute(theta_2 - theta_1) < MAX_LANE_ORIENTATION_DIFFERENCE


def convert_str_lists_to_int_lists(paths: List[List[str]]) -> List[List[int]]:
    """
        Args:
        -   paths

        Returns:
        -   paths
    """
    for i, path in enumerate(paths):
        path = [int(lane_id) for lane_id in path]
        paths[i] = path

    return paths


def trim_paths_with_no_inliers(
    paths: List[List[int]], lane_vote_dict: Mapping[int, int]
):
    """
        Args:
        -   paths: list of paths. each path is represented by a list of lane IDs (integers)
        -   lane_vote_dict

        Returns:
        -   paths
    """
    for i, path in enumerate(paths):
        del_ids = []

        lane_iter = iter(path)
        lane_id = next(lane_iter)
        while lane_vote_dict[lane_id] == 0:
            del_ids.append(lane_id)
            lane_id = next(lane_iter)

        lane_iter = iter(path[::-1])
        lane_id = next(lane_iter)
        while lane_vote_dict[lane_id] == 0:
            del_ids.append(lane_id)
            lane_id = next(lane_iter)

        for del_id in del_ids:
            k = path.index(del_id)
            del path[k]

        paths[i] = path

    return paths


def draw_lane_ids(
    lane_ids: List[int], am: ArgoverseMap, ax: Axes, city_name: str
) -> None:
    """
        Args:
        -   lane_ids
        -   am
        -   ax
        -   city_name

        Returns:
        -   None
    """
    for lane_id in lane_ids:
        centerline = am.get_lane_segment_centerline(int(lane_id), city_name)
        ax.text(centerline[2, 0], centerline[2, 1], f"s_{lane_id}")
        ax.text(centerline[-3, 0], centerline[-3, 1], f"e_{lane_id}")


def plot_all_nearby_lanes(am, ax, city_name, query_x, query_y):
    """
        Args:
        -   am
        -   ax
        -   city_name
        -   query_x
        -   query_y

        Returns:
        -   
    """
    am.plot_nearby_halluc_lanes(
        ax, city_name, query_x, query_y, patch_color="y", radius=40
    )

    # for lane_id in paths[best_path_id]:
    #   polygon_3d = am.get_lane_segment_polygon(int(lane_id), city_name)
    #   plot_lane_segment_patch(polygon_3d[:,:2], ax, color=color)


def get_dict_key_with_max_value(dictionary: Mapping[Any, int]) -> List[Any]:
    """ 
        Args:
        -   dictionary:

        Returns:
        -   top_keys
    """
    max_val = max(dictionary.values())

    top_keys = []
    for key, value in dictionary.items():
        if max_val == value:
            top_keys += [key]

    return top_keys


def draw_traj(traj: np.ndarray, ax: Axes) -> None:
    """
        Args:
        -   

        Returns:
        -   None
    """
    lineX = traj[:, 0]
    lineY = traj[:, 1]
    ax.scatter(
        lineX[:OBS_NUM_WAYPTS],
        lineY[:OBS_NUM_WAYPTS],
        20,
        marker=".",
        color="g",
        zorder=10,
    )
    ax.scatter(
        lineX[OBS_NUM_WAYPTS:],
        lineY[OBS_NUM_WAYPTS:],
        20,
        marker=".",
        color="r",
        zorder=10,
    )
    # ax.plot(lineX, lineY, "--", color='k', linewidth=1, zorder=0)
    ax.text(lineX[0], lineY[0], "s")
    ax.text(lineX[-1], lineY[-1], "e")
    ax.axis("equal")


def remove_repeated_paths(paths):
    """ 
    """
    num_paths = len(paths)
    is_dup_arr = np.zeros(num_paths, dtype=bool)

    for i,path in enumerate(paths):
        if paths[i] in paths[:i]:
            is_dup_arr[i] = True

    del_indices = np.where(is_dup_arr)[0]
    nondup_paths = [path for i, path in enumerate(paths) if i not in del_indices]
    return nondup_paths


def find_all_paths_from_src(graph, start, max_depth=2, remove_duplicates=True):
    """ 
    from source only, iteratively

    """
    paths = []
    stack = []
    stack.append([start])

    while len(stack) > 0:
        path = stack.pop()
        u = path[-1]

        if u not in graph:
            continue

        for v in graph[u]:
            if (v not in path) and (len(path) <= max_depth):
                newpath = path + [v]
                paths += [newpath]
                stack.append(newpath)

    if remove_duplicates:
        paths = remove_duplicate_paths(paths)
    return paths


def main(data_dir):
    """ 
    """
    fnames = glob.glob(f"{data_dir}/*.csv")
    fnames = [Path(fname).name for fname in fnames]

    am = ArgoverseMap()
    city_graph_dict = build_city_lane_graphs(am)

    for fname in fnames:

        # # very hard cases
        # if int(Path(fname).stem) not in [
        #     166633, 150381,11905, 136010, 49854, 27155]:
        #     continue

        # # hard cases -- ,
        # [174545,119781, 210709, 139445, 11381, 175883, 122703,  166633]: #23333,,124414]:
        #

        csv_fpath = f"{data_dir}/{fname}"
        traj, city_name = get_traj_and_city_name_from_csv(csv_fpath)

        plausible_start_ids = set()
        lane_vote_dict = defaultdict(int)
        for j, pt in enumerate(traj):
            contained_ids = am.get_lane_segments_containing_xy(pt[0], pt[1], city_name)
            for id in contained_ids:
                lane_vote_dict[id] += 1
                plausible_start_ids.add(id)

        plausible_start_ids = list(plausible_start_ids)
        plausible_start_ids.sort()
        paths = []
        # START BFS IN ANY PLAUSIBLE LANE ID!
        for start_id in plausible_start_ids:
            paths.extend(
                find_all_paths_from_src(city_graph_dict[city_name], str(start_id), max_depth=DFS_MAX_DEPTH)
            )

        paths = convert_str_lists_to_int_lists(paths)
        paths = trim_paths_with_no_inliers(paths, lane_vote_dict)
        paths = remove_repeated_paths(paths)

        path_votes_dict = defaultdict(int)
        for path_id, path in enumerate(paths):
            for id in path:
                path_votes_dict[path_id] += lane_vote_dict[id]

        # find which path has most inliers
        best_path_ids = get_dict_key_with_max_value(path_votes_dict)

        # if they are all tied, take the shortest
        best_path_lengths = [len(paths[id]) for id in best_path_ids]
        min_best_path_length = min(best_path_lengths)
        best_path_ids = [id for id in best_path_ids if len(paths[id]) == min_best_path_length]

        fig = plt.figure(figsize=(15, 15))
        plt.axis("off")
        ax = fig.add_subplot(111)
        plot_all_nearby_lanes(
            am, ax, city_name, np.mean(traj[:, 0]), np.mean(traj[:, 1])
        )

        colors = ["g", "b", "r", "m"]
        # then plot this path
        for best_path_id in best_path_ids:
            color = colors[best_path_id % len(colors)]
            print(
                "Candidate: ",
                paths[best_path_id],
                " with ",
                path_votes_dict[best_path_id],
            )
            for lane_id in paths[best_path_id]:
                polygon_3d = am.get_lane_segment_polygon(lane_id, city_name)
                plot_lane_segment_patch(polygon_3d[:, :2], ax, color=color)
                ax.text(
                    np.mean(polygon_3d[:, 0]), np.mean(polygon_3d[:, 1]), f"{lane_id}"
                )
            # just use one for now
            break

        # draw_lane_ids(plausible_start_ids, am, ax, city_name)

        all_nearby_lane_ids = []
        for path in paths:
            all_nearby_lane_ids.extend(path)
        draw_lane_ids(set(all_nearby_lane_ids), am, ax, city_name)

        draw_traj(traj, ax)
        xmin, ymin, xmax, ymax = compute_point_cloud_bbox(traj)

        WINDOW_RADIUS_MARGIN = 10
        xmin -= WINDOW_RADIUS_MARGIN
        xmax += WINDOW_RADIUS_MARGIN
        ymin -= WINDOW_RADIUS_MARGIN
        ymax += WINDOW_RADIUS_MARGIN
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        plt.savefig(f"/Users/johnlamb/Documents/argoverse-api/temp_files_oracle/{Path(fname).stem}.png")
        plt.close("all")



if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/johnlamb/Downloads/train/data",
        help="path to dataset where csv files live",
    )

    args = parser.parse_args()
    main(args.data_dir)


