import os
import os.path as osp
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import numpy as np
from omegaconf import OmegaConf

from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.datasets.rearrange.rearrange_generator import (
    RearrangeEpisodeGenerator,
)
from habitat.datasets.rearrange.samplers.receptacle import (
    get_all_scenedataset_receptacles,
)

if TYPE_CHECKING:
    from habitat.config import DictConfig
from habitat.datasets.rearrange.run_episode_generator import (
    SceneSamplerParamsConfig,
    SceneSamplerConfig,
    RearrangeEpisodeGeneratorConfig
)
import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    path_is_navigable_given_robot,
)
from IPython import embed

import glob
import gzip
import json
import copy
# import habitat 

if __name__ == "__main__":
    dataset = RearrangeDatasetV0()
    output_path = "/habitat-lab/data/test_dataset.json.gz"
    json_file_path = "/habitat-lab/data/social_nav_episode_0415_samples.json"
    corrupt_data_path = "/habitat-lab/data/custom_data_run0.json"
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    # all_episodes = contents.copy()
    with open(corrupt_data_path, 'r') as j:
        corrupt_data = json.loads(j.read())
    new_data = corrupt_data.copy()
    # current_episode = corrupt_data["episodes"][0]
    id = 0
    for current_episode in new_data["episodes"]:
        all_episodes = contents.copy()
        for eps in all_episodes["episodes"]:
            ep = copy.deepcopy(eps)
            if ep["scene_id"] == current_episode["scene_id"] and ep["info"]["door_start"] == current_episode["info"]["door_start"]:
                ep["start_position"] = current_episode["start_position"]
                ep["start_rotation"] = np.array(current_episode["start_rotation"])
                new_ep_info = copy.deepcopy(ep["info"])
                new_ep_info["human_start"] = current_episode["info"]["human_start"]
                new_ep_info["human_rot"] = current_episode["info"]["human_rot"]
                new_ep_info["robot_goal"] = current_episode["info"]["robot_goal"]
                new_ep_info["human_goal"] = current_episode["info"]["human_goal"]
                ep["info"] = copy.deepcopy(new_ep_info)
                ep["episode_id"] = id
                dataset.episodes.append(ep)
                print(id, dataset.episodes[-1]["info"]["robot_goal"], current_episode["info"]["robot_goal"])
                id = id+1
                break
    with gzip.open(output_path, "wt") as f:
        f.write(dataset.to_json())

    # env = habitat.Env(config = config)