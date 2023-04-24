#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict
from habitat.core.dataset import Episode

import numpy as np

from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv


@registry.register_task(name="CatNavToObjTask-v0")
class CatDynNavRLEnv(DynNavRLEnv):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self._receptacle_semantic_ids: Dict[int, int] = {}

    @property
    def receptacle_semantic_ids(self):
        return self._receptacle_semantic_ids

    def reset(self, episode: Episode):
        obs = super().reset(episode)
        self._cache_receptacles()
        return obs

    def _cache_receptacles(self):
        # TODO: potentially this is slow, get receptacle list from episode instead
        rom = self._sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)
            user_attr_keys = obj.user_attributes.get_subconfig_keys()
            if any(key.startswith("receptacle_") for key in user_attr_keys):
                self._receptacle_semantic_ids[obj.object_id] = obj.creation_attributes.semantic_id

    def _generate_nav_to_pos(
        self, episode, start_hold_obj_idx=None, force_idx=None
    ):
        # learn nav to pick skill if not holding object currently
        if start_hold_obj_idx is None:
            # starting positions of candidate objects
            all_pos = np.stack([
                view_point.agent_state.position \
                for goal in episode.candidate_objects \
                for view_point in goal.view_points
            ], axis=0)
            if force_idx is not None:
                raise NotImplementedError
        else:
            # positions of candidate goal receptacles
            all_pos = np.stack([
                view_point.agent_state.position \
                for goal in episode.candidate_goal_receps \
                for view_point in goal.view_points
            ], axis=0)

        return all_pos