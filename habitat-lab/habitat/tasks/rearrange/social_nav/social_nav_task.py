#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Optional, cast

import numpy as np
import magnum as mn

from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_task import PddlTask
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import NavToInfo, MyNavToInfo
from IPython import embed

@registry.register_task(name="RearrangePddlSocialNavTask-v0")
class PddlSocialNavTask(PddlTask):
    """
    Social nav task based on PddlTask for training low-level policies under multi-agent setting
    """

    _nav_to_info: Optional[NavToInfo]
    my_nav_to_info: Optional[MyNavToInfo]
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.force_obj_to_idx = None
        self.force_recep_to_name = None
        self._object_in_hand_sample_prob = config.object_in_hand_sample_prob
        self._min_start_distance = config.min_start_distance
        self._initial_robot_trans = None
        

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the articulated_agent hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_start_goal(self, episode, agent_idx, force_idx=None) -> NavToInfo:
        """
        Returns the starting information for a navigate to object task.
        """
        # print("social_nav_task is being called")
        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        # if (
        #     force_idx is None
        #     and random.random() < self._object_in_hand_sample_prob
        # ):
            # start_hold_obj_idx = self._generate_snap_to_obj()
        # print("start_hold_obj_idx: ", start_hold_obj_idx)
        # if start_hold_obj_idx is None:
        #     # Select an object at random and navigate to that object.
        #     all_pos = self._sim.get_target_objs_start()
        #     print("all_pos: ", all_pos)
        #     if force_idx is None:
        #         nav_to_pos = all_pos[np.random.randint(0, len(all_pos))]

        #     else:
        #         nav_to_pos = all_pos[force_idx]
        # else:
        #     # Select a goal at random and navigate to that goal.
        #     _, all_pos = self._sim.get_targets()
        #     nav_to_pos = all_pos[np.random.randint(0, len(all_pos))]
        
        #KL: since we are not considering hold object or force idx
        human_goal_pos = np.array(episode.info["human_goal"])
        #for robot nav_to_pos: 
        nav_to_pos = self._sim.pathfinder.get_random_navigable_point_near(
            circle_center=human_goal_pos,
            radius=2
            )
        
        def filter_func(start_pos, _):
            return (
                np.linalg.norm(start_pos - nav_to_pos)
                > self._min_start_distance
            )
        
        # (
        #     articulated_agent_pos,
        #     articulated_agent_angle,
        # ) = self._sim.set_articulated_agent_base_to_random_point(
        #     filter_func=filter_func,
        # )
        # set robot pose to robot start position
        if agent_idx == 0: #robot
            articulated_agent_pos = np.array(episode.start_position)
            articulated_agent_angle = episode.start_rotation[2]
            # nav_to_pos = np.array(episode.info["human_start"])
            # change to somewhere navigable
            # nav_to_pos = self._sim.pathfinder.get_random_navigable_point_near(
            # circle_center=np.array(episode.info["robot_goal"]),
            # radius=2
            # )
            nav_to_pos = np.array(episode.info["robot_goal"])
        elif agent_idx == 1: #human
            articulated_agent_pos = np.array(episode.info["human_start"])
            try:
                articulated_agent_angle = episode.info["human_rot"][2]
            except:
                articulated_agent_angle = 0.0
            nav_to_pos = np.array(episode.info["human_goal"])
        # print("agent_idx: ", agent_idx,"robot nav_to_pos: ", nav_to_pos)
        # print("Articulated_agent pos & angle: ", articulated_agent_pos, articulated_agent_angle)
        
        return NavToInfo(
            nav_goal_pos=nav_to_pos, #KL
            articulated_agent_start_pos=articulated_agent_pos,
            articulated_agent_start_angle=articulated_agent_angle,
            start_hold_obj_idx=start_hold_obj_idx,
        )

    @property
    def nav_goal_pos(self):
        return self.my_nav_to_info.robot_info.nav_goal_pos

    @nav_goal_pos.setter
    def nav_goal_pos(self, value):
        self._nav_to_info.nav_goal_pos = value

    @property
    def initial_robot_trans(self):
        return self._initial_robot_trans

    @initial_robot_trans.setter
    def initial_robot_trans(self, value):
        self._initial_robot_trans = value

    def reset(self, episode: Episode):
        # Process the nav target
        # print("TEST ORIGINAL agent base_pos:")
        # print("robot base_pos: ", self._sim.get_agent_data(0).articulated_agent.base_pos)
        # print("human base_pos: ", self._sim.get_agent_data(1).articulated_agent.base_pos)
        
        for agent_id in range(self._sim.num_articulated_agents):
            #KL
            
            self._nav_to_info = self._generate_nav_start_goal(
                    episode, agent_id, force_idx=self.force_obj_to_idx
                )
            print("------------Get Agent ID: ", agent_id, "--------------", self._nav_to_info)
            if (agent_id == 0):
                robot_info = self._nav_to_info
            else:
                human_info = self._nav_to_info
            self._sim.get_agent_data(
                agent_id
            ).articulated_agent.base_pos = (
                self._nav_to_info.articulated_agent_start_pos
                
            )
                
            self._sim.get_agent_data(
                agent_id
            ).articulated_agent.base_rot = (
                self._nav_to_info.articulated_agent_start_angle
            )
        print("Episode is ", episode.episode_id)
        self.my_nav_to_info = MyNavToInfo(robot_info, human_info)
        super().reset(episode)
        
        self.pddl_problem.bind_to_instance(
            self._sim, cast(RearrangeDatasetV0, self._dataset), self, episode
        )

        if self._sim.habitat_config.debug_render:
            # Visualize the position the agent is navigating to.
            self._sim.viz_ids["nav_targ_pos"] = self._sim.visualize_position(
                self._nav_to_info.nav_goal_pos,
                self._sim.viz_ids["nav_targ_pos"],
                r=0.2,
            )

        self._sim.maybe_update_articulated_agent()

        # Get the agent initial base transformation
        for agent_id in range(self._sim.num_articulated_agents):
            target_agent = self._sim.get_agent_data(agent_id).articulated_agent
            if not isinstance(target_agent, KinematicHumanoid):
                self.initial_robot_trans = target_agent.base_transformation
        self._sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_0_third_rgb'] = self._sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']

        return self._get_observations(episode)
