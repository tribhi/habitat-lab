# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces
import magnum as mn

import habitat_sim
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.robot_action import RobotAction
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction
from habitat.tasks.rearrange.actions.actions import BaseVelAction
from habitat.tasks.rearrange.utils import get_robot_spawns
from habitat.tasks.utils import get_angle


@registry.register_task_action
class OracleNavAction(BaseVelAction, HumanoidJointAction, RobotAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene.
    """

    def __init__(self, *args, task, **kwargs):
        config = kwargs['config']
        self.motion_type = config.motion_control
        if self.motion_type == 'base_velocity':
            BaseVelAction.__init__(self, *args, **kwargs)
        else:
            self.humanoid_controller = None
            HumanoidJointAction.__init__(self, *args, **kwargs)
        self._task = task
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self._targets = {}

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = get_robot_spawns(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                1,
            )
            if self.motion_type == "human_joints":
                if self.humanoid_controller is not None:
                    self.humanoid_controller.reset(self.cur_articulated_agent.base_pos)
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def step(self, *args, is_last_action, **kwargs):
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.base_velocity)
            else:
                return {}
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        curr_path_points = self._path_to_point(final_nav_targ)
        
        if curr_path_points is None:
            raise Exception
        else:
            cur_nav_targ = curr_path_points[1]

            robot_pos = np.array(self.cur_articulated_agent.base_pos)
            base_T = self.cur_articulated_agent.base_transformation
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos
            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(self, *args, is_last_action=is_last_action, **kwargs)

        
            else:
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        new_pos, new_trans = self.humanoid_controller.compute_turn(
                            mn.Vector3([rel_pos[0], 0., rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        new_pos, new_trans = self.humanoid_controller.get_walk_pose(
                            mn.Vector3([rel_targ[0], 0., rel_targ[1]])
                        )
                else:
                    new_pos, new_trans = self.humanoid_controller.get_stop_pose()
                base_action = self.humanoid_controller.vectorize_pose(
                    new_pos, new_trans
                )
                kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action
                    
                return HumanoidJointAction.step(self, *args, is_last_action=is_last_action, **kwargs)
