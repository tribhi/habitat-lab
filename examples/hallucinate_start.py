import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

import magnum as mn
import numpy as np
from habitat.core.spaces import ActionSpace
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
import cv2
import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.utils.geometry_utils import quaternion_from_two_vectors
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import orthonormalize_rotation_shear

from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations.utils import images_to_video
import habitat_sim
# sys.path.append("/home/catkin_ws/src/")
from get_trajectory_rvo import *
# sys.path.remove("/home/catkin_ws/src/")
# sys.path.append("/root/miniconda3/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages/")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float64, Int32MultiArray
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf
import tf2_ros
import threading
from gym import spaces
from collections import OrderedDict
import imageio
import struct
from nav_msgs.msg import Path

from habitat.core.simulator import Observations

from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
import torch
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from habitat_baselines.agents.simple_agents import GoalFollower
import csv
from IPython import embed
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
THIRD_RGB_SIZE = 128
def to_grid(pathfinder, points, grid_dimensions):
    map_points = maps.to_grid(
                        points[2],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    return ([map_points[1]*0.025, map_points[0]*0.025])

def from_grid(pathfinder, points, grid_dimensions):
    floor_y = 0.0
    map_points = maps.from_grid(
                        points[1],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
    # # agent_state.position = np.array(map_points_3d)  # in world space
    # # agent.set_state(agent_state)
    map_points_3d = pathfinder.snap_point(map_points_3d)
    return map_points_3d

def img_to_world(proj, cam, W,H, u, v, debug = False):
    K = proj
    T_world_camera = cam
    rotation_0 = T_world_camera[0:3,0:3]
    translation_0 = T_world_camera[0:3,3]
    uv_1=np.array([[u,v,1]], dtype=np.float32)
    uv_1=np.array([[2*u/W -1,-2*v/H +1,1]], dtype=np.float32)
    uv_1=np.array([[2*v/H -1,-2*u/W +1,1]], dtype=np.float32)
    uv_1=uv_1.T
    assert(W == H)
    if (debug):
        embed()
    inv_rot = np.linalg.inv(rotation_0)
    A = np.matmul(np.linalg.inv(K[0:3,0:3]), uv_1)
    A[2] = 1
    t = np.array([translation_0])
    c = (A-t.T)
    d = inv_rot.dot(c)
    return d

def world_to_img(proj, cam, agent_state, W, H, debug = False):
    K = proj
    T_cam_world = cam
    pos = np.array([agent_state[0], agent_state[1], agent_state[2], 1.0])
    projection = np.matmul(T_cam_world, pos)
    # projection = np.array([projection[0], projection[2], projection[1], 1.0])
    image_coordinate = np.matmul(K, projection)
    if (debug):
        embed()
    image_coordinate = image_coordinate/image_coordinate[2]
    v = H-(image_coordinate[0]+1)*(H/2)
    u = W-(1-image_coordinate[1])*(W/2)
    return [int(u),int(v)]

class Hallucinate():
    def __init__(self, config):
        self.env = habitat.Env(config = config)
        remove_ep_list = [0,1,2,8]
        self.observations = self.env.reset()
        while self.env.current_episode.episode_id in remove_ep_list:
            self.observations = self.env.reset()
        meters_per_pixel =0.025
        map_name = "sample_map"
        hablab_topdown_map = maps.get_topdown_map(
                self.env._sim.pathfinder, 0.0, meters_per_pixel=meters_per_pixel
            )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        floor_y = 0.0
        self.top_down_map = maps.get_topdown_map(
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.025
        )
        self.third_camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']
        self.third_camera.render_camera.projection_matrix = mn.Matrix4([
            [0.3000000059604645, 0, 0, 0],
            [0, 0.3000000059604645, 0, 0],
            [0, 0, -0.002000020118430257, 0],
            [0, 0, -1.0000200271606445, 1]
        ])
        
        # self.new_img = np.asarray(self.top_down_map)
        # self.new_img = cv2.cvtColor(self.new_img,cv2.COLOR_GRAY2RGB)  
        self.new_img = hablab_topdown_map      
        self.proj = np.array(self.third_camera.render_camera.projection_matrix)
        self.cam = np.array(self.third_camera.render_camera.camera_matrix)
        world_coord = img_to_world(proj = self.proj, cam = self.cam, W = THIRD_RGB_SIZE, H = THIRD_RGB_SIZE, u = 0, v = 0)
        # head_camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_head_rgb']
        world_coord_1 = img_to_world(proj = self.proj, cam = self.cam, W = THIRD_RGB_SIZE, H = THIRD_RGB_SIZE, u = THIRD_RGB_SIZE, v = THIRD_RGB_SIZE)
        self.img_res = abs(world_coord_1[0] - world_coord[0])/THIRD_RGB_SIZE
        self.initial_state = []
        self.number_of_agents = len(self.env.sim.agents_mgr)
        self.objs = []
        self.grid_dimensions = (self.top_down_map.shape[0], self.top_down_map.shape[1])
        agents_goal_pos_3d = [self.env.current_episode.info['robot_goal'], self.env.current_episode.info['human_goal']]
        for i in range(self.number_of_agents):
            agent_pos = self.env.sim.agents_mgr[i].articulated_agent.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            print(start_pos)
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            # agents_goal_pos_3d = [self.env.current_episode.info['human_start']]
            agents_initial_velocity = [0.5,0.0]
            goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
            self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
            self.objs.append(self.env.sim.agents_mgr[i].articulated_agent)
        self.linear_velocity = [0,0,0]
        self.angular_velocity = [0,0,0]
        self.objs[0].base_rot = self.env.current_episode.start_rotation[2]
        try:
            self.objs[1].base_rot = self.env.current_episode.info['human_rot'][2]
        except:
            self.objs[1].base_rot = 0.0
            
        print("started epsiode")
        
    def read_data(self, folder):
        img = self.observations["agent_1_third_rgb"]
        im = Image.fromarray(np.uint8(img))
        im.save(folder+"/old_obs.png")
        self.image_fol = folder
        with open(self.image_fol+"/human_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        human_past_traj = full_traj
        

        with open(self.image_fol+"/robot_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        
        robot_past_traj = full_traj

        
        
        robot_pos = np.array([robot_past_traj[-1,0], robot_past_traj[-1,1]])
        human_pos = np.array([human_past_traj[-1,0], human_past_traj[-1,1]])
        factor = 0.1/self.img_res
        factor = 1/factor
        robot_pos_world = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = (60-robot_pos[1])/factor, v = (60-robot_pos[0])/factor)
        robot_pos_world[1] = 0.0
        self.objs[0].base_pos = mn.Vector3([robot_pos_world[0], robot_pos_world[1], robot_pos_world[2]])
        human_pos_world = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = (60-human_pos[1])/factor, v = (60-human_pos[0])/factor)
        human_pos_world[1] = 0.0
        self.objs[1].base_pos = mn.Vector3([human_pos_world[0], human_pos_world[1], human_pos_world[2]])
        base_vel = [0.0, 0.0]
            # print("Caught here ", not self.start_ep,  self.waiting_for_traj , self.current_point is None)
        self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
        img = self.observations["agent_1_third_rgb"]
        im = Image.fromarray(np.uint8(img))
        im.save(folder+"/new_obs.png")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()
    
    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=512, width=512, 
                        orientation =[-1.519, 0.0, 0.0], position = [0, 2.39, 0]
                    )
                }
            )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()

    my_env = Hallucinate(config)
    folder = "/home/catkin_ws/src/habitat_ros_interface/data/dataset_full_ep/irl_sept_24_3/demo_1/8"
    my_env.read_data(folder)