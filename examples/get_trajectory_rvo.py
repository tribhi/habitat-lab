from pathlib import Path
import numpy as np
import sys
import rvo2
sys.path.append("/Py_Social_ROS")
# sys.path.append("/PySocialForce")
# import pysocialforce as psf
from PIL import Image
import numpy as np
import yaml
import itertools
from IPython import embed

from habitat.utils.visualizations import maps
import toml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages/")
import tf
def my_ceil(a, precision=2):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=2):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

class ped_rvo():
    
    obs = []
    def __init__(self, my_env, map_path, config_file = "/home/catkin_ws/src/habitat_ros_interface/scripts/rvo2_default.toml", resolution = 0.025):
        num_sqrt_meter = np.sqrt(my_env.grid_dimensions[0] * my_env.grid_dimensions[1]*0.025*0.025)
        self.config = {}
        user_config = toml.load(config_file)
        self.config.update(user_config)
        self.num_sqrt_meter_per_ped = self.config.get(
            'num_sqrt_meter_per_ped', 8)
        self.num_pedestrians = max(1, int(
            num_sqrt_meter / self.num_sqrt_meter_per_ped))

        """
        Parameters for our mechanism of preventing pedestrians to back up.
        Instead, stop them and then re-sample their goals.

        num_steps_stop         A list of number of consecutive timesteps
                               each pedestrian had to stop for.
        num_steps_stop_thresh  The maximum number of consecutive timesteps
                               the pedestrian should stop for before sampling
                               a new waypoint.
        neighbor_stop_radius   Maximum distance to be considered a nearby
                               a new waypoint.
        backoff_radian_thresh  If the angle (in radian) between the pedestrian's
                               orientation and the next direction of the next
                               goal is greater than the backoffRadianThresh,
                               then the pedestrian is considered backing off.
        """
        self.num_steps_stop = [0] * self.num_pedestrians
        self.neighbor_stop_radius = self.config.get(
            'neighbor_stop_radius', 1.0)
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = self.config.get(
            'num_steps_stop_thresh', 20)
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = self.config.get(
            'backoff_radian_thresh', np.deg2rad(135.0))

        """
        Parameters for ORCA

        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        """
        self.neighbor_dist = self.config.get('orca_neighbor_dist', 2)
        self.max_neighbors = self.num_pedestrians
        self.time_horizon = self.config.get('orca_time_horizon', 4.0)  # 2.0
        self.time_horizon_obst = self.config.get('orca_time_horizon_obst', 0.5)
        self.orca_radius = self.config.get('orca_radius', 0.40)
        self.orca_max_speed = self.config.get('orca_max_speed', 1.3)
        self.dt = 1.0 / (my_env.control_frequency)
        print("DT is", self.dt)
        self.dt = 1/12   # 1/12 usually
        self.backup_pos = None
        self.orca_sim = rvo2.PyRVOSimulator(
            self.dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.orca_radius,
            self.orca_max_speed)
        print("About to load obs")
        # self.load_obs_from_map(map_path, resolution)
        # self.load_obstacles(my_env.env)
        self.fig, self.ax = plt.subplots()
        # self.plot_obstacles()
        self.max_counter = int(3/self.dt)
        self.update_number = 0
        self.orca_ped = []
        initial_state = my_env.initial_state
        for i in range(len(initial_state)):
            if i==0:
                self.orca_ped.append(self.orca_sim.addAgent((initial_state[i][0],initial_state[i][1]), velocity = (initial_state[i][2], initial_state[i][3]), radius = 0.25))
            else:
                self.orca_ped.append(self.orca_sim.addAgent((initial_state[i][0],initial_state[i][1]), velocity = (initial_state[i][2], initial_state[i][3])))
            desired_vel = np.array([initial_state[i][4] - initial_state[i][0], initial_state[i][5]-initial_state[i][1]]) 
            desired_vel = desired_vel/np.linalg.norm(desired_vel) * self.orca_max_speed
            self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], tuple(desired_vel))
        self.orca_sim.setAgentRadius(self.orca_ped[0], 0.40)
        # img = Image.open("/Py_Social_ROS/default.pgm").convert('L')
        # img.show()
        # img_np = np.array(img)  # ndarray
        # white=0
        # wall=0
        # space=0
        # obs = []
        # for i in np.arange(img_np.shape[0]):
        #     for j in np.arange(img_np.shape[1]):
        #         if img_np[i][j]== 255:  # my-map 254 ->space, 0 -> wall, 205-> nonspace
        #             white=white+1
        #             # obs.append([j,i])
        #         if img_np[i][j]== 0:    # sample-map 128 -> space, 0 -> wall, 255-> nonspace
        #             wall=wall+1
        #             self.obs.append([j,i])
        #         if img_np[i][j]== 128:
        #             space=space+1 
        self.agent_backed = False
    def load_obstacles(self, env):
        # Add scenes objects to ORCA simulator as obstacles
        
        sem_scene = env._sim.semantic_annotations()
        embed()
        for obj in sem_scene.objects:
            embed()
            if obj.category.name() in ['floor', 'ceiling']:
                continue
            
            center = obj.aabb.center
            x_len, _, z_len = (
                obj.aabb.sizes / 2.0
            )
            # Nodes to draw rectangle
            corners = [
                center + np.array([x, 0, z])
                for x, z in [
                    (-x_len, -z_len),
                    (-x_len, z_len),
                    (x_len, z_len),
                    (x_len, -z_len),
                    (-x_len, -z_len),
                ]
            ]
            quat = tf.transformations.quaternion_inverse(obj.obb.rotation)
            trans = tf.transformations.quaternion_matrix(quat)
            map_corners = [
                np.dot(trans,np.array([p[0],p[1],p[2],1.0]))
                for p in corners
            ]
            map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                40,
                                74,
                            ),
                            sim=env._sim,
                        )
                        for p in corners
                    ]
            # map_corners = [
            #     np.dot(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),np.array([p[0],p[1],p[2],1.0]))
            #     for p in corners
            # ]
            self.obs.append([map_corners[0][0], map_corners[1][0], map_corners[0][1], map_corners[1][1]])
            self.obs.append([map_corners[1][0], map_corners[2][0], map_corners[1][1], map_corners[2][1]])
            self.obs.append([map_corners[2][0], map_corners[3][0], map_corners[2][1], map_corners[3][1]])
            self.obs.append([map_corners[3][0], map_corners[4][0], map_corners[3][1], map_corners[4][1]])
    
    def load_obs_from_map(self, map_path, resolution):
        img = Image.open(map_path).convert('L')
        # img.show()
        img_np = np.array(img)  # ndarray
        white=0
        wall=0
        space=0
        for i in np.arange(img_np.shape[0]):
            for j in np.arange(img_np.shape[1]):
                if img_np[i][j]== 255:  # my-map 254 ->space, 0 -> wall, 205-> nonspace
                    white=white+1
                    # obs.append([j,i])
                if img_np[i][j]== 0:    # sample-map 128 -> space, 0 -> wall, 255-> nonspace
                    wall=wall+1
                    self.orca_sim.addObstacle([tuple([my_floor(j*resolution), my_floor(i*resolution)]), tuple([my_ceil(j*resolution),my_floor(i*resolution)]), tuple([my_ceil(j*resolution),my_ceil(i*resolution)]), tuple([my_floor(j*resolution), my_ceil(i*resolution)])])
                    # self.orca_sim.addObstacle([tuple([my_floor(j/40), my_floor(i/40)]), tuple([my_floor(j/40), my_floor(i/40)]), tuple([my_floor(j/40), my_floor(i/40)]), tuple([my_floor(j/40), my_floor(i/40)])])
                    # self.orca_sim.addObstacle([tuple([j/40, i/40]), tuple([j/40,i/40]), tuple([j/40,i/40]), tuple([j/40, i/40])])
                if img_np[i][j]== 128:
                    space=space+1
        print("building the obstacle tree")
        self.orca_sim.processObstacles()

    def get_velocity(self,initial_state, current_heading = None, groups = None, filename = None, save_anim = False):
        for i in range(len(initial_state)):
            self.orca_sim.setAgentPosition(self.orca_ped[i], tuple(np.array([initial_state[i][0], initial_state[i][1]])))
            self.orca_sim.setAgentVelocity(self.orca_ped[i], tuple(np.array([initial_state[i][2], initial_state[i][3]])))
            desired_vel = np.array([initial_state[i][4] - initial_state[i][0], initial_state[i][5]-initial_state[i][1]]) 
            goal_dist = np.linalg.norm(desired_vel)
            if goal_dist<0.3:
                desired_vel = np.array([0.0,0.0])
                print("Agent ", i, " has reached its goal")
            desired_vel = desired_vel/np.linalg.norm(desired_vel) * self.orca_max_speed
            self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], tuple(desired_vel))
        self.orca_sim.doStep()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
        alpha = np.linspace(0.5,1,self.max_counter+1)
        computed_velocity=[]
        actual_velocity = []
        for j in range(len(initial_state)):
            [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
            velx = (x - initial_state[j][0])/self.dt
            vely = (y - initial_state[j][1])/self.dt
            computed_velocity.append([velx,vely])
            [velx, vely] = self.orca_sim.getAgentVelocity(self.orca_ped[j])
            actual_velocity.append([velx,vely])
            if (self.update_number < self.max_counter):
                self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=2.5, color=colors[j], alpha = alpha[self.update_number])
                self.ax.plot(initial_state[j][4], initial_state[j][5], "-x", label=f"ped {j}", markersize=2.5, color=colors[j], alpha = alpha[self.update_number])
            print("Initial state is ",initial_state[j])
            print("Point reaches in this step is ", [x,y])
        if (self.update_number == self.max_counter):
                print("saving the offline plot!!")
                self.fig.savefig("save_stepwise_rvo2"+".png", dpi=300)
                plt.close(self.fig)
        self.update_number+=1
        
        print("Computed velocity by rvo2 is ",computed_velocity)
        print("Given velocity by rvo2 is ", actual_velocity)
        
        if save_anim:
            self.plot_obstacles()
            num_steps = 1000
            for i in range(num_steps):
                self.orca_sim.doStep()
                colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
                vel = []
                position = []
                for j in range(len(initial_state)):
                    [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
                    velx = (x - initial_state[j][0])/self.dt
                    vely = (y - initial_state[j][1])/self.dt
                    initial_state[j][0] = x
                    initial_state[j][1] = y
                    vel.append([velx,vely])
                    position.append([x,y])
                    print(velx,vely)
                    self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=0.5, color=colors[j])
                if np.all(vel[0] == [0.0,0.0] and vel[1] == [0.0,0.0]):
                    break
            self.fig.savefig(filename+".png", dpi=300)
            plt.close(self.fig)
        return np.array(computed_velocity)
            
    def reset_peds(self, initial_state):
        for i in range(len(initial_state)):
            self.orca_sim.setAgentPosition(self.orca_ped[i],(initial_state[i][0],initial_state[i][1]))
            self.orca_sim.setAgentVelocity(self.orca_ped[i], (initial_state[i][2], initial_state[i][3]))
            desired_vel = np.array([initial_state[i][4] - initial_state[i][0], initial_state[i][5]-initial_state[i][1]]) 
            if np.linalg.norm(desired_vel) > 0.2:
                desired_vel = desired_vel/np.linalg.norm(desired_vel) * self.orca_max_speed
                self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], tuple(desired_vel))
            else:
                self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], (0,0))
            self.orca_sim.setAgentVelocity(self.orca_ped[i], tuple(desired_vel))
            print("Desired velocity for agent ", i ," is ", desired_vel)

    def get_future_position(self,initial_state, sampled_goals = None, num_steps = 1000):
        print("Initial state is ",initial_state)
        # self.reset_peds(initial_state)
        vels = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
        self.ax.plot(initial_state[0][0], initial_state[0][1], "-o", label=f"ped {0}", markersize=2.5, color=colors[0])
        self.ax.plot(initial_state[0][4], initial_state[0][5], "-x", label=f"ped {0}", markersize=2.5, color=colors[0])
        self.ax.plot(initial_state[1][0], initial_state[1][1], "-o", label=f"ped {1}", markersize=2.5, color=colors[1])
        self.ax.plot(initial_state[1][4], initial_state[1][5], "-x", label=f"ped {1}", markersize=2.5, color=colors[1])
        for i in range(num_steps):
            self.orca_sim.doStep()
            vel = []
            position = []
            for j in range(len(initial_state)):
                [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
                velx = (x - initial_state[j][0])/((i+1)*self.dt)
                vely = (y - initial_state[j][1])/((i+1)*self.dt)
                vel.append([velx,vely])
                position.append([x,y])
                
                # print(velx,vely)
                # if (i == num_steps-1):
                self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=0.5, color=colors[j])
            vels.append(vel)
        # print("Velocities inside are ", vels)
            # if np.all(vel[0] == [0.0,0.0] and vel[1] == [0.0,0.0]):
            #     print("Stopping at step", i)
            #     break
            # if (self.update_number == self.max_counter):
        if True:
            # print("saving the offline plot!!")
            # self.fig.savefig("rvo2_img"+str(self.update_number)+".png", dpi=300)
            plt.close(self.fig)
            self.fig, self.ax = plt.subplots()
            self.plot_obstacles()
        self.update_number+=1
        print("Velocities are ", self.orca_sim.getAgentVelocity(self.orca_ped[0]), self.orca_sim.getAgentVelocity(self.orca_ped[1]))
        print("Agent radius is ", self.orca_sim.getAgentRadius(self.orca_ped[0]), self.orca_sim.getAgentRadius(self.orca_ped[1]))
        position[0] = self.orca_sim.getAgentPosition(self.orca_ped[0])
        position[1] = self.orca_sim.getAgentPosition(self.orca_ped[1])

        # print("Rteunring position", position)
        # if np.linalg.norm(vel[0]) <0.1 and num_steps > 100 and self.agent_backed == False:
        #     des_vel = self.orca_sim.getAgentPrefVelocity(self.orca_ped[0])
        #     position[0] = initial_state[0][0:2] - des_vel
        #     print(position)
        #     self.backup_pos = position[0]
        #     self.agent_backed = True
        #     return np.array(position)
        # if np.linalg.norm(vel[0]) < 0.5 and num_steps > 100 and self.agent_backed == True:
        #     position[0] = self.backup_pos
        #     return np.array(position)
        # self.reset_peds(initial_state)
        
        return np.array(position)
    # def get_position(self,initial_state, current_heading = None, groups = None, filename = None, save_anim = False):
    #     self.orca_ped = []
    #     for i in range(len(initial_state)):
    #         self.orca_ped.append(self.orca_sim.addAgent((initial_state[i][0],initial_state[i][1]), velocity = (initial_state[i][2], initial_state[i][3])))
    #         desired_vel = np.array([initial_state[i][4] - initial_state[i][0], initial_state[i][5]-initial_state[i][1]]) 
    #         desired_vel = desired_vel/np.linalg.norm(desired_vel) * self.orca_max_speed
    #         self.orca_sim.setAgentPrefVelocity(self.orca_ped[-1], tuple(desired_vel))
    #     self.orca_sim.doStep()
    #     colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
    #     computed_velocity=[]
    #     for j in range(len(initial_state)):
    #         [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
    #         velx = (x - initial_state[j][0])/self.dt
    #         vely = (y - initial_state[j][1])/self.dt
    #         computed_velocity.append([velx,vely])
    #         self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=2.5, color=colors[j])
    #         self.update_number+=1
    #         if (np.all(computed_velocity[0] == [0.0,0.0] and computed_velocity[1] == [0.0,0.0]) or self.update_number >= self.max_counter):
    #             self.fig.savefig(filename+".png", dpi=300)
    #             plt.close(self.fig)
    #     print(computed_velocity)
    #     if save_anim:
    #         self.plot_obstacles()
    #         num_steps = 10
    #         for i in range(num_steps):
    #             self.orca_sim.doStep()
    #             colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
    #             vel = []
    #             for j in range(len(initial_state)):
    #                 [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
    #                 velx = (x - initial_state[j][0])/self.dt
    #                 vely = (y - initial_state[j][1])/self.dt
    #                 initial_state[j][0] = x
    #                 initial_state[j][1] = y
    #                 vel.append([velx,vely])
    #                 print(velx,vely)
    #                 self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=2.5, color=colors[j])
    #             if np.all(vel[0] == [0.0,0.0] and vel[1] == [0.0,0.0]):
    #                 break
    #         self.fig.savefig(filename+".png", dpi=300)
    #         plt.close(self.fig)
    #     return np.array(computed_velocity)

    
    def plot_obstacles(self):
        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle="dotted")
        self.ax.set_aspect("equal")
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        plt.rcParams["animation.html"] = "jshtml"

        
        # x, y limit from states, only for animation
        margin = 2.0 
        obstacles = []
        pt = []
        if self.orca_sim.getNumObstacleVertices() <100:
            for i in range(self.orca_sim.getNumObstacleVertices()):
                pt_0 = self.orca_sim.getObstacleVertex(i)
                pt_1 = self.orca_sim.getObstacleVertex(self.orca_sim.getNextObstacleVertexNo(i))
                pt.append(pt_0)
                pt.append(pt_1)
                pt = np.array(pt)
                self.ax.plot(pt[:, 0], pt[:, 1], "-o", color="black", markersize=0.1)
                pt = []
                obstacles.append(self.orca_sim.getObstacleVertex(i))
        else:
            for i in range(self.orca_sim.getNumObstacleVertices()):
                obstacles.append(self.orca_sim.getObstacleVertex(i))
            xy_limits=np.array(obstacles)
            self.ax.plot(xy_limits[:, 0], xy_limits[:, 1], "o", color="black", markersize=0.1)
        xy_limits=np.array(obstacles)
        xmin = 10000
        ymin = 10000
        xmax = -10000
        ymax = -10000
        for obs in xy_limits:
            xmin = min(xmin,obs[0])
            xmax = max(xmax,obs[0])
            ymin = min(ymin,obs[1])
            ymax = max(ymax,obs[1])
        self.ax.set(xlim=(xmin-2,xmax+3), ylim=(ymin-2, ymax+3))
        
    