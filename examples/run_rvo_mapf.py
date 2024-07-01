#!/usr/bin/env python

import rvo2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from IPython import embed
sim = rvo2.PyRVOSimulator(1/60., 2.0, 5, 1.5, 2.0, 0.5, 2)


# self.orca_sim = rvo2.PyRVOSimulator(
#             self.dt,
#             self.neighbor_dist,
#             self.max_neighbors,
#             self.time_horizon,
#             self.time_horizon_obst,
#             self.orca_radius,
#             self.orca_max_speed)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((12, 14))
a1 = sim.addAgent((14, 14))
# a2 = sim.addAgent((1, 1))
# a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))
def plot_obstacles(orca_sim):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.grid(linestyle="dotted")
    ax.set_aspect("equal")
    ax.margins(2.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.rcParams["animation.html"] = "jshtml"

    # x, y limit from states, only for animation
    margin = 2.0 
    obstacles = []
    pt = []
    for i in range(orca_sim.getNumObstacleVertices()):
        print(orca_sim.getNextObstacleVertexNo(i))
        pt_0 = orca_sim.getObstacleVertex(i)
        pt_1 = orca_sim.getObstacleVertex(orca_sim.getNextObstacleVertexNo(i))
        pt.append(pt_0)
        pt.append(pt_1)
        pt = np.array(pt)
        ax.plot(pt[:, 0], pt[:, 1], "-o", color="black", markersize=0.1)
        pt = []
        obstacles.append(orca_sim.getObstacleVertex(i))
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
    ax.set(xlim=(xmin-2,xmax+3), ylim=(ymin-2, ymax+3))
    
    return fig, ax
# Obstacles are also supported.
# list_tuples = [[(15.654244422912598, 11.568668365478516),
#   (16.163601875305176, 10.263792037963867)],
#  [(15.390317916870117, 11.604558944702148),
#   (15.654244422912598, 11.568668365478516)],
#  [(15.106627464294434, 12.00654125213623),
#   (15.390317916870117, 11.604558944702148)],
#  [(14.610485076904297, 12.479110717773438),
#   (15.106627464294434, 12.00654125213623)],
#  [(14.563272476196289, 13.782571792602539),
#   (14.610485076904297, 12.479110717773438)],
#  [(15.069197654724121, 14.191190719604492),
#   (14.563272476196289, 13.782571792602539)],
#  [(15.473734855651855, 15.233688354492188),
#   (15.069197654724121, 14.191190719604492)],
#  [(11.684793472290039, 15.310093879699707),
#   (15.473734855651855, 15.233688354492188)],
#  [(11.681167602539062, 11.902411460876465),
#   (11.684793472290039, 15.310093879699707)],
#  [(11.209988594055176, 11.87455940246582),
#   (11.681167602539062, 11.902411460876465)],
#  [(11.174468040466309, 10.239873886108398),
#   (11.209988594055176, 11.87455940246582)],
#  [(16.163601875305176, 10.263792037963867),
#   (16.183063507080078, 10.262405395507812)],
#  [(13.429237365722656, 14.77068042755127),
#   (13.507006645202637, 12.581744194030762)],
#  [(12.569046020507812, 14.767863273620605),
#   (13.429237365722656, 14.77068042755127)],
#  [(13.507006645202637, 12.581744194030762),
#   (12.598413467407227, 12.566718101501465)]]
# list_tuple_now = [[10.263792037963867, 16.163601875305176], [11.568668365478516, 15.654244422912598], [11.604558944702148, 15.390317916870117], [12.00654125213623, 15.106627464294434], [12.479110717773438, 14.610485076904297], [13.782571792602539, 14.563272476196289], [14.191190719604492, 15.069197654724121], [15.233688354492188, 15.473734855651855], [15.310093879699707, 11.684793472290039], [11.902411460876465, 11.681167602539062], [11.87455940246582, 11.209988594055176], [10.239873886108398, 11.174468040466309], [10.262405395507812, 16.183063507080078]]
list_table_1 = [[11.588330268859863, 15.655499458312988], [11.802519798278809, 15.68454360961914], [11.816425323486328, 15.763737678527832], [11.961987495422363, 15.808496475219727], [12.042765617370605, 15.904221534729004], [14.214700698852539, 15.886024475097656], [14.234253883361816, 15.708464622497559], [14.550985336303711, 15.485882759094238], [14.538640975952148, 15.317964553833008], [14.42050552368164, 15.308235168457031], [14.435802459716797, 15.120619773864746], [14.264750480651855, 15.12462043762207], [14.23427677154541, 15.033063888549805], [13.839558601379395, 15.027502059936523], [13.844256401062012, 14.517410278320312], [13.253166198730469, 14.511800765991211], [13.280839920043945, 15.014991760253906], [13.059856414794922, 15.03444766998291], [13.032105445861816, 14.581336975097656], [12.469135284423828, 14.541062355041504], [12.462349891662598, 15.02047348022461], [11.95616340637207, 15.026178359985352], [11.93496322631836, 15.216469764709473], [11.674623489379883, 15.231595039367676], [11.646646499633789, 15.32201099395752], [11.567437171936035, 15.326179504394531]]
list_table_1 = list_table_1[:-2]
list_table_2 = [[12.555318832397461, 13.287395477294922], [12.742836952209473, 13.305675506591797], [12.741669654846191, 13.491787910461426], [13.129401206970215, 13.505683898925781], [13.116826057434082, 13.305461883544922], [13.34615421295166, 13.297121047973633], [13.36707878112793, 13.496026039123535], [13.736709594726562, 13.511171340942383], [13.743658065795898, 13.322402000427246], [14.034088134765625, 13.309804916381836], [14.064624786376953, 13.48914623260498], [14.32740306854248, 13.495879173278809], [14.34408950805664, 13.308235168457031], [14.751039505004883, 13.30294132232666], [14.749950408935547, 13.127532958984375], [14.831957817077637, 13.105292320251465], [14.833391189575195, 12.528411865234375], [12.57761001586914, 12.559179306030273], [12.56087875366211, 13.29295539855957]]
list_two = [[10.297499656677246, 16.22457790374756], [15.36426067352295, 16.23254680633545], [15.39486312866211, 11.830757141113281], [11.898244857788086, 11.7006254196167], [11.90356731414795, 11.212478637695312], [10.246871948242188, 11.215006828308105], [10.3082914352417, 16.21781635284424]]
# for i in range(len(list_tuples)-1):
#   sim.addObstacle([tuple([list_tuples[i][1], list_tuples[i][0]]), tuple([list_tuples[i+1][1], list_tuples[i+1][0]])])
# sim.addObstacle([tuple([list_tuples[i][1], list_tuples[i][0]]), tuple([list_tuples[0][1], list_tuples[0][0]])])
new_list_tuples = []
for i in list_table_1:
    new_list_tuples.append(tuple(i))
sim.addObstacle(new_list_tuples)
new_list_tuples = []
for i in list_table_2: 
    new_list_tuples.append(tuple(i))
sim.addObstacle(new_list_tuples)

new_list_tuples = []
for i in list_two:
    new_list_tuples.append(tuple(i))
sim.addObstacle(new_list_tuples)
# for i in list_tuples:
#     sim.addObstacle(i)
sim.processObstacles()
fig, ax = plot_obstacles(sim)
sim.setAgentPrefVelocity(a0, (1, 0))
sim.setAgentPrefVelocity(a1, (-1, 0))
print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))
# goal0 = np.array((15, 15))
# goal1 = np.array((12, 12))

goal0 = np.array((13, 14))
goal1 = np.array((13, 14))
print('Running simulation')

for step in range(2000):
    sim.doStep()
    [x0,y0] = sim.getAgentPosition(0)
    [x1,y1] = sim.getAgentPosition(1)
    diff0 = goal0 - np.array([x0, y0])
    diff1 = goal1 - np.array([x1, y1])
    if np.linalg.norm(diff0) <= 0.1 and np.linalg.norm(diff1) <= 0.1:
        break
    vel0 = diff0 / np.linalg.norm(diff0)
    vel1 = diff1 / np.linalg.norm(diff1)

    print("pref velocities aree ", vel0, vel1)
    sim.setAgentPrefVelocity(a0, (vel0[0], vel0[1]))
    sim.setAgentPrefVelocity(a1, (vel1[0], vel1[1]))
    positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
                 for agent_no in (a0, a1)]
    # print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
    ax.plot(x0, y0, "-o", label=f"ped {0}", markersize=0.5, color="red")
    ax.plot(x1, y1, "-o", label=f"ped {1}", markersize=0.5, color="blue")

fig.savefig("orca_sim.png")

