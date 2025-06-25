import math
import os
import random
import subprocess
import time
from os import path
import math, numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import torch, numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
GOAL_REACHED_DIST = 0.7
COLLISION_DIST = 0.45   #小范围需要改称0.15，原来是0.35  
TIME_DELTA = 0.1


class SingleObsEncoder(nn.Module):
    def __init__(self, embed_dim=64, heads=4):
        super().__init__()
        self.g1 = GATv2Conv(7, 128, heads=heads, concat=True)
        self.g2 = GATv2Conv(128*heads, embed_dim, heads=1, concat=False)

    def forward(self, data: Data, flatten=True):
        x = F.elu(self.g1(data.x, data.edge_index))   # (21, 128*heads)
        x = self.g2(x, data.edge_index)               # (21, embed_dim)

        if flatten:                                   # NFV = Vec(H)
            return x.view(-1)                        # (21*embed_dim,)
        else:                                        # 亦可全局平均
            pooled = global_mean_pool(x, data.batch) # (1, embed_dim)
            return pooled.squeeze(0)


# ① 21 节点完全图的 edge_index
N = 21
row = torch.arange(N).repeat(N)
col = torch.arange(N).repeat_interleave(N)
EDGE_INDEX = torch.stack([row, col], dim=0)             # (2, 441)

# ② 20 个扇区的 (sinθ, cosθ)
angles = torch.linspace(-np.pi/2, np.pi/2+0.03, 21)[:-1] + np.pi/20
ANGLE_FEAT = torch.stack([torch.sin(angles), torch.cos(angles)], dim=1) # (20,2)



@torch.no_grad()
def build_graph(state24, max_range=1.0, device='cpu'):
    """
    state24 : torch.FloatTensor or np.ndarray, shape (24,)
    return  : torch_geometric.data.Data  (单图，无 batch 维)
    """
    if isinstance(state24, np.ndarray):
        state24 = torch.from_numpy(state24).float()
    state24 = state24.to(device)

    laser = (state24[:20] / max_range).unsqueeze(1)        # (20,1)
    angle = ANGLE_FEAT.to(device)                          # (20,2)
    zeros3 = torch.zeros(20, 4, device=device)             # (20,4)
    sector = torch.cat([laser, angle, zeros3], dim=1)      # (20,7)

    robot = state24[20:].unsqueeze(0)                      # (1,4)
    zeros1 = torch.zeros(1, 3, device=device)              # (1,3)
    robot = torch.cat([zeros1, robot], dim=1)              # (1,7)

    x = torch.cat([sector, robot], dim=0)                  # (21,7)
    batch_vec = torch.zeros(21, dtype=torch.long, device=device)  # 同一幅图

    return Data(x=x, edge_index=EDGE_INDEX.to(device), batch=batch_vec)
# Check if the random goal position is located on an obstacle and do not accept it if it is
# Check if the random goal position is located on an obstacle and do not accept it if it is

# Check if the random goal position is located on an obstacle and do not accept it if it is
# Person 模型的坐标
person_positions = [
    (-0.367907, 4.20599),    # person_standing
    (3.36913, 1.11529),      # person_standing_0
    (-2.40739, -1.43645)     # person_standing_clone
]

# Bookshelf 书柜模型的坐标
bookshelf_positions = [
    (-0.066691, 4.316188),     # bookshelf
    {3.833930,3.833930}
]

# Cube 模型的坐标
cube_positions = [
    (0.879655, -1.18307),  # cube_20k
    (-1.79869, 1.06001),    # cube_20k_0
    (3.33188, 4.31104),      # cube_20k_1
    (-4.86938,0.495988),   # cube_20k_2
    (-1.34271, -4.27865),       # cube_20k_6
    (-4.69208, -4.36245)     # cube_20k_0_clone
]

# Dumpster 垃圾箱模型的坐标
dumpster_positions = [
    (3.83278, -3.46434),     # Dumpster
    (-3.91946, 4.33831)      # Dumpster_0
]
#十字架的坐标
ten_number=[(3.827450,-3.842281)]
#桌子的坐标
table=[(3.049900,4.569090),
       (-4.371780,4.602530),
    #    (0.613912,-3.311370),
    #    (5.100920,-1.543050),
    #    (1.163820,1.528840)
    ]
# Cabinet 柜子模型的坐标
chair=[(3.813390,0.850337),
       (0.014060,4.195960)
       ]

def check_pos(x, y):
    goal_ok = True
    
    # 检查 Person 模型，禁止区域半径为 1 米
    # for px, py in person_positions:
    #     distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
    #     if distance <= 1:
    #         goal_ok = False
    #         break
    
    # # 检查 Bookshelf 书柜模型，禁止区域半径为 1 米
    # if goal_ok:
    #     for bx, by in bookshelf_positions:
    #         distance = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
    #         if distance <= 1:
    #             goal_ok = False
    #             break
    
   # 检查 Cube 模型，禁止区域半径为 1.5 米
    if goal_ok:
        for cx, cy in cube_positions:
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance <= 2.5:
                goal_ok = False
                break
    if goal_ok:#检查椅子模型
        for cx, cy in chair:
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance <= 2:
                goal_ok = False
                break
    # # 检查 Dumpster 垃圾箱模型，禁止区域半径为 2 米
    # if goal_ok:
    #     for dx, dy in dumpster_positions:
    #         distance = math.sqrt((x - dx) ** 2 + (y - dy) ** 2)
    #         if distance <= 2:
    #             goal_ok = False
    #             break
                # 检查十字架模型，禁止区域半径为 2.2 米
    if goal_ok:
        for dx, dy in ten_number:
            distance = math.sqrt((x - dx) ** 2 + (y - dy) ** 2)
            if distance <= 2.5:
                goal_ok = False
                break

                            # 检查桌子架模型，禁止区域半径为 2.2 米
    if goal_ok:
        for dx, dy in table:
            distance = math.sqrt((x - dx) ** 2 + (y - dy) ** 2)
            if distance <= 2.2:
                goal_ok = False
                break
    # # 检查 Cabinet 模型，禁止区域半径为 2 米
    # if goal_ok:
    #     for cx, cy in cabinet_positions:
    #         distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    #         if distance <= 2:
    #             goal_ok = False
    #             break
    if x<-6.1 or x>6.1 or y<-6.1 or y>6.1:#检查边界
        goal_ok = False
    return goal_ok


        
class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim,roscoreip,gazebo_port,t):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.angle=0
        self.goal_x = 1
        self.goal_y = 0.0
        self.succes=0
        self.step_penalty=-1
        self.test=1
        self.choice=False
        self.collision_check=False
        self.end_check=False
        self.upper = 3.5
        self.lower = -3.5
        self.last_angular_velocity=0
        self.scale_to_goal=11
        self.obs_scale=0.5
        self.acceleration_scale=0
        self.last_speed=0
        self.last_distance=0
        self.smoothness_scale=0.1
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.obs_points=[]
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "zzx_run_robot"#zzx
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = roscoreip
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['ROS_MASTER_URI'] = f'http://localhost:{port}'
        os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
        subprocess.Popen(["roscore", "-p", str(port)])
        time.sleep(10)
        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        # os.environ['ROS_MASTER_URI'] = f'http://localhost:{port}'
        # os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
        print(port)
        print(gazebo_port)
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        command = f'source /home/l/DRL-robot-navigation-main-LZH/DRL-robot-navigation-main/catkin_ws/devel/setup.bash && roslaunch  {fullpath}'
        subprocess.Popen(['bash', '-c', command])
        
        print("Gazebo launched!")
        #这里要注意，如果出现pose无法找到，就是这个时间间隔过小
        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1#zzx
        )
        self.odom = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback, queue_size=1#zzx
        )
        time.sleep(25)

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
       
    def velodyne_callback(self, v):
        self.obs_points.clear()
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.angle = round(euler[2], 4)
      #  print(angle)
        for i in range(len(data)):         
            if data[i][2] > -0.2:
      #          print(data[i][0],data[i])
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 )   #

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)   #这个角度里面就保存一个最近dist的障碍物，这个距离是由障碍物的x,y,z算出来的,那xy一样的话肯定z=0最小，实际上也就是比的x,y算出来的dist，但是
                        #与单线的区别是，这个可以扫描较矮的障碍物
                  #      print(self.gaps[j][0],self.velodyne_data[j])
                        obs_angle=self.angle+self.gaps[j][0]#计算障碍物相对于小车的角度
                        if(obs_angle>3.14):
                            obs_angle=-3.14+(obs_angle-3.14)
                        if(obs_angle<-3.14):
                            obs_angle=3.14+(obs_angle+3.14)
                     #   print(obs_angle)
                        obs_position_x=self.last_odom.pose.pose.position.x+math.cos(obs_angle)*self.velodyne_data[j]#计算障碍物的全局坐标
                        obs_position_y=self.last_odom.pose.pose.position.y+math.sin(obs_angle)*self.velodyne_data[j]
                        self.obs_points.append((obs_position_x,obs_position_y,self.velodyne_data[j]))
                #        print(obs_position_x,obs_position_y)
                        
                        break
     
    def calculate_total_repulsion_force(self,x_robot, y_robot,force_magnitude_attract):
        total_force_x = 0
        total_force_y = 0
        Rho_att=math.sqrt((self.goal_x-x_robot)**2+(self.goal_y-y_robot)**2)
        for obs in self.obs_points:
          dx = obs[0] - x_robot
          dy = obs[1] - y_robot
          dr = math.sqrt(dx**2 + dy**2)
          
          if dr > 0:  # 避免除以零，理论上障碍物不应该和机器人重合
            k=10 #0.001
            Rho_obs=dr
            Rho=3
            
            if(Rho_obs<Rho):
                force_magnitude = k * (1 / Rho_obs-1.0/Rho ) *Rho_att / math.sqrt(Rho_obs)
                # if(force_magnitude>force_magnitude_attract):
                #     force_magnitude=force_magnitude_attract
                force_x = -force_magnitude * dx / dr
                force_y = -force_magnitude * dy / dr
            else:
                force_x = 0
                force_y = 0
            
            total_force_x += force_x
            total_force_y += force_y
        #    print(total_force_x,total_force_y)
        return (total_force_x, total_force_y)
    
    def calculate_attraction_force(self,x_robot, y_robot, x_target, y_target, k_a):
        dx = x_target - x_robot
        dy = y_target - y_robot
        da = math.sqrt(dx**2 + dy**2)
        force_magnitude = k_a * (da )
        return (force_magnitude * dx / da, force_magnitude * dy / da)
    
    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, last_state,action,omega):
        target = False
     
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        sigma = 0.01  # 噪声强度，根据实际调整
        # 原始数据 v_state是list
        v_state = np.array(self.velodyne_data)  # 转换为numpy数组

# 加入高斯噪声
        sigma = 0.001  # 噪声强度
        noisy_v_state = v_state 

        # Min-Max归一化（假设测距范围是0~10米）
        laser_min, laser_max = 0.0, 10.0
        normalized_v_state = (noisy_v_state - laser_min) / (laser_max - laser_min)

        # 限制在[0,1]之间
        normalized_v_state = np.clip(normalized_v_state, 0.0, 1.0)

        # 如果后续需要放回列表中：
        laser_state = [noisy_v_state.tolist()]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        #for j in range(len(self.gaps)):
           # if(self.velodyne_data[j]<10):
          #   print(self.gaps[j][0],self.gaps[j][1])
           # print(self.velodyne_data[j])#这里保存当前时刻的扫描情况
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.angle = round(euler[2], 4)
       # print(angle)
        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            print("goal reached! !  !")
            
            self.succes+=1
            self.end_check=True
            done = True
        if collision:
            self.collision_check=True
        robot_state = [distance, theta, action[0], action[1]]
        
        state = np.append(v_state, robot_state)

        graph = build_graph(state, device='cpu')         # Data 对象
   #     print("graph shape:",graph.shape)
        print("graph:",graph.x)
        print("state:",state)
        # b) 图 → NFV
        encoder = SingleObsEncoder(embed_dim=64)         # NFV 长度 21×64=1344
        nfv = encoder(graph, flatten=True).detach().cpu().numpy()               # (1344,)

    #    print("NFV shape:", nfv.shape)
        reward = self.get_reward(self,target, collision, action, min_laser,last_state,omega,distance)
        return nfv, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        self.angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, self.angle)
        object_state = self.set_self_state
        self.change_goal()



        
        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-6, 6)
            y = np.random.uniform(-6, 6)
            position_ok = check_pos(x, y)
            # for cx, cy in cylinder:
            #     distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            #     if distance <= 2.2:
            #         position_ok = False
            #         break
            # if  x<2 and x>-8 and y>-3.5 and y<-1.5:
            #         box_ok=False
            # if  x>-4 and x<-2 and y>-7 and y<3:
            #         box_ok=False
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)
        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y




                # randomly scatter boxes in the environment
        cylinder=[]
        for i in range(6):
            name = "drc_practice_blue_cylinder_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-7, 7)
                y = np.random.uniform(-7, 7)
                box_ok = check_pos(x, y)
                for j in cylinder:
                    dx=abs(x-j[0])
                    dy=abs(y-j[1])
                    if  np.linalg.norm([dx, dy]) < 4:                                                                                                                             
                        box_ok=False
                        break
                # if  x<2 and x>-8 and y>-3.5 and y<-1.5:
                #     box_ok=False
                # if  x>-4 and x<-2 and y>-7 and y<3:
                #     box_ok=False
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 2.2 or distance_to_goal < 2.2:
                    box_ok = False
            cylinder.append((x,y))
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

        # set a random goal in empty space in environment
        
        self.distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        
        # randomly scatter boxes in the environment
        
        self.publish_markers([0.0, 0.0])
        self.random_target()
        
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)
       # time.sleep(1)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        graph = build_graph(state, device='cpu')         # Data 对象

        # b) 图 → NFV
        encoder = SingleObsEncoder(embed_dim=64)         # NFV 长度 21×64=1344
        nfv = encoder(graph, flatten=True).detach().cpu().numpy()
        return nfv

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 8:
            self.upper += 0.004*30
        if self.lower > -8:
            self.lower -= 0.004*30

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)
        # self.goal_x = 2
        # self.goal_y = 2
        print("upper:",self.upper)
        print(self.goal_x,self.goal_y)

    def random_target(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment:
            name = "Target_0"     
            target_state = ModelState()
            target_state.model_name = name
            target_state.pose.position.x = self.goal_x
            target_state.pose.position.y = self.goal_y
            target_state.pose.position.z = 0.0
            target_state.pose.orientation.x = 0.0
            target_state.pose.orientation.y = 0.0
            target_state.pose.orientation.z = 0.0
            target_state.pose.orientation.w = 1.0
            self.set_state.publish(target_state)
           # time.sleep(TIME_DELTA)
            
    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        cylinder=[]
        for i in range(2):
            name = "drc_practice_blue_cylinder_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-7, 7)
                y = np.random.uniform(-7, 7)
                box_ok = check_pos(x, y)
                for j in cylinder:
                    dx=abs(x-j[0])
                    dy=abs(y-j[1])
                    if  np.linalg.norm([dx,dy]) <3:                                                                                                                             
                        box_ok=False
                        break
                # if  x<1 and x>-8 and y>-3 and y<-1:
                #  box_ok=False
                # if  x>-4 and x<-2 and y>-6 and y<3.5:
                #  box_ok=False
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 2.5 or distance_to_goal < 2.5:
                    box_ok = False
            cylinder.append((x,y))
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        
        
        min_laser = min(laser_data)
        #print(min_laser)
        if min_laser < COLLISION_DIST:
            print("collsion!")
            return True, True, min_laser
        return False, False, min_laser

    # @staticmethod
    # def get_reward(self, target, collision, action, min_laser,last_state,omega,distance_to_goal):
    #     if target:
    #         return 100.0
    #     elif collision:
    #         return -100.0
    #     else:
    #         r3 = lambda x: 1 - x if x < 1 else 0.0
    #         return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2





    @staticmethod
    def get_reward(self, target, collision, action, min_laser,last_state,omega,distance_to_goal):
        # Calculate the reward for the current state    35
            reward=0.0 
            if target:
                reward= 0  
            current_distance = distance_to_goal  # 获取当前与目标的距离
            distance_diff = self.last_distance - current_distance  # 距离变化
            reward += distance_diff * self.scale_to_goal  # 放大系数，可根据需要调整
            self.last_distance = current_distance  # 更新上一次的距离
         #   print("reward_goal:",reward)
            base = 1
            k=250
          #  print("goal_reard:",reward)
            threshold = 0.45  # 阈值，可根据需要调整
           # reward=0.0
            distance=10
            for obs in self.obs_points:
                #print(obs[2])
                distance = min(distance,obs[2])  # 假设obs[2]是障碍物的距离
            #    print(distance)
            # if(distance< threshold):
            #     reward -= self.obs_scale * math.exp(-(distance - threshold) ** 2)
            reward-=self.obs_scale* 1.0/ (1 + np.exp(k * (distance - threshold)))
            # if  distance<0.7:
                
            #     print("distance:",distance)
            #     print("reward_obs:",self.obs_scale* 1/ (1 + np.exp(k * (distance - threshold))))
            # if 1 / (1 + np.exp(k * (distance - threshold)))>0.01:
            #     print("distance:",distance)
            #     print("distance_reard:", -self.obs_scale*1 / (1 + np.exp(k * (distance - threshold))))
          #  print("reward_all:",rewWard)
            
            reward+=self.step_penalty

                        # 计算真实的角度差，范围在 (-π, π)
            angular_velocity_diff = abs(action[1] - self.last_angular_velocity)
            # 使用余弦函数计算平滑度惩罚，使得范围在 [0, -1] 之间
            smoothness_penalty = math.tanh(angular_velocity_diff) * self.smoothness_scale
            reward -= smoothness_penalty  # 更新奖励值
            # 更新角度
            self.last_angular_velocity = action[1]


            # self.last_angle = self.angle  # 更新上一时刻的角速度


                        # 计算当前速度和加速度的变化（考虑机器人的稳定性）
            speed_diff = abs(action[0] - self.last_speed)  # 假设action[0]是当前速度
            acceleration_penalty = math.tanh(speed_diff) * self.acceleration_scale  # 加速度惩罚系数
            reward -= acceleration_penalty
            self.last_speed = action[0]  # 更新上一个时间步的速度


            #reward/=2
         #   print("obs_reard:",reward)
            return  reward

    #     ""
        
    #     计算奖励值

    #     参数:
    #     target (bool): 是否到达目标
    #     collision (bool): 是否发生碰撞
    #     action (list): 当前动作
    #     min_laser (float): 最小激光雷达距离

    #     返回:
    #     float: 奖励值
    #     """
    #  #   print(last_state[22],last_state[23])
    #     # 到达目标给予100的正向奖励
    #     # reward=0.0
    #     # if target:
    #     #    reward=600.0
    #     # if collision:
    #     #     reward=-300.0    
    #     # else:
    #     #     # 根据障碍物距离计算负向奖励
    #     #     base = 1
    #     #     threshold = 1.5
    #     #     scale = 0.0001
    #     #     for obs in self.obs_points:
    #     #         distance = obs[2]  # 假设obs[2]是障碍物的距离
    #     #     #    print(distance)
    #     #         if(distance< 1):
    #     #             reward -= scale * math.exp(-(distance - threshold) ** 2)
    #     #     #print(reward)
            
    #     # reward2=0.0
    #     # angele_offset=omega
    #     # #print("angel_coffset %d",angele_offset)
    #     # reward2=0.1*math.cos(angele_offset)-0.2
    #     # #print("calculate angel reward:",reward2)
    #     # #print(reward)
    #     # #print(angele_offset,reward2)
    #     # reward+=reward2
    #     # #print(reward2)
    #     # #print(reward)
    #     # #print("\n")
    #     # return reward
    #     #print(reward)
    