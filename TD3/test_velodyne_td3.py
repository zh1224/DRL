import time
import subprocess
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import os
from velodyne_env_test import GazeboEnv
import sys
import numpy as np
start_world=0
if len(sys.argv) > 1:
    start_world = int(sys.argv[1]) if len(sys.argv) > 1 else 6
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
      #  s = s.view(1, -1)
        
         # ----- Q1 -----
        h_s1 = F.relu(self.layer_1(s))     # ① state → Linear → ReLU
        h_s1 = self.layer_2_s(h_s1)        # ② 再 Linear（无激活）
        h_a1 = self.layer_2_a(a)           # ③ action 只做线性映射
        h1   = F.relu(h_s1 + h_a1)         # ④ 汇合后再一次 ReLU
        q1   = self.layer_3(h1)

        # ----- Q2 -----
        h_s2 = F.relu(self.layer_4(s))
        h_s2 = self.layer_5_s(h_s2)
        h_a2 = self.layer_5_a(a)
        h2   = F.relu(h_s2 + h_a2)
        q2   = self.layer_6(h2)
        return q1, q2
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
       # self.lstm = nn.LSTM(input_size=600, hidden_size=64, num_layers=1, batch_first=True)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a
# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        self.max_action = max_action
        self.iter_count = 0
    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.actor_target.share_memory()
        self.critic_target.share_memory()
    def get_action(self, state):
        with torch.no_grad():
                state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        try:
            self.actor.load_state_dict(
                torch.load(
                    "%s/%s_actor.pth" % (directory, filename),
                    map_location=torch.device('cpu'),
                    weights_only=True
                )
            )
        except:
            try:
                self.actor.load_state_dict(
                    torch.load(
                        "%s/%s_actor.pth" % (directory, filename),
                        map_location=torch.device('cpu')
                    )
                )
            except:
                raise ValueError("Could not load the stored model parameters")
        # try:
        #     self.critic.load_state_dict(
        #         torch.load(
        #             "%s/%s_critic.pth" % (directory, filename),
        #             map_location=torch.device('cpu'),
        #             weights_only=True
        #         )
        #     )
        # except:
        #     try:
        #         self.critic.load_state_dict(
        #             torch.load(
        #                 "%s/%s_critic.pth" % (directory, filename),
        #                 map_location=torch.device('cpu')
        #             )
        #         )
        #     except:
        #         raise ValueError("Could not load the stored model parameters")

# Set the parameters for the implementation
device = torch.device( "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep =500 # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from
base_path = "/home/l/DRL-robot-navigation-main-LZH/DRL-robot-navigation-main/catkin_ws/src/jackal-map-creation-master/test_data/end_world/"
sum=0
success_list=[]
episodetime_list=[]
average_speed_list=[]
decision_time_list=[]
timeout_error=[]
episode_real_time_list = []  # 实际耗时（秒）
path_length_list = []  # 新增：存储每回合路径长度
success=0
environment_dim = 20
robot_dim = 4
state_dim = environment_dim + robot_dim
action_dim = 2
# Create the network
network = TD3(state_dim, action_dim,1)
network.actor.eval()
try:
        network.load(file_name, "./pytorch_models/test")
except:
        raise ValueError("Could not load the stored model parameters")
for i in range(start_world,start_world+1):
    world_file = f"world_{i}.world"
    world_file_path = os.path.join(base_path, world_file)
    print(world_file_path)
# Create the testing environment
    
    env = GazeboEnv("/home/l/DRL-robot-navigation-main-LZH/DRL-robot-navigation-main/catkin_ws/src/multi_robot_scenario/launch/TD2_world_test.launch", environment_dim,"11311","11345",world_file_path)
    time.sleep(5)
    torch.manual_seed(seed)
    np.random.seed(seed)
    done = False
    episode_timesteps = 0
    state = env.reset()
   # time.sleep(35)
    last_v=0
    last_w=0
    all_decision_time=0
    print("reset success")
    all_speed=0.0
    env.choice=True
    path_length = 0.0
    last_x = env.last_odom.pose.pose.position.x
    last_y = env.last_odom.pose.pose.position.y
    episode_start_time = time.time()
    #action=[0,0]
    # Begin the testing loop
    while True :
        #print(env.goal_x,env.goal_y)

        start_time=time.perf_counter()
        action = network.get_action(np.array(state))
        end_time=time.perf_counter()
        all_decision_time=all_decision_time+(end_time-start_time)
        a_in = [(action[0] +1 ) / 2, action[1]]

        last_v=a_in[0]
        last_w=a_in[1]
         
        all_speed+=a_in[0]
        
        next_state, reward, done, target = env.step(state,a_in,0)


        current_x = env.last_odom.pose.pose.position.x
        current_y = env.last_odom.pose.pose.position.y
        step_distance = math.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
        path_length += step_distance
        last_x, last_y = current_x, current_y

      #  print(env.angle)
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        if episode_timesteps + 1 >= max_ep:
            timeout_error.append(i)
        # On termination of episode
        if done:
            sum+=1
            if episode_timesteps==0:
                episode_timesteps=1
            episode_end_time = time.time()
            episode_real_time = episode_end_time - episode_start_time
            episode_real_time_list.append(episode_real_time)  # 记录实际耗时
            path_length_list.append(path_length)
            episodetime_list.append(episode_timesteps)
            print("episode_timesteps:",episode_timesteps)
            average_speed_list.append(all_speed/episode_timesteps*1.0)
            decision_time_list.append(all_decision_time/episode_timesteps*1.0)
            
            
           
            if env.end_check:
                success_list.append(i)
                
                success+=1
                print("get!!!!")
                env.end_check=False
            # subprocess.run(["rosnode", "killall", "-a",])
            # subprocess.run(["pkill", "-f", "gzserver","gzclient","roslaunch", "robot_state_publisher"])
            # time.sleep(15)
            all_decision_time=0
            all_speed=0
            break
        else:
            state = next_state
            episode_timesteps += 1
with open("./END_TD3/success.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in success_list:
        write_and_print(item)
with open("./END_TD3/timeout.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in timeout_error:
        write_and_print(item)
with open("./END_TD3/decision_time_list.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in decision_time_list:
        write_and_print(item)      
with open("./END_TD3/average_speed_list.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in average_speed_list:
        write_and_print(item)  
with open("./END_TD3/path_length_list.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in path_length_list:
        write_and_print(item)  
with open("./END_TD3/episodetime_list.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in episodetime_list:
        write_and_print(item)  
with open("./END_TD3/episode_real_time_list.txt", "a") as f:
    def write_and_print(text):
        print(text)
        f.write(f"{text}\n")
    for item in episode_real_time_list:
         write_and_print(item)  
#     # write_and_print("\ntimeout list:")
#     # for item in timeout_error:
    #     write_and_print(item)
    # write_and_print("\nepisodetime_list:")
    # for item in episodetime_list:
    #     write_and_print(item)
    # write_and_print("\nEpisode real time list (seconds):")  # 新增实际耗时列表
    # for item in episode_real_time_list:
    #         write_and_print(f"{item:.2f}")
    # write_and_print("\naverage_speed_list:")
    # for item in average_speed_list:
    #     write_and_print(item)
    
    # write_and_print("\ndecision_time_list:")
    # for item in decision_time_list:
    #     write_and_print(item)

    # write_and_print("\nPath length list:")  # 新增路径长度输出
    # for item in path_length_list:
    #         write_and_print(item)
print("success rate:",1.0*success/sum)
print("sucess number",success)