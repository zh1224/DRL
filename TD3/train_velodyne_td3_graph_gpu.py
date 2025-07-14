# -*- coding: utf-8 -*

from copy import deepcopy
import os
import time
import math
import numpy as np
import gc, torch
import torch.nn as nn
import queue
import sys
import torch.nn.functional as F
from torch_geometric.data import Batch
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from collections import deque
import torch.multiprocessing as mp #A
from replay_buffer_graph import ReplayBuffer
from velodyne_env import GazeboEnv
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
# —— 完全图 edge_index：21×21，只生成一次 —— 
# =============================================================

# ────────────────── 全局常量 ──────────────────
torch.set_printoptions(sci_mode=False, precision=4)
N_NODES = 21                                         # 20 LiDAR + 1 Robot
robot_idx = torch.full((20,), 20)
lidar_idx = torch.arange(20)

row = torch.cat([robot_idx, lidar_idx])
col = torch.cat([lidar_idx, robot_idx])

EDGE_INDEX_CONST = torch.stack([row, col], dim=0)  # (2, 40)


# 20 条分界：-90° … +90°
bound = torch.linspace(-np.pi/2, np.pi/2, 21)[:-1]      # (20,)
# 右移半扇区 π/20 (=9°) 得中心角
angles = bound + np.pi/20
ANGLE_FEAT_CONST = torch.stack([torch.sin(angles), torch.cos(angles)], 1)  # (20,2)

def evaluate(network, env,epoch, eval_episodes,success_queue,win):
    avg_reward = 0.0
    env.end_check=False
    col = 0
    wi=0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            last_state=state
            state, reward, done, _ = env.step(last_state,a_in,0)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
        if(env.end_check):
            env.end_check=False
            wi+=1
    win.value=1.0*wi/eval_episodes
    success_queue.put((win.value,epoch))
    
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    # print("..............................................")
    # print(
    #     "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
    #     % (eval_episodes, epoch, avg_reward, avg_col)
    # )
    # print("..............................................")
    return avg_reward

class GATEncoder(nn.Module):
    def __init__(self, in_dim=7, embed_dim=128, heads=1, dropout=0.1):
        super().__init__()
        self.conv = GATv2Conv(in_dim, embed_dim, heads=heads,
                              concat=True, dropout=dropout)
        # 注意：输出维度是 embed_dim * heads

    def forward(self, data: Data):
        x = F.elu(self.conv(data.x, data.edge_index))  
        # 直接全局平均池化到图级特征： (B, embed_dim*heads)
        return global_mean_pool(x, data.batch)
    

# ── Actor：直接把 Encoder + MLP 定义在一起 ────────────────────────────
class Actor(nn.Module):
    def __init__(self, encoder: GATEncoder, 
                 feat_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.encoder    = encoder
        self.max_action = max_action
        # MLP Head
        self.fc1 = nn.Linear(feat_dim, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

        # 用于构图：常量 buffer
        self.register_buffer("edge_index", EDGE_INDEX_CONST, persistent=False)
        self.register_buffer("angle_feat", ANGLE_FEAT_CONST, persistent=False)

    @torch.no_grad()
    def _build_graph(self, state24):
        # 与你原逻辑一致的单条 / 批量图构建
        if isinstance(state24, torch.Tensor) and state24.dim() == 1:
            flat = state24
            laser = flat[:20].unsqueeze(1)
            angle = self.angle_feat
            zeros3 = torch.zeros(20, 4, device=flat.device)
            sector = torch.cat([laser, angle, zeros3], dim=1)

            robot = flat[20:].unsqueeze(0)
            zeros1 = torch.zeros(1, 3, device=flat.device)
            robot = torch.cat([zeros1, robot], dim=1)

            x = torch.cat([sector, robot], dim=0)
            batch = torch.zeros(21, dtype=torch.long, device=flat.device)
            return Data(x=x, edge_index=self.edge_index, batch=batch)

        # 批量情况
        B = state24.size(0)
        laser  = state24[:, :20].unsqueeze(-1)
        angle  = self.angle_feat.unsqueeze(0).expand(B, -1, -1)
        zeros3 = torch.zeros(B, 20, 4, device=state24.device)
        sector = torch.cat([laser, angle, zeros3], dim=-1)

        robot  = state24[:, 20:].unsqueeze(1)
        zeros1 = torch.zeros(B, 1, 3, device=state24.device)
        robot  = torch.cat([zeros1, robot], dim=-1)

        x = torch.cat([sector, robot], dim=1).view(B*21, 7)
        ei = self.edge_index.unsqueeze(0).repeat(B,1,1)
        offset = (torch.arange(B, device=state24.device)*21).view(B,1,1)
        eiB = (ei + offset).view(2, -1)
        batch = torch.repeat_interleave(torch.arange(B, device=state24.device), 21)
        return Data(x=x, edge_index=eiB, batch=batch)

    def forward(self, state):
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            data = self._build_graph(state)
        else:
            data = self._build_graph(state.to(device))
        feat = self.encoder(data)                # (B, feat_dim)
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        return self.max_action * self.tanh(self.fc3(h))
    def compute_action_from_feat(self, feat: torch.Tensor):
        """给定已经编码好的图级特征 feat，计算最终动作输出"""
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        return self.max_action * self.tanh(self.fc3(h))



# ── Critic：同样内置双 Q ────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, encoder: GATEncoder, 
                 feat_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        # 双 Q 网络
        def make_q():
            return nn.ModuleDict({
                "s1": nn.Linear(feat_dim, 800),
                "s2": nn.Linear(800, 600),
                "a" : nn.Linear(action_dim, 600),
                "out": nn.Linear(600, 1)
            })
        self.Q1 = make_q()
        self.Q2 = make_q()

        # 构图时用到的常量
        self.register_buffer("edge_index", EDGE_INDEX_CONST, persistent=False)
        self.register_buffer("angle_feat", ANGLE_FEAT_CONST, persistent=False)

    @torch.no_grad()
    def _build_graph(self, state24):
        # 同 Actor 中的实现，可抽成公用函数
        return Actor._build_graph(self, state24)

    def _q_forward(self, Q, s_feat, a):
        h_s = F.relu(Q["s1"](s_feat))
        h_s = Q["s2"](h_s)
        h_a = Q["a"](a)
        return Q["out"](F.relu(h_s + h_a))

    def forward(self, state, action):
        # 构图 & 编码
        if isinstance(state, torch.Tensor) and state.dim() in (1,2):
            data = self._build_graph(state)
        else:
            data = state  # 如果外部已组好 Batch(Data)
        s_feat = self.encoder(data)       # (B, feat_dim)

        if action.dim() == 1:
            action = action.unsqueeze(0)
        action = action.to(s_feat.device)
        q1 = self._q_forward(self.Q1, s_feat, action)
        q2 = self._q_forward(self.Q2, s_feat, action)
        return q1, q2
    def compute_Q_from_feat(self, feat, action):
        q1 = self._q_forward(self.Q1, feat, action)
        q2 = self._q_forward(self.Q2, feat, action)
        return q1, q2



class OUNoise:
    def __init__(self, action_dim=2, mu=0.0, theta=0.00006, sigma=1, sigma_min=0.05, sigma_decay_steps=30000):
        """
        初始化 OU 噪声参数
        :param action_dim: 动作的维度
        :param mu: 噪声的均值，默认值为0
        :param theta: 噪声向均值收敛的速度参数，默认值为0.2（较大以增强回归趋势）
        :param sigma: 初始噪声的波动幅度，默认值为0.2（较小以减少随机波动）
        :param sigma_min: 最小噪声波动幅度，默认值为0.05
        :param sigma_decay_steps: 噪声波动幅度的衰减步数
        """
        self.action_dim = action_dim
        self.mu = mu  # 噪声的均值
        self.theta = theta  # 噪声向均值收敛的速度
        self.sigma = sigma  # 初始噪声波动幅度
        self.sigma_min = sigma_min  # 最小噪声波动幅度
        self.sigma_decay = (sigma - sigma_min) / sigma_decay_steps  # 每步的噪声衰减量
        self.state = np.ones(self.action_dim) * self.mu  # 初始化状态

    def reset(self):
        """
        重置噪声状态为均值
        """
        self.state = np.ones(self.action_dim) * self.mu
        self.decay_sigma()  # 在每个 Episode 结束时衰减 sigma

    def evolve_state(self):
        """
        更新噪声状态
        :return: 更新后的噪声状态
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.action_dim)
        x = x + dx
        x = np.tanh(x)  # 将值压缩到 (-1, 1)
        self.state = (x + 1) / 2  # 映射到 [0, 1]
        return self.state

    def get_noise(self):
        """
        获取当前噪声值，并衰减噪声幅度
        :return: 当前噪声值
        """
        noise = self.evolve_state()

        return noise

    def decay_sigma(self):
        """
        衰减噪声的波动幅度，直到达到最小值
        """
        if self.sigma > self.sigma_min:
            self.sigma -= self.sigma_decay
            self.sigma = max(self.sigma, self.sigma_min)
            
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action,
                 embed_dim=128, heads=1):
        # 1) 构建共享 Encoder 及其 target 版
        self.encoder = GATEncoder(in_dim=7, embed_dim=embed_dim, heads=heads).to(device)
        self.encoder_target = deepcopy(self.encoder).to(device)
        
        feat_dim = embed_dim * heads
        
        # 2) Actor & Actor_target
        self.actor = Actor(self.encoder, feat_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(self.encoder_target, feat_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # 3) Critic & Critic_target
        self.critic = Critic(self.encoder, feat_dim, action_dim).to(device)
        self.critic_target = Critic(self.encoder_target, feat_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.max_action = max_action
        self.iter_count = 0

    def share_memory(self):
        self.encoder.share_memory()
        self.encoder_target.share_memory()
        self.actor.share_memory()
        self.actor_target.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()

    @torch.no_grad()
    def get_action(self, state):
        # state: (state_dim,) numpy 或 Tensor
        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        return self.actor(state_tensor).cpu().numpy().flatten()
    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        Critic_Loss_queue,
        Actor_Loss_queue,
        dccs_avg_queue,
        dccs_std_queue,
        Av_Q_queue,
        Max_Q_queue,
        counter_epoch,
        batch_size=40,
        discount=1,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        beta=0,
        Lamda=0,
        a_l=0,
    ):
        # --------------------------------------------------
        # 0. 学习率与优化器
        # --------------------------------------------------
        actor_lr  = a_l
        critic_lr = a_l
        print("actor_lr:", actor_lr)

        critic_optimizer  = torch.optim.Adam(self.critic.parameters(),  lr=critic_lr)
        actor_optimizer   = torch.optim.Adam(self.actor.parameters(),   lr=actor_lr)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=actor_lr)

        # --------------------------------------------------
        # 1. 统计变量（改为 float，避免 GPU 张量跨进程）
        # --------------------------------------------------
        av_Q      = 0.0
        max_Q     = -float("inf")
        c_av_loss = 0.0
        a_av_loss = 0.0

        for it in range(iterations):
            # 1.1 取 batch
            (batch_states, batch_actions, batch_rewards,
            batch_dones, batch_next_states,
            batch_U_s, batch_omega) = replay_buffer.sample_batch(batch_size)

            state      = torch.as_tensor(batch_states,      device=device, dtype=torch.float32)  # ★
            next_state = torch.as_tensor(batch_next_states, device=device, dtype=torch.float32)  # ★
            action     = torch.as_tensor(batch_actions,     device=device, dtype=torch.float32)  # ★
            reward     = torch.as_tensor(batch_rewards,     device=device, dtype=torch.float32)  # ★
            done       = torch.as_tensor(batch_dones,       device=device, dtype=torch.float32)  # ★
            U_s        = torch.as_tensor(batch_U_s,         device=device, dtype=torch.float32)  # ★
            omega      = torch.as_tensor(batch_omega,       device=device, dtype=torch.float32) 

            # --------------------------------------------------
            # 2. 目标 Q 计算
            # --------------------------------------------------
            data_next  = self.actor_target._build_graph(next_state)
            feat_next  = self.encoder_target(data_next)
            next_action = self.actor_target.compute_action_from_feat(feat_next)

            noise = torch.normal(
                mean=0.0,
                std=policy_noise,
                size=action.shape,
                device=device,
            ).clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target.compute_Q_from_feat(feat_next, next_action)
            target_Q  = torch.min(target_Q1, target_Q2)
            av_Q     += target_Q.mean().item()                 # ★ 立即转 float
            max_Q     = max(max_Q, target_Q.max().item())      # ★

            target_Q  = reward + ((1 - done) * discount * target_Q).detach()

            # --------------------------------------------------
            # 3. Critic 更新
            # --------------------------------------------------
            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            critic_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()
            encoder_optimizer.step()

            c_av_loss += loss.item()                           # ★ float

            # --------------------------------------------------
            # 4. Actor & Encoder 更新（每 policy_freq 步）
            # --------------------------------------------------
            if it % policy_freq == 0:
                data = self.actor._build_graph(state)
                feat = self.encoder(data)
                predicted_action = self.actor.compute_action_from_feat(feat)

                actor_grad, _ = self.critic.compute_Q_from_feat(feat, predicted_action)
                q_PF = ((1 - torch.cos(predicted_action[:, 0] - U_s))
                        + (1 - torch.cos(predicted_action[:, 1] - omega)))

                actor_grad_combined = (-(1 - Lamda) * actor_grad + Lamda * q_PF).mean()

                actor_optimizer.zero_grad()
                encoder_optimizer.zero_grad()                   # ★ 保留一次即可
                actor_grad_combined.backward()
                actor_optimizer.step()
                encoder_optimizer.step()

                a_av_loss += actor_grad_combined.item()         # ★ float

                # soft-update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # --------------------------------------------------
        # 5. 把标量写入队列（非阻塞）
        # --------------------------------------------------
        def _safe_put(q, item):
            try:                        # ★ put_nowait → 若队列满直接丢帧
                q.put_nowait(item)
            except queue.Full:
                pass

        _safe_put(Critic_Loss_queue, (c_av_loss / iterations, counter_epoch))
        _safe_put(Actor_Loss_queue,  (a_av_loss / iterations, counter_epoch))
        _safe_put(Av_Q_queue,        (av_Q / iterations,      counter_epoch))
        _safe_put(Max_Q_queue,       (max_Q,                  counter_epoch))

        self.iter_count += 1

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(self.encoder.state_dict(),
                   f"{directory}/{filename}_encoder.pth")

    def load(self, filename, directory):
        self.encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_encoder.pth")
        )
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.actor_target  .load_state_dict(self.actor.state_dict())
        self.critic_target .load_state_dict(self.critic.state_dict())

def worker(t,network,counter_epoch,counter_epoch_success,param,success_queue,reward_queue,Critic_Loss_queue,Actor_Loss_queue,Av_Q_queue,Max_Q_queue,succes_all,counter_epoch_all,timestep_worker,expl_noise_episode,reward_all,win,all_step,speed,scale_to_goal,beta,Lamda,beta_Lamda_decay,obs_scale,timestep_episode,logtest,up_speed,smoothness_scale,acceleration_scale,ounoise_decay_step,counter_epoch_collision,collision_queue,counter_epoch_timeout,timeout_queue,step_penalty,W_speed,test,dccs_avg_queue,dccs_std_queue,a_l):
   # writer = SummaryWriter(log_dir=f"runs/1worker_, reward14+nopre") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用设备:", device) 

    print("cuda数量:",torch.cuda.device_count())
    roscoreip =  str(t*3+11311)
    gazebo_port =   str(11345+3*t)
    # Set the parameters for the implementation
   # device = torch.device( "cpu")  # cuda or cpu
    seed =t*5  # Random seed number
    eval_freq = 5e3  # After how many steps to perform the evaluation
    max_ep =700  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        30000  # Number of steps over which the initial exploration noise will decay over    default:500000
    )
    expl_min = 0.0  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 512  # Size of the mini-batch
    discount = 0.9999999  # Discount factor to calculate the discounted future reward (should be close to 1)     #改环境之后，这个也要对应更改，越小的环境这个越小，之前是0.99999
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "TD3_velodyne"  # name of the file to store the policy
    save_model = True # Weather to save the model or not
    load_model = False # Weather to load a stored model
    random_near_obstacle = True # To take random actions near obstacles or not
    ou_noise = OUNoise(sigma_decay_steps=ounoise_decay_step)
    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Create the training environment
    environment_dim = 20
    robot_dim = 4
    os.environ['ROS_MASTER_URI'] = f'http://localhost:{roscoreip}'
    os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
    env = GazeboEnv("/home/l/DRL-robot-navigation-main-LZH/DRL-robot-navigation-main/catkin_ws/src/multi_robot_scenario/launch/TD2_world.launch", environment_dim,roscoreip,gazebo_port,t)
    os.environ['ROS_MASTER_URI'] = f'http://localhost:{roscoreip}'
    os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
    time.sleep(5)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1
    speed_all=0
    W_SPEED_all=0
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            network.load(file_name, "./pytorch_models")
        except:
            print(
                "Could not load the stored model parameters, initializing training with random parameters"
            )

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1
    
    count_rand_actions = 0
    random_action = []
    episode_reward=0
    episode_timesteps=0
    #choice=False
    # Begin the training loop
    action=[0,0]
    epsilon=0.5
    o_noise=0
    decay_amount=(0.5-0.1)/1500
    noise1=(0,0)
    # if(np.random.uniform(0, 1) > epsilon):
    #         env.choice=True
    # else:
    #     env. choice=False
    env.choice=True 
    env.scale_to_goal=scale_to_goal
    env.obs_scale=obs_scale
    env.smoothness_scale=smoothness_scale
    env.acceleration_scale=acceleration_scale
    env.step_penalty=step_penalty
    history_wiggle_action=[]
    wiggle_flag=True
    wiggle_len=5
    while timestep < max_timesteps:
        
        # On termination of episode
        if done:
                    ou_noise.reset()
                    print("all step:",all_step.value)
                    print("process id:",t)
                    history_wiggle_action.clear()
                    with counter_epoch_all.get_lock():
                        if(env.end_check):
                            counter_epoch_success.value+=1
                            env.end_check=False
                        # if(env.choice):
                        if(episode_timesteps >=max_ep):
                            print("timeout!!!")
                            counter_epoch_timeout.value+=1
                        if(env.collision_check):
                            counter_epoch_collision.value+=1
                            env.collision_check=False
                        counter_epoch_all.value+=1
                        
                        print("counter_epoch_all:",counter_epoch_all.value)                    
                        timestep_worker.put((timestep,counter_epoch_all.value))
                        timestep_episode.put((episode_timesteps,counter_epoch_all.value))
                        expl_noise_episode.put((ou_noise.sigma,counter_epoch_all.value))
                            # if(counter_epoch_all.value>1000):
                            #     expl_noise=0
                        # if(env.end_check):
                        #     if(env.choice):
                        #         counter_epoch_success.value+=1
                        #     env.end_check=False
                        # if(env.choice):
                        #     counter_epoch.value+=1   
                        if(counter_epoch_all.value>logtest):
                              #beta_Lamda_decay.value=0    
                              beta.value=0
                              Lamda.value=0
                        # if(counter_epoch_all.value<1000):
                        #     env.scale_to_goal=env.scale_to_goal+up_speed/1000
                        #     env.obs_scale-=0.0004 
                        if(counter_epoch_all.value%50==0):
                                #evaluate(network=network, env=env,epoch=counter_epoch_all.value, eval_episodes=eval_ep,success_queue=success_queue,win=win)
                                succes_all.value=1.0*counter_epoch_success.value/50
                                #writer.add_scalar("Success Rate",  1.0*counter_epoch_success_copy/100,counter_epoch_copy)
                                success_queue.put((1.0*counter_epoch_success.value/50,counter_epoch_all.value))
                                collision_queue.put((1.0*counter_epoch_collision.value/50,counter_epoch_all.value))
                                timeout_queue.put((1.0*counter_epoch_timeout.value/50,counter_epoch_all.value))
                                counter_epoch_collision.value=0
                                counter_epoch_timeout.value=0
                                counter_epoch_success.value=0
                               # counter_epoch_success.value=0
                            
                        if(episode_timesteps!=0 and env.choice):    #这里记录的是真正的自己的奖励，而不是人工市场法的
                            #writer.add_scalar("Rward", episode_reward/episode_timesteps, counter_epoch_copy)
                            reward_queue.put((1.0*episode_reward/episode_timesteps,counter_epoch_all.value))
                            reward_all.put((episode_reward,counter_epoch_all.value))
                            speed.put((speed_all/episode_timesteps,counter_epoch_all.value))    
                            W_speed.put((W_SPEED_all/episode_timesteps,counter_epoch_all.value)) 
                            speed_all=0
                            W_SPEED_all=0
                        #print("oooooooooooooooooooooooooo")
                        if timestep != 0:
                    #        print("ttttttttttttttttttttttttttttttttttttttt")
                            network.train(
                                replay_buffer,
                                episode_timesteps,
                                #writer,
                                Critic_Loss_queue,
                                Actor_Loss_queue,
                                dccs_avg_queue,
                                dccs_std_queue,
                                Av_Q_queue,
                                Max_Q_queue,
                                counter_epoch_all.value,
                                batch_size,
                                discount,
                                tau,
                                policy_noise,
                                noise_clip,
                                policy_freq,
                                beta.value,
                                Lamda.value,
                                a_l
                            )
    
                    if timesteps_since_eval >= eval_freq:
                        #print("Validating")
                        timesteps_since_eval %= eval_freq
                        # evaluations.append(
                        #     evaluate(network=network, env=env,epoch=epoch, eval_episodes=eval_ep)
                        # )
                        network.save(file_name+str(test), directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations)
                        epoch += 1

                    state = env.reset()
                    print(11111111)
                    # 先增加一个新维度，将形状变为 (1, state_dim)
             #       state = np.tile(state, 1) 
                    #env.random_target()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                    # if(np.random.uniform(0, 1) > epsilon):
                    #     env.choice=True
                    # else:
                    #     env.choice=False 
                    env.choice=True
                    if timestep==0:
                        env.choice=True
                    if(win.value>0.3):
                        epsilon=0
                    else:  
                        epsilon-=decay_amount
                        #epsilon=max(0.1,epsilon)

        
        # add some exploration noise
        if expl_noise > expl_min:
            expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
            expl_noise=max(0,expl_noise)
         #   ou_noise.decay_sigma()
        print(33333333333)
        p=env.calculate_attraction_force(env.last_odom.pose.pose.position.x,env.last_odom.pose.pose.position.y,env.goal_x,env.goal_y,650)
        f=env.calculate_total_repulsion_force(env.last_odom.pose.pose.position.x,env.last_odom.pose.pose.position.y,math.sqrt(p[0]**2+p[1]**2))
        z=(f[0]+p[0],f[1]+p[1])
        v = math.sqrt(z[0]**2 + z[1]**2)
        v=min(v,0.5)
        v=max(v,0)
        omega = math.atan2(z[1],z[0])
        #print("angel1:",omega)
        #print("angle2:",env.angle)
        omega=env.angle-omega
        
        if(omega>3.1):
            omega=6.2-omega
            omega=-omega
        if(omega<-3.1):
            omega=omega+6.2
        omega1=omega
        omega=min(omega,1)
        omega=max(omega,-1)
        omega=-omega
        # l=abs(omega)/math.pi*2
        # l=abs(v)-l
        # if(v<0):
        #     v=-l
        # else:
        #      v=l
        if(True):
           # print("mode predict")
            #print("model predict")
            print(444444444444444444444444444)
            action = network.get_action(np.array(state))
        #    print("model:",action)
          #  noise1=ou_noise.get_noise()
        #     action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        #      -max_action, max_action
        #  )
        #     o_noise=ou_noise.get_noise()
            action = (action + np.random.normal(0, ou_noise.get_noise(), size=action_dim)).clip(
             -max_action, max_action
         )  
        #     action = (action + noise1).clip(
        #     -max_action, max_action
        # )
        # If the robot is facing an obstacle, randomly force it to take a consistent random action.
        # This is done to increase exploration in situations near obstacles.
        # Training can also be performed without it
            if random_near_obstacle:
                if (
                    np.random.uniform(0, 1) > 0.85
                    and min(state[4:-8]) < 0.6
                    and count_rand_actions < 1
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
                    action[0] = -1
        else:
            #print("artificial")
       
            action[0]=(v*2-1)    #人工市场法需要根据环境来改参数，这里的0.5和1.5是simple环境需要的，还有上面的算引力的32也是，原先是256或者128
            action[1]=omega
        
        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        
        a_in = [(action[0]+1)/2, action[1]]
        speed_all+=a_in[0]
        W_SPEED_all=a_in[1]
     #   print(a_in)
        #print(a_in)
        print(55555555555555555555555555)
        
     #   print("artificial:",v*2-1,omega)    
        next_state, reward, done, target = env.step(state,a_in,omega1)
        print(666666666666666666)
        # 先增加一个新维度，将形状变为 (1, state_dim)
       # next_state = np.concatenate((state, next_state))
      #  next_state=next_state[state_dim:]
        # if(env.choice):
        #     if(len(history_wiggle_action)==wiggle_len ):
        #         for j in range (0,len(history_wiggle_action)-1):
        #             if(history_wiggle_action[j][1]*history_wiggle_action[j+1][1]>0):
        #                 wiggle_flag=False
        #                 break
        #         if(wiggle_flag):
        #             env.choice=False
        #         else:
        #             wiggle_flag=True
        #         history_wiggle_action.clear()
        #     else:
        #         history_wiggle_action.append(a_in)

        done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        episode_reward += reward

        # Save the tuple in replay buffer
        replay_buffer.add(state, action, reward, done_bool, next_state,v*2-1,omega)

        # Update the counters
        state = next_state
        episode_timesteps += 1
        if(env.choice):
            all_step.value+=1
            timestep += 1
        timesteps_since_eval += 1

    # After the training is done, evaluate the network and save it
    # evaluations.append(evaluate(network=network, env=env,epoch=epoch, eval_episodes=eval_ep))
    if save_model:
        network.save("%s" % file_name, directory="./models")
    np.save("./results/%s" % file_name, evaluations)


def logger_process(k,data_queue, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    while True:
        try:
            item = data_queue.get(timeout=10)
            if item == "END":  # 检测到结束标志，退出循环
                break
            success_rate, epoch = item
            if(k==0):
                writer.add_scalar("Reward", success_rate, epoch)
      #          print("11111111111111111111111111111")
            if (k==1):
                writer.add_scalar("Success Rate", success_rate, epoch)
          #      print("333333333333333333333333333333")
            if(k==2):
                writer.add_scalar("Critic-Loss", success_rate, epoch)
          #      print("4444444444444444444444444444")
            if(k==3):
          #      print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz:",epoch)
                writer.add_scalar("Actor-Loss", success_rate, epoch)
          #      print("5555555555555555555555555555555555") 
            if(k==4):
                writer.add_scalar("Av_Q", success_rate, epoch)
          #      print("666666666666666666666666666666666")
            if(k==5):
                writer.add_scalar("Max-Q", success_rate, epoch)
            #    print("777777777777777777777777777777777")   
            if(k==6):
                writer.add_scalar("time_steps", success_rate, epoch)
            #    print("88888888888888888888888888888888")   
            if(k==7):
                writer.add_scalar("expl_noise", success_rate, epoch)
          #      print("9999999999999999999999999999999999")   
            if(k==8):
                writer.add_scalar("Episode Rward", success_rate, epoch)
           #     print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")   
            if(k==9):
                writer.add_scalar("episode step",success_rate,epoch)
            if(k==10):
                writer.add_scalar("linear speed",success_rate,epoch)
            if(k==11):
                writer.add_scalar("collision",success_rate,epoch)
            if(k==12):
                writer.add_scalar("timeout",success_rate,epoch)
            if(k==13):
                writer.add_scalar("angular_speed",success_rate,epoch)
            if(k==14):
                writer.add_scalar("dccs_avg_queue",success_rate,epoch)
            if(k==15):
                writer.add_scalar("dccs_std_queue",success_rate,epoch)
        except queue.Empty:  # 捕获 queue.Empty 异常
            continue
    writer.close()
    
if __name__ == '__main__':
    mp.set_start_method('spawn')

    scale_to_goal = 18
    obs_scale=0.5
    environment_dim = 20
    robot_dim = 4
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1
    MasterNode = TD3(state_dim, action_dim, max_action) #A
   # MasterNode.share_memory() #B
    counter_epoch=mp.Value('i',0)
    succes_all=mp.Value('d',0.0)
    win=mp.Value('d',0.0)
    counter_epoch_all=mp.Value('i',0)
    counter_epoch_success=mp.Value('i',0)
    all_step=mp.Value('i',0)
    beta = mp.Value('d', 0.0)  # 双精度浮点数
    Lamda = mp.Value('d', 0.0)
    beta_Lamda_decay = mp.Value('d', 0.0)
    counter_epoch_timeout=mp.Value('i',0)
    counter_epoch_collision=mp.Value('i',0)
    beta.value=1
    Lamda.value=1
    if len(sys.argv) > 1:
        # 获取传递的参数并转换为浮点数
        scale_to_goal = float(sys.argv[1])  # 使用 float() 获取浮点数值
        obs_scale = float(sys.argv[2])
        logtest = float(sys.argv[3])
        test = float(sys.argv[4])
        up_speed=float(sys.argv[5])
        smoothness_scale=float(sys.argv[6])
        acceleration_scale=float(sys.argv[7])
        ounoise_decay_step=float(sys.argv[8])
        step_penalty=float(sys.argv[9])
        a_l=float(sys.argv[10])
       # decay_steps = float(sys.argv[3])  # decay_steps 也改为浮点数

       # beta_Lamda_decay.value = beta.value*1.0 / decay_steps  # 计算结果也是浮点数
      #  beta_Lamda_decay.value=0
    #print("dsadsadsa:",beta.value,Lamda.value,decay_steps,beta_Lamda_decay.value)
    logdir = f"./graph/max_speed_1_collsion_dist_0.45_k_250_threshold_0.45_31/reward+{scale_to_goal}+{obs_scale}+{logtest}+{test}+{up_speed}+{smoothness_scale}+{acceleration_scale}+{ounoise_decay_step}+{step_penalty}+{a_l}"
    processes=[]
    params={'n_workers':1,
            'n_loggers':2,
            }
    success_queue = mp.Queue()
    reward_queue = mp.Queue()
    Critic_Loss_queue=mp.Queue()
    Actor_Loss_queue=mp.Queue()
    Max_Q_queue=mp.Queue()
    Av_Q_queue=mp.Queue()
    timeout_queue=mp.Queue()
    collision_queue=mp.Queue()
    loss_queue = mp.Queue()
    timestep_worker = mp.Queue()
    timestep_episode = mp.Queue()
    expl_noise_episode = mp.Queue()
    reward_all=mp.Queue()
    speed=mp.Queue()  
    W_speed=mp.Queue()  
    dccs_avg_queue=mp.Queue()
    dccs_std_queue=mp.Queue()
    logger = mp.Process(target=logger_process, args=(0,reward_queue, logdir))
    logger.start()
    processes.append(logger)
    
    logger1 = mp.Process(target=logger_process, args=(1,success_queue, logdir))
    logger1.start()
    processes.append(logger1)   

    logger2 = mp.Process(target=logger_process, args=(2,Critic_Loss_queue, logdir))
    logger2.start()
    processes.append(logger2) 
    
    logger3 = mp.Process(target=logger_process, args=(3,Actor_Loss_queue, logdir))
    logger3.start()
    processes.append(logger3) 
    
    logger4 = mp.Process(target=logger_process, args=(4,Av_Q_queue, logdir))
    logger4.start()
    processes.append(logger4) 
    
    logger5 = mp.Process(target=logger_process, args=(5,Max_Q_queue, logdir))
    logger5.start()
    processes.append(logger5) 
    
    logger6 = mp.Process(target=logger_process, args=(6,timestep_worker, logdir))
    logger6.start()
    processes.append(logger6)
    
    logger7 = mp.Process(target=logger_process, args=(7,expl_noise_episode, logdir))
    logger7.start()
    processes.append(logger7)
    
    logger8 = mp.Process(target=logger_process, args=(8,reward_all, logdir))
    logger8.start()
    processes.append(logger8)

    logger9 = mp.Process(target=logger_process, args=(9,timestep_episode, logdir))
    logger9.start()
    processes.append(logger9)

    logger10 = mp.Process(target=logger_process, args=(10,speed, logdir))
    logger10.start()
    processes.append(logger10)

    logger11 = mp.Process(target=logger_process, args=(11,collision_queue, logdir))
    logger11.start()
    processes.append(logger11)
    
    logger12 = mp.Process(target=logger_process, args=(12,timeout_queue, logdir))
    logger12.start()
    processes.append(logger12)
    
    logger13 = mp.Process(target=logger_process, args=(13,W_speed, logdir))
    logger13.start()
    processes.append(logger13)

    logger14 = mp.Process(target=logger_process, args=(14,dccs_avg_queue, logdir))
    logger14.start()
    processes.append(logger14)

    logger15 = mp.Process(target=logger_process, args=(15,dccs_std_queue, logdir))
    logger15.start()
    processes.append(logger15)
    for i in range(params["n_workers"]):
            p = mp.Process(target=worker, args=(i, MasterNode, counter_epoch, counter_epoch_success, params, success_queue, reward_queue, Critic_Loss_queue, Actor_Loss_queue, Av_Q_queue, Max_Q_queue, succes_all, counter_epoch_all, timestep_worker, expl_noise_episode, reward_all, win, all_step,speed,scale_to_goal,beta,Lamda,beta_Lamda_decay,obs_scale,timestep_episode,logtest,up_speed,smoothness_scale,acceleration_scale,ounoise_decay_step,counter_epoch_collision,collision_queue,counter_epoch_timeout,timeout_queue,step_penalty,W_speed,test,dccs_avg_queue,dccs_std_queue,a_l))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()
    for p in processes:
            p.terminate()