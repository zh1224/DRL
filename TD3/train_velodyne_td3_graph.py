# -*- coding: utf-8 -*

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
from velodyne_env_graph import GazeboEnv
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
# —— 完全图 edge_index：21×21，只生成一次 —— 
# =============================================================

# ────────────────── 全局常量 ──────────────────
N_NODES = 21                                         # 20 LiDAR + 1 Robot
row = torch.arange(N_NODES).repeat(N_NODES)
col = torch.arange(N_NODES).repeat_interleave(N_NODES)
EDGE_INDEX_CONST = torch.stack([row, col], dim=0)    # (2, 441)

# 20 条分界：-90° … +90°
bound = torch.linspace(-np.pi/2-0.03, np.pi/2, 21)[:-1]      # (20,)
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


# ── Actor with nested Encoder ──────────────────
class Actor(nn.Module):
    # ---------- 嵌套的 GAT 编码器 ----------
    class _Encoder(nn.Module):
        def __init__(self, embed_dim=32, heads=4):
            super().__init__()
            self.g1 = GATv2Conv(7, 64, heads=heads, concat=True)
            self.g2 = GATv2Conv(64*heads, embed_dim, heads=1, concat=False)

        def forward(self, data: Data, flatten=False):
            
            x = F.elu(self.g1(data.x, data.edge_index))
            x = self.g2(x, data.edge_index)              # (N_total, embed_dim)

            if flatten:
                B = int(data.batch.max()) + 1            # 图数
           #     print("B:",x.view(B, -1))
                return x.view(B, -1)                     # (B, 21*embed_dim)
            else:
                return global_mean_pool(x, data.batch)   # (B, embed_dim)

    # ---------- Actor 本体 ----------
    def __init__(self, embed_dim=64, act_dim=2,
                 max_range=10.0, device="cpu"):
        super().__init__()
        self.device     = torch.device(device)
        self.max_range  = max_range

        # 1) 注册常量
        self.register_buffer("edge_index", EDGE_INDEX_CONST,  persistent=False)
        self.register_buffer("angle_feat", ANGLE_FEAT_CONST,  persistent=False)

        # 2) 嵌入式 GAT 编码器
        self.encoder = Actor._Encoder(embed_dim)
        f_in=embed_dim
        # 3) MLP Head
               # 21 * embed_dim
        self.fc1  = nn.Linear(f_in, 256)
        self.fc2  = nn.Linear(256, 256)
        self.fc3  = nn.Linear(256, act_dim)
        self.tanh = nn.Tanh()

    # -------- （单/批量）构图函数，同前 --------
    @torch.no_grad()
    def _build_single(self, state24):
        # (24,) → Data
        laser  = (state24[:20] / self.max_range).unsqueeze(1)
        angle  = self.angle_feat
        zeros3 = torch.zeros(20, 4, device=self.device)
        sector = torch.cat([laser, angle, zeros3], dim=1)

        robot  = state24[20:].unsqueeze(0)
        zeros1 = torch.zeros(1, 3, device=self.device)
        robot  = torch.cat([zeros1, robot], dim=1)

        x = torch.cat([sector, robot], dim=0)
        batch_vec = torch.zeros(21, dtype=torch.long, device=self.device)
        return Data(x=x, edge_index=self.edge_index, batch=batch_vec)

    @torch.no_grad()
    def _build_batch(self, flat):
        B = flat.size(0)
        laser  = (flat[:, :20] / self.max_range).unsqueeze(-1)
        angle  = self.angle_feat.unsqueeze(0).expand(B, -1, -1)
        zeros3 = torch.zeros(B, 20, 4, device=self.device)
        sector = torch.cat([laser, angle, zeros3], dim=-1)

        robot  = flat[:, 20:].unsqueeze(1)
        zeros1 = torch.zeros(B, 1, 3, device=self.device)
        robot  = torch.cat([zeros1, robot], dim=-1)

        x = torch.cat([sector, robot], dim=1).reshape(B*21, 7)
        ei  = self.edge_index.unsqueeze(0).repeat(B,1,1)
        offset = (torch.arange(B, device=self.device)*21).view(B,1,1)
        eiB = (ei + offset).reshape(2, -1)
        batch_vec = torch.repeat_interleave(torch.arange(B, device=self.device), 21)
        return Data(x=x, edge_index=eiB, batch=batch_vec)

    @torch.no_grad()
    def _build_graph(self, state24):
        if isinstance(state24, np.ndarray):
            state24 = torch.from_numpy(state24).float()
        state24 = state24.to(self.device)
        if state24.dim() == 1:
            return self._build_single(state24)
        else:
            return self._build_batch(state24)

    # -------- 前向 --------
    def forward(self, state24):
        graph = self._build_graph(state24)
        #print("Data:",graph)
     #   print("angel_feat:",ANGLE_FEAT_CONST)
        nfv   = self.encoder(graph)                 # (B, 21*embed_dim)
        h = F.relu(self.fc1(nfv))
        h = F.relu(self.fc2(h))
        return self.tanh(self.fc3(h))               # (B, act_dim)




class Critic(nn.Module):
    class _Encoder(nn.Module):
        def __init__(self, embed_dim=32, heads=4):
            super().__init__()
            self.g1 = GATv2Conv(7, 128, heads=heads, concat=True)
            self.g2 = GATv2Conv(128*heads, embed_dim, heads=1, concat=False)

        def forward(self, data: Data, flatten=True):
            x = F.elu(self.g1(data.x, data.edge_index))
            x = self.g2(x, data.edge_index)              # (N_total, embed_dim)

            if flatten:
                B = int(data.batch.max()) + 1            # 图数
                return x.view(B, -1)                     # (B, 21*embed_dim)
            else:
                return global_mean_pool(x, data.batch)   # (B, embed_dim)

    """TD3 双 Q 网络，状态经 GAT 编码"""
    def __init__(self,
                 act_dim: int,
                 embed_dim: int = 32,
                 max_range: float = 10.0,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device     = torch.device(device)
        self.max_range  = max_range

        # ── 图常量 ───────────────────────────────────
        self.register_buffer("edge_index", EDGE_INDEX_CONST,  persistent=False)
        self.register_buffer("angle_feat", ANGLE_FEAT_CONST,  persistent=False)

        # ── 共享 GAT 编码器 ──────────────────────────
        self.encoder = Critic._Encoder(embed_dim)

        # ── 两条 Q-net ──────────────────────────────
        def q_net():
            return nn.Sequential(
                nn.Linear(embed_dim + act_dim, 800), nn.ReLU(),
                nn.Linear(800, 600),                 nn.ReLU(),
                nn.Linear(600, 1)
            )
        self.Q1, self.Q2 = q_net(), q_net()

    # ---------- 单图构图 ----------
    @torch.no_grad()
    def _build_single(self, state24_t: torch.Tensor) -> Data:
        laser  = (state24_t[:20] / self.max_range).unsqueeze(1)       # (20,1)
        angle  = self.angle_feat                                      # (20,2)
        zeros3 = torch.zeros(20, 4, device=self.device)
        sector = torch.cat([laser, angle, zeros3], dim=1)             # (20,7)

        robot  = state24_t[20:].unsqueeze(0)                          # (1,4)
        zeros1 = torch.zeros(1, 3, device=self.device)
        robot  = torch.cat([zeros1, robot], dim=1)                    # (1,7)

        x = torch.cat([sector, robot], dim=0)                         # (21,7)
        batch_vec = torch.zeros(21, dtype=torch.long, device=self.device)
        return Data(x=x, edge_index=self.edge_index, batch=batch_vec)

    # ---------- 批量构图 ----------
    @torch.no_grad()
    def _build_batch(self, flat_t: torch.Tensor) -> Data:
        B = flat_t.size(0)
        laser  = (flat_t[:, :20] / self.max_range).unsqueeze(-1)           # (B,20,1)
        angle  = self.angle_feat.unsqueeze(0).expand(B, -1, -1)            # (B,20,2)
        zeros3 = torch.zeros(B, 20, 4, device=self.device)
        sector = torch.cat([laser, angle, zeros3], dim=-1)                 # (B,20,7)

        robot  = flat_t[:, 20:].unsqueeze(1)                               # (B,1,4)
        zeros1 = torch.zeros(B, 1, 3, device=self.device)
        robot  = torch.cat([zeros1, robot], dim=-1)                        # (B,1,7)

        x = torch.cat([sector, robot], dim=1).reshape(B * 21, 7)
        ei  = self.edge_index.unsqueeze(0).repeat(B, 1, 1)
        offset = (torch.arange(B, device=self.device) * 21).view(B, 1, 1)
        eiB = (ei + offset).reshape(2, -1)
        batch_vec = torch.repeat_interleave(torch.arange(B, device=self.device), 21)
        return Data(x=x, edge_index=eiB, batch=batch_vec)

    # ---------- 统一调度 ----------
    @torch.no_grad()
    def _build_graph(self, state24):
        if isinstance(state24, np.ndarray):
            state24 = torch.from_numpy(state24).float()
        state24 = state24.to(self.device)
        if state24.dim() == 1:
            return self._build_single(state24)
        else:
            return self._build_batch(state24)

    # ---------- forward ----------
    def forward(self, state, act):
        """
        state : (24,) or (B,24) or Data/Batch
        act   : (act_dim,) or (B, act_dim)
        """
        # 1) 把 state 编码成 embed
        if isinstance(state, (Data, Batch)):
            z = self.encoder(state)               # (B, embed_dim)
        else:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            if state.dim() == 1:
                state = state.unsqueeze(0)        # (1,24)
            graph = self._build_graph(state)      # Data/Batch
            z = self.encoder(graph)               # (B, embed_dim)

        # 2) 拼接动作
        if act.dim() == 1:
            act = act.unsqueeze(0)
        sa = torch.cat([z, act.to(z.device)], dim=-1)

        # 3) 双 Q 输出
        return self.Q1(sa), self.Q2(sa)
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
            
            
device = torch.device( "cpu")  # cuda or cpu
# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(64,action_dim).to(device)
        self.actor_target = Actor(64,action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
 

        # Initialize the Critic networks
        self.critic = Critic(16,action_dim).to(device)
        self.critic_target = Critic(16,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    
        self.max_action = max_action
     #   self.writer = SummaryWriter()
        self.iter_count = 0
    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.actor_target.share_memory()
        self.critic_target.share_memory()
    @torch.no_grad()
    def get_action(self, state24, noise_std: float = 0.0):
        """
        参数
        ----
        state24   : (24,)  numpy.ndarray 或 torch.Tensor(float32)
        noise_std : 探索噪声标准差；0 表示无噪声

        返回
        ----
        action_np : (act_dim,) numpy.ndarray
        """
        # ---------- 1) numpy → Tensor，搬到 device ----------
        if isinstance(state24, np.ndarray):
            state_t = torch.from_numpy(state24).float()
        else:                               # 已是 Tensor
            state_t = state24.float()


        state_t = state_t.to(device, non_blocking=True)

        # ---------- 2) 前向推理 (Actor 内部会自动 build_graph) ----------
        action_t = self.actor(state_t).squeeze(0)        # (act_dim,)

     
        # ---------- 4) 回到 NumPy ----------
        return action_t.cpu().numpy()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        #writer,
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
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
        beta=0,
        Lamda=0,
        a_l=0,
    ):
        

        # dccs_window = iterations          # 滑动窗口 W
        # dccs_buf    = deque(maxlen=dccs_window)
        # eps_dccs    = 0.02         # 稳定阈值 ε_q
        # critic_stable = False      # 标志位
        # lambda_bc   = 1.0          # 初始 λ_BC，可放到参数表
        # decay_rate  = 0.995        # 每 step 衰减系数
        # min_lambda  = 0.02         # 衰减下限


        critic_lr = 5e-2 # Critic的学习率
        actor_lr = a_l # Actor的学习率
        print("actor_lr:",actor_lr)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        av_Q = 0
        max_Q = -inf
        c_av_loss = 0
        a_av_loss = 0
        for it in range(min(int(iterations),1)):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
                batch_U_s,
                batch_omega,
            ) = replay_buffer.sample_batch(batch_size)
            # state      = Batch.from_data_list(batch_states).to(device)
            # next_state = Batch.from_data_list(batch_next_states).to(device)

            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            # ★ 仅此一次，把 (B,24) → 图
            # state   = build_graph_batch(batch_states,  10.0, ANGLE_FEAT_CONST, EDGE_INDEX_CONST, device)
            # next_state  = build_graph_batch(batch_next_states, 10.0, ANGLE_FEAT_CONST, EDGE_INDEX_CONST, device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            U_s=torch.Tensor(batch_U_s).to(device)
            omega=torch.Tensor(batch_omega).to(device)

        #     # Obtain the estimated action from the next state by using the actor-target
        #     with torch.inference_mode():          # 更彻底 than no_grad
        #      next_action = self.actor_target(next_state)

        #     # Add noise to the action
        #     noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
        #     noise = noise.clamp(-noise_clip, noise_clip)
        #     next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        #     # Calculate the Q values from the critic-target network for the next state-action pair
        #     target_Q1, target_Q2 = self.critic_target(next_state, next_action)

        #     # Select the minimal Q value from the 2 calculated values
        #     target_Q = torch.min(target_Q1, target_Q2)
        #     av_Q += torch.mean(target_Q)
        #     max_Q = max(max_Q, torch.max(target_Q))
        #     # Calculate the final Q value from the target network parameters by using Bellman equation
        #     target_Q = reward + ((1 - done) * discount * target_Q).detach()

        #     # Get the Q values of the basis networks with the current parameters
        #     current_Q1, current_Q2 = self.critic(state, action)


        #     # with torch.no_grad():
        #     #     dccs_batch = torch.mean(torch.abs(current_Q1 - current_Q2)).item()
        #     #   #  print(dccs_batch)
        #     #     dccs_buf.append(dccs_batch)        # 推入滑窗

            
        #     #q_PF = q_PF.view(-1, 1)
        #    #print(f"q_PF shape: {q_PF.shape}")
        #     #print("action:",action)
        #     #print("q_PF",action[:,0],U_s,action[:,1],omega,q_PF)
        #   #  F_C_Ei = F.mse_loss(current_Q1, q_PF) + F.mse_loss(current_Q2, q_PF)
        #     # Calculate the loss between the current Q value and the target Q value
        #     critic_sub_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #     #loss = (1-beta)*critic_sub_loss+beta*F_C_Ei
        #     loss = critic_sub_loss
        #     # Perform the gradient descent
        #     critic_optimizer.zero_grad()
        #     loss.backward()
        #     critic_optimizer.step()

            # # ======= 新增：Critic 稳定性判定 =======
            # if len(dccs_buf) == dccs_window:
            #     dccs_avg = sum(dccs_buf) / dccs_window
            #     dccs_std = (sum((x - dccs_avg) ** 2 for x in dccs_buf) / dccs_window) ** 0.5
            #     if dccs_avg<0:
            #         print("111111111111111111111111111111111111111111111",dccs_avg)
            #     dccs_avg_queue.put(( (1.0*dccs_avg), counter_epoch))
            #     dccs_std_queue.put(( (1.0*dccs_std), counter_epoch))
               # Critic_Loss_queue.put()
            #    dccs_std_queue,
                # 判定一次即可，将标志置真
                # if (dccs_avg < eps_dccs) and (dccs_std < eps_dccs * 0.5):
                #     critic_stable = True

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
              #  U_s = U_s.unsqueeze(1)  # 变为 torch.Size([40, 1])
              #  omega = omega.unsqueeze(1)  # 变为 torch.Size([40, 1])
    # 将它们沿着列方向（dim=1）拼接成 [40, 2] 的张量
              #  combined_action = torch.cat((U_s, omega), dim=1)
                predicted_action=self.actor(state)

               # actor_grad, _ = self.critic(state, predicted_action)
                q_PF = ((1 - torch.cos(predicted_action[:, 0] - U_s)) + (1 - torch.cos(predicted_action[:, 1] - omega)))
                #actor_grad=actor_grad.detach()
                #actor_grad = -(1-Lamda)*actor_grad.mean()+Lamda*F.mse_loss(predicted_action,combined_action)
                # 计算L2范数
                # actor_grad_norm = torch.norm(actor_grad, p=2)  # actor_grad的L2范数
                # q_PF_norm = torch.norm(q_PF, p=2)  # q_PF的L2范数

                # # 归一化梯度
                # if actor_grad_norm > 0:  # 避免除零错误
                #     actor_grad_normalized = actor_grad / actor_grad_norm
                # else:
                #     actor_grad_normalized = actor_grad  # 如果梯度为零，保持不变

                # if q_PF_norm > 0:  # 避免除零错误
                #     q_PF_normalized = q_PF / q_PF_norm
                # else:
                #     q_PF_normalized = q_PF  # 如果梯度为零，保持不变
                # 计算余弦相似度
                # cos_sim = torch.cosine_similarity(actor_grad_normalized, q_PF_normalized, dim=0)
                # cos_sim = cos_sim.mean()  # 对批量数据取均值，得到标量
                # # 根据阈值选择更新策略
                # if cos_sim < 0:  # 方向不一致
                #     actor_grad_combined = q_PF_normalized.mean()  # 优先采用TD3，并转换为标量
                # else:
                actor_grad_combined = (Lamda * q_PF).mean()  # 结合APF和TD3，并转换为标量
                #actor_grad_combined = q_PF_normalized.mean()
                #actor_grad = -(1-Lamda)*actor_grad_normalized.mean()+Lamda*q_PF_normalized.mean()
                actor_optimizer.zero_grad()
                actor_grad_combined.backward()
                  # ▶▶▶ 在这行和 actor_opt.step() 之间插入打印
                # print("enc grad mean:",
                #     sum(p.grad.abs().mean().item() 
                #         for p in self.actor.encoder.parameters()))
                actor_optimizer.step()

                # del state,next_state,action,reward,done,U_s,omega
                # torch.cuda.empty_cache()      # 仅清掉未用缓存，不影响速度
                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                
      #      c_av_loss += loss
            a_av_loss+=actor_grad_combined
        self.iter_count += 1
       # gc.collect()
        # Write new values for tensorboard
        # Critic_Loss_queue.put(( (1.0*c_av_loss / iterations).detach(), counter_epoch))
        Actor_Loss_queue.put(( (1.0*a_av_loss / iterations).detach(), counter_epoch))
        # Av_Q_queue.put(( (1.0*av_Q / iterations).detach(), counter_epoch))
        # Max_Q_queue.put(( (1.0*max_Q).detach(), counter_epoch))


        
        # writer.add_scalar("Critic-Loss", c_av_loss / iterations, counter_epoch)
        # writer.add_scalar("Actor-Loss", a_av_loss / iterations,counter_epoch)
        # writer.add_scalar("Av. Q", av_Q / iterations, counter_epoch)
        # writer.add_scalar("Max. Q", max_Q, counter_epoch)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )

def worker(t,network,counter_epoch,counter_epoch_success,param,success_queue,reward_queue,Critic_Loss_queue,Actor_Loss_queue,Av_Q_queue,Max_Q_queue,succes_all,counter_epoch_all,timestep_worker,expl_noise_episode,reward_all,win,all_step,speed,scale_to_goal,beta,Lamda,beta_Lamda_decay,obs_scale,timestep_episode,logtest,up_speed,smoothness_scale,acceleration_scale,ounoise_decay_step,counter_epoch_collision,collision_queue,counter_epoch_timeout,timeout_queue,step_penalty,W_speed,test,dccs_avg_queue,dccs_std_queue,a_l):
   # writer = SummaryWriter(log_dir=f"runs/1worker_, reward14+nopre") 

    roscoreip =  str(t*3+11311)
    gazebo_port =   str(11345+3*t)
    # Set the parameters for the implementation
    device = torch.device( "cpu")  # cuda or cpu
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
    batch_size = 32  # Size of the mini-batch
    discount = 0.9999999  # Discount factor to calculate the discounted future reward (should be close to 1)     #改环境之后，这个也要对应更改，越小的环境这个越小，之前是0.99999
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e3  # Maximum size of the buffer
    file_name = "TD3_velodyne"  # name of the file to store the policy
    save_model = True # Weather to save the model or not
    load_model = False # Weather to load a stored model
    random_near_obstacle = False # To take random actions near obstacles or not
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
            action = network.get_action(state)
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
        
        
     #   print("artificial:",v*2-1,omega)    
        next_state, reward, done, target = env.step(state,a_in,omega1)
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
    MasterNode.share_memory() #B
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
    logdir = f"./graph/max_speed_1_collsion_dist_0.45_k_250_threshold_0.45_4/reward+{scale_to_goal}+{obs_scale}+{logtest}+{test}+{up_speed}+{smoothness_scale}+{acceleration_scale}+{ounoise_decay_step}+{step_penalty}+{a_l}"
    processes=[]
    params={'n_workers':20,
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
