import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import queue

import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import torch.multiprocessing as mp #A
from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv


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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


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
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2
class OUNoise:
    def __init__(self, action_dim=2, mu=0.0, theta=0.15, sigma=1, sigma_min=0.1, sigma_decay_steps=50000):
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

    def evolve_state(self):
        """
        更新噪声状态
        :return: 更新后的噪声状态
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self):
        """
        获取当前噪声值，并衰减噪声幅度
        :return: 当前噪声值
        """
        noise = self.evolve_state()
        self.decay_sigma()
        return noise

    def decay_sigma(self):
        """
        衰减噪声的波动幅度，直到达到最小值
        """
        if self.sigma > self.sigma_min:
            self.sigma -= self.sigma_decay
            
            
device = torch.device( "cpu")  # cuda or cpu
# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
 

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    
        self.max_action = max_action
     #   self.writer = SummaryWriter()
        self.iter_count = 0
    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.actor_target.share_memory()
        self.critic_target.share_memory()
    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        #writer,
        Critic_Loss_queue,
        Actor_Loss_queue,
        Av_Q_queue,
        Max_Q_queue,
        counter_epoch,
        batch_size=40,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        print("11111111111111111111111111111111111111111111111111111111train parameter:",batch_size,discount)
        critic_optimizer = torch.optim.Adam(self.critic.parameters())
        actor_optimizer = torch.optim.Adam(self.actor.parameters())
        av_Q = 0
        max_Q = -inf
        c_av_loss = 0
        a_av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                actor_optimizer.zero_grad()
                actor_grad.backward()
                actor_optimizer.step()

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

            c_av_loss += loss
            a_av_loss+=actor_grad
        self.iter_count += 1
        # Write new values for tensorboard
        Critic_Loss_queue.put(( (1.0*c_av_loss / iterations).detach(), counter_epoch))
        Actor_Loss_queue.put(( (1.0*a_av_loss / iterations).detach(), counter_epoch))
        Av_Q_queue.put(( (1.0*av_Q / iterations).detach(), counter_epoch))
        Max_Q_queue.put(( (1.0*max_Q).detach(), counter_epoch))
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

def worker(t,network,counter_epoch,counter_epoch_success,param,success_queue,reward_queue,Critic_Loss_queue,Actor_Loss_queue,Av_Q_queue,Max_Q_queue,succes_all,counter_epoch_all,timestep_episode,expl_noise_episode,reward_all,win):
   # writer = SummaryWriter(log_dir=f"runs/1worker_, reward14+nopre") 
   
    roscoreip = "1131" + str(t+1)
    gazebo_port = "1134" + str(t+4)
    # Set the parameters for the implementation
    device = torch.device( "cpu")  # cuda or cpu
    seed =t*5  # Random seed number
    eval_freq = 5e3  # After how many steps to perform the evaluation
    max_ep = 500  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        100000  # Number of steps over which the initial exploration noise will decay over    default:500000
    )
    expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 40  # Size of the mini-batch
    discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)     #改环境之后，这个也要对应更改，越小的环境这个越小，之前是0.99999
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "TD3_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False # Weather to load a stored model
    random_near_obstacle = False # To take random actions near obstacles or not
    #ou_noise = OUNoise()
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
    env = GazeboEnv("/home/l/DRL-robot-navigation-main/catkin_ws/src/multi_robot_scenario/launch/TD2_world.launch", environment_dim,roscoreip,gazebo_port)
    os.environ['ROS_MASTER_URI'] = f'http://localhost:{roscoreip}'
    os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
    time.sleep(5)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

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
    decay_amount=(0.5-0.1)/1500
    noise1=(0,0)
    if(np.random.uniform(0, 1) > epsilon):
            env.choice=True
    else:
        env. choice=False
    env.choice=True   
    
    history_wiggle_action=[]
    wiggle_flag=True
    wiggle_len=5
    while timestep < max_timesteps:

        # On termination of episode
        if done:
                    if(env.choice):
                        env.test+=1
                        print("success rate:",1.0*env.succes/env.test)
                    history_wiggle_action.clear()
                    with counter_epoch_all.get_lock():
                        counter_epoch_all.value+=1
                        
                        print("counter_epoch_all:",counter_epoch_all.value)                    
                        if(env.choice):
                            timestep_episode.put((timestep,counter_epoch_all.value))
                            expl_noise_episode.put((expl_noise,counter_epoch_all.value))
                        # if(env.end_check):
                        #     if(env.choice):
                        #         counter_epoch_success.value+=1
                        #     env.end_check=False
                        # if(env.choice):
                        #     counter_epoch.value+=1        
                        if(counter_epoch_all.value%100==0):
                                evaluate(network=network, env=env,epoch=counter_epoch_all.value, eval_episodes=eval_ep,success_queue=success_queue,win=win)
                                #succes_all.value=1.0*counter_epoch_success.value/100
                                #writer.add_scalar("Success Rate",  1.0*counter_epoch_success_copy/100,counter_epoch_copy)
                                #success_queue.put((1.0*counter_epoch_success.value/100,counter_epoch_all.value))
                               # counter_epoch_success.value=0
                            
                        if(episode_timesteps!=0 and env.choice):    #这里记录的是真正的自己的奖励，而不是人工市场法的
                            #writer.add_scalar("Rward", episode_reward/episode_timesteps, counter_epoch_copy)
                            reward_queue.put((1.0*episode_reward/episode_timesteps,counter_epoch_all.value))
                            reward_all.put((episode_reward,counter_epoch_all.value))
                        #print("oooooooooooooooooooooooooo")
                        if timestep != 0:
                            #print("ttttttttttttttttttttttttttttttttttttttt")
                            network.train(
                                replay_buffer,
                                episode_timesteps,
                                #writer,
                                Critic_Loss_queue,
                                Actor_Loss_queue,
                                Av_Q_queue,
                                Max_Q_queue,
                                counter_epoch_all.value,
                                batch_size,
                                discount,
                                tau,
                                policy_noise,
                                noise_clip,
                                policy_freq,
                            )
    
                    if timesteps_since_eval >= eval_freq:
                        print("Validating")
                        timesteps_since_eval %= eval_freq
                        # evaluations.append(
                        #     evaluate(network=network, env=env,epoch=epoch, eval_episodes=eval_ep)
                        # )
                        network.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations)
                        epoch += 1

                    state = env.reset()
                    #env.random_target()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                    if(np.random.uniform(0, 1) > epsilon):
                        env.choice=True
                    else:
                        env.choice=False 
                   # env.choice=True
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
        
        p=env.calculate_attraction_force(env.last_odom.pose.pose.position.x,env.last_odom.pose.pose.position.y,env.goal_x,env.goal_y,128)
        f=env.calculate_total_repulsion_force(env.last_odom.pose.pose.position.x,env.last_odom.pose.pose.position.y,math.sqrt(p[0]**2+p[1]**2))
        z=(f[0]+p[0],f[1]+p[1])
        v = math.sqrt(z[0]**2 + z[1]**2)
        v=min(v,1)
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
        l=abs(omega)/math.pi*2
        l=abs(v)-l
        if(v<0):
            v=-l
        else:
             v=l
        if(env.choice):
            print("mode predict")
            #print("model predict")
            action = network.get_action(np.array(state))
          #  noise1=ou_noise.get_noise()
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
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
            print("artificial")
       
            action[0]=(v*2-1)    #人工市场法需要根据环境来改参数，这里的0.5和1.5是simple环境需要的，还有上面的算引力的32也是，原先是256或者128
            action[1]=omega
        
        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        
        a_in = [(action[0]+1)/2, action[1]]
        print(a_in)
        #print(a_in)
        
            
        next_state, reward, done, target = env.step(state,a_in,omega1)
        
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
        replay_buffer.add(state, action, reward, done_bool, next_state)

        # Update the counters
        state = next_state
        episode_timesteps += 1
        if(env.choice):
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
                print("11111111111111111111111111111")
            if (k==1):
                writer.add_scalar("Success Rate", success_rate, epoch)
                print("333333333333333333333333333333")
            if(k==2):
                writer.add_scalar("Critic-Loss", success_rate, epoch)
                print("4444444444444444444444444444")
            if(k==3):
                print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz:",epoch)
                writer.add_scalar("Actor-Loss", success_rate, epoch)
                print("5555555555555555555555555555555555") 
            if(k==4):
                writer.add_scalar("Av_Q", success_rate, epoch)
                print("666666666666666666666666666666666")
            if(k==5):
                writer.add_scalar("Max-Q", success_rate, epoch)
                print("777777777777777777777777777777777")   
            if(k==6):
                writer.add_scalar("time_steps", success_rate, epoch)
                print("88888888888888888888888888888888")   
            if(k==7):
                writer.add_scalar("expl_noise", success_rate, epoch)
                print("9999999999999999999999999999999999")   
            if(k==8):
                writer.add_scalar("Episode Rward", success_rate, epoch)
                print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")   
        except queue.Empty:  # 捕获 queue.Empty 异常
            continue
    writer.close()
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    logdir=f"runs/7998"
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
    processes=[]
    params={'n_workers':3,
            'n_loggers':2,
            }
    success_queue = mp.Queue()
    reward_queue = mp.Queue()
    Critic_Loss_queue=mp.Queue()
    Actor_Loss_queue=mp.Queue()
    Max_Q_queue=mp.Queue()
    Av_Q_queue=mp.Queue()
    
    loss_queue = mp.Queue()
    timestep_episode = mp.Queue()
    expl_noise_episode = mp.Queue()
    reward_all=mp.Queue()
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
    
    logger6 = mp.Process(target=logger_process, args=(6,timestep_episode, logdir))
    logger6.start()
    processes.append(logger6)
    
    logger7 = mp.Process(target=logger_process, args=(7,expl_noise_episode, logdir))
    logger7.start()
    processes.append(logger7)
    
    logger8 = mp.Process(target=logger_process, args=(8,reward_all, logdir))
    logger8.start()
    processes.append(logger8)
 

    for i in range (params["n_workers"]):
        p=mp.Process(target=worker,args=(i,MasterNode,counter_epoch,counter_epoch_success,params,success_queue,reward_queue,Critic_Loss_queue,Actor_Loss_queue,Av_Q_queue,Max_Q_queue,succes_all,counter_epoch_all,timestep_episode,expl_noise_episode,reward_all,win))
        p.start()
        processes.append(p)
    for p in    processes:
        p.join()
    for p in processes:
        p.terminate()