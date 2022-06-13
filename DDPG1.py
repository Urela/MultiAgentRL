import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import collections
import random

# Hyperparameters
gamma = 0.99
tau = 0.001  # for target network soft update

class ReplayBuffer:
  def __init__(self, size=50000):
    self.memory = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append(experiance)

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
    actions = torch.tensor([x[1] for x in batch]).float()
    rewards = torch.tensor([[x[2]] for x in batch]).float()
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
    dones   = torch.tensor([[x[4]] for x in batch])
    return states, actions, rewards, nstates, dones

# https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
class Actor(nn.Module):
  def __init__(self, in_dims, out_dims, lr=5e-4):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_dims, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, out_dims)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    return x


class Critic(nn.Module):
  def __init__(self, in_dims, out_dims, lr=1e-3):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_dims+out_dims, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, o, a):
    x = torch.cat([o, a], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class DDPG():
  def __init__(self, in_dims, out_dims):
    self.out_dims = out_dims
    self.memory = ReplayBuffer()

    self.actor  = Actor(in_dims,  out_dims).to('cpu')
    self.critic = Critic(in_dims, out_dims).to('cpu')
    self.targ_actor  = Actor(in_dims,  out_dims).to('cpu') # target critic
    self.targ_critic = Critic(in_dims, out_dims).to('cpu') # target actor

    # intialize the targets to match their networks
    network, target = self.actor, self.targ_actor
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_( param.data )

    network, target = self.critic, self.targ_critic
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_( param.data )

  def store(self, exp):
    self.memory.store((exp))

  def get_action(self, obs):
    self.actor.eval()
    obs = torch.from_numpy(obs).float().to('cpu')
    mu  = self.actor(obs).cpu().data.numpy()
    # DDPG is a deterministic, so in order to explore we need use
    # epsilon greedy or we can add noise the makes the policy stochasic
    # https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3
    # Normal distribution is easier to implement than OrnsteinUhlenbeckNoise
    normal_scalar = .25
    action = mu.item() + np.random.randn(self.out_dims) * normal_scalar
    return action # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]

  def train(self):
    if(len(self.memory) >= 2000):
      for i in range(1):
        states, actions, rewards, nstates, dones = self.memory.sample(batch_size=64)

        q = self.critic(states, actions)
        a_targ = self.targ_actor(nstates)
        q_targ = self.targ_critic(nstates, a_targ)
        q_targ = rewards + gamma*q_targ #* dones
        critic_loss = F.smooth_l1_loss(q, q_targ)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        a_pred = self.actor(states)
        actor_loss = -self.critic(states, a_pred).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
    pass

  def update_targets(self):
    # soft update https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    network, target = self.actor, self.targ_actor
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

    network, target = self.critic, self.targ_critic
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)


env = gym.make('LunarLanderContinuous-v2')
env = gym.make('Pendulum-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

print("running")
scores = []
avg_scores =[]
for epi in range(1000):
  obs = env.reset()
  while True:
    # env.render()
    action = agent.get_action(obs)
    #action = env.action_space.sample()
    _obs, reward, done, info = env.step(action)
    agent.store((obs, action, reward, _obs, done))
    agent.train()

    obs = _obs
    if "episode" in info.keys():
      agent.update_targets()
      scores.append(info['episode']['r'])
      avg_scores = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {info['episode']['r']}")
      break

env.close()

y = scores 
x = np.arange(len(y))

from bokeh.plotting import figure, show
p = figure(title="TODO", x_axis_label="Episodes", y_axis_label="Scores")
p.line(x, y,  legend_label="Scores", line_color="blue", line_width=2)
show(p) 
