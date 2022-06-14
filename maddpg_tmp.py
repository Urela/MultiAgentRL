import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

gamma = 0.99
tau   = 0.005 # for target network soft update

class ReplayBuffer():
  def __init__(self, size=50000, num_agents=1):
    self.memory = collections.deque(maxlen=size)
    self.agent_buffers = [collections.deque(maxlen=size) for _ in range(num_agents)]
    self.num_agents = num_agents

  def __len__(self): return len(self.memory)

  #def store(self, experiance): self.memory.append(experiance)
  def store_transition(self, raw_obs, state, action, reward, raw_n_obs, nstate, done):

    for idx in range( self.num_agents ):
      self.agent_buffers[idx].append( raw_obs[idx], action[idx], raw_n_obs[idx] )
    self.memory.append((state, reward, nstate, done))


  def sample(self, batch_size):
    #batch = random.sample(self.memory, batch_size)
    batch = np.random.choice(len(self.memory), batch_size, replace=False)

    states  = torch.tensor([x[0]   for x in self.memory[batch]], dtype=torch.float)
    rewards = torch.tensor([[x[2]] for x in self.memory[batch]]).float()
    nstates = torch.tensor([x[3]   for x in self.memory[batch]], dtype=torch.float)
    dones   = torch.tensor([1-x[4] for x in self.memory[batch]])

    for idx in range( self.num_agents ):
      actor_states.append( self.memory[batch][0] )
      actor_nstates.append(self.memory[batch][2] )
      actions.append( self.memory[batch][1] )

    return actor_states, states, actions, rewards, actor_nstates, nstate, dones

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims, lr=5e-4):
    super(Actor, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(1,   32, 8, stride=4), nn.ReLU(),
      nn.Conv2d(32,  64, 8, stride=2), nn.ReLU(),
      nn.Conv2d(64,  64, 3, stride=1), nn.ReLU(),
      nn.Flatten(),
      nn.Linear(1600, 512), nn.ReLU(),
      nn.Linear(512, 1)
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.to('cpu')

  def forward(self, x):
    x = self.net(x )
    return x

class Critic(nn.Module):
  def __init__(self, in_dims, out_dims, lr=1e-3):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_dims+out_dims, 128)
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, o, a):
    x = torch.cat([o, a], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


import supersuit as ss
from pettingzoo.butterfly import pistonball_v6

num_agents = 20

env = pistonball_v6.parallel_env(
    n_pistons = num_agents,
    time_penalty=-0.1,
    continuous=True,
    random_drop=True,
    random_rotate=True,
    ball_mass=0.75,
    ball_friction=0.3,
    ball_elasticity=1.5,
    max_cycles=125,
)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

# Preprocesssing
print( 'setting world')
env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 1)

# The environment dictates the number of agents we get
#env.action_space(agent).shape:
env.reset()

memory = ReplayBuffer(num_agents=num_agents)
#AGENTS = [Actor(0, env.action_space(a).shape[0]).to('cpu') for a in env.agent_iter()]
AGENTS = [Actor(0, 1).to('cpu') for a in range(num_agents)]

#for i,a in enumerate(env.agents):
#  print( i, env.action_space(a).shape )
#  print( i, env.observation_space(a).shape )
#print ( "Yeahh" )

#for agent in env.agent_iter():
for i, agent in enumerate(env.agent_iter()):

  observation, reward, done, info = env.last()
  #action = np.random.uniform(low=-1, high=1, size=1) # random policy for every agent
  #print(torch.tensor([observation.T]).float().shape)
  action = AGENTS[i%num_agents]( torch.tensor([observation.T]).float() ).detach().numpy()[0]
  #print(action1, action)
  #print(action.shape, observation.shape)

  env.step(action)
  env.render()
env.close()
