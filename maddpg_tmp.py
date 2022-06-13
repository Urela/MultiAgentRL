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

class ReplayBuffer:
  def __init__(self, size=50000):
    self.memory = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append(experiance)

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
    actions = torch.tensor([[x[1]] for x in batch]).float()
    rewards = torch.tensor([[x[2]] for x in batch]).float()
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
    dones   = torch.tensor([1-x[4] for x in batch])
    return states, actions, rewards, nstates, dones

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims, lr=5e-4):
    super(Actor, self).__init__()
    self.net = nn.Sequential(
      self.layer_init(nn.Conv2d(4,  32, 8, stride=4)), nn.ReLU(),
      self.layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
      self.layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
      nn.Flatten(),

      self.layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
      self.layer_init(nn.Linear(512, out_space.n), std=0.01),
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

class DDPG():
  def __init__(self, in_dims, out_dims):
    self.out_dims = out_dims
    self.memory = ReplayBuffer()

    self.actor  = Actor(in_dims,  out_dims).to('cpu')
    self.critic = Critic(in_dims, out_dims).to('cpu')
    self.targ_actor  = Actor(in_dims,  out_dims).to('cpu') # target critic
    self.targ_critic = Critic(in_dims, out_dims).to('cpu') # target actor

    # intialize the targets to match their networks
    self.targ_critic.load_state_dict(self.critic.state_dict())
    self.targ_actor.load_state_dict(self.actor.state_dict())

  def store(self, exp):
    self.memory.store((exp))

  def get_action(self, obs):
    action = self.actor(torch.from_numpy(obs).float()) 
    # DDPG is a deterministic, so in order to explore we need use
    # epsilon greedy or we can add noise the makes the policy stochasic
    # https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3
    # Normal distribution is easier to implement than OrnsteinUhlenbeckNoise
    normal_scalar = .25
    action = action.item() + np.random.randn(self.out_dims) * normal_scalar
    return action.item()


  def train(self):
    if(len(self.memory) >= 2000):
      for i in range(10):

        states, actions, rewards, nstates, dones = self.memory.sample(32)

        q = self.critic(states, actions)
        a_targ = self.targ_actor(nstates)
        q_targ = self.targ_critic(nstates, a_targ)
        q_targ = rewards + gamma*q_targ * dones
        critic_loss = F.smooth_l1_loss(q, q_targ.detach() )

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        a_pred = self.actor(states)
        actor_loss = -self.critic(states, a_pred).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
      
    self.update_targets()

  def update_targets(self):
    # soft update https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    network, target = self.actor, self.targ_actor
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

    network, target = self.critic, self.targ_critic
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)




import supersuit as ss
from pettingzoo.butterfly import pistonball_v6

NUM_AGENTS = 20

env = pistonball_v6.parallel_env(
    n_pistons = NUM_AGENTS,
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

### HELP ME PLEASE
#agent =  MADDPG( in_dims, out_dims, NUM_AGENTS)
#agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

# Preprocesssing
print( 'setting world')
env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 1)

# The environment dictates the number of agents we get
#env.action_space(agent).shape:
env.reset()
for i,a in enumerate(env.agents):
  print( i, env.action_space(a).shape )
  print( i, env.observation_space(a).shape )

#print ( "Yeahh" )
#env.reset()
#for agent in env.agent_iter():
#
#  observation, reward, done, info = env.last()
#  action = np.random.uniform(low=-1, high=1, size=1) # random policy for every agent
#  #print( observation.shape )
#
#  #print(agent, action)
#  env.step(action)
#  env.render()
#env.close()
