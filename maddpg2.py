import os
import numpy as np
import collections
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
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
    self.fc1 = nn.Linear(in_dims, 64)
    self.fc2 = nn.Linear(64, 64)
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
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, o, a):
    x = torch.cat([o, a], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class DDPG_Agent():
  def __init__(self, in_dims, out_dims):
    #self.memory = ReplayBuffer()
    self.actor  = Actor(in_dims,  out_dims).to('cpu')
    self.critic = Critic(in_dims, out_dims).to('cpu')
    self.targ_actor  = Actor(in_dims,  out_dims).to('cpu') # target critic
    self.targ_critic = Critic(in_dims, out_dims).to('cpu') # target actor

    # intialize the targets to match their networks
    self.targ_critic.load_state_dict(self.critic.state_dict())
    self.targ_actor.load_state_dict(self.actor.state_dict())

class MADDPG():
  def __init__(self, env):
    self.main_memory = ReplayBuffer()
    self.agents = {}
    self.agent_memorys = {}
    for a in env.agents:
      in_dims  = env.observation_space(a).shape[0] 
      out_dims = env.action_space(a).n 
      self.agents[a] = DDPG_Agent(in_dims, out_dims)
      self.agent_memorys[a] = ReplayBuffer()

  def get_action(self, obs):
    actions = {}
    for i, a in enumerate(self.agents):  # each agent select action according to their obs
      state = torch.from_numpy(obs[a])
      out = self.agents[a].actor(state)  # torch.Size([1, action_size])
      #out = F.gumbel_softmax(out, hard=True)  ##?????
      actions[a] = out.argmax().item()
    return actions

  def train(self):
    for i, a in enumerate(self.agents):  
      if(len(self.agent_memorys[a]) >= 2000):
        print("learning")
        states, actions, rewards, nstates, dones = self.agent_memorys[a].sample(32)
        print(i, states.shape, actions.shape)
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x9 and 13x128)
        q = self.agents[a].critic(states, actions)
        #a_targ = self.agents[a].targ_actor(nstates)
        #q_targ = self.agents[a].targ_critic(nstates, a_targ)
        #q_targ = rewards + gamma*q_targ * dones
        #critic_loss = F.smooth_l1_loss(q, q_targ.detach() )

        #self.agents[a].critic.optimizer.zero_grad()
        #critic_loss.backward()
        #self.agents[a].critic.optimizer.step()

        #a_pred = agent.actor(states)
        #actor_loss = self.agents[a].critic(states, a_pred).mean()
        #self.agents[a].actor.optimizer.zero_grad()
        #actor_loss.backward()
        #self.agents[a].actor.optimizer.step()

    #self.update_targets()

  def update_targets(self):
      # soft update https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    for i, agent in enumerate(self.agents):  
      network, target = agent.actor, agent.targ_actor
      for param_targ, param in zip(target.parameters(), network.parameters()):
        param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

      network, target = agent.critic, agent.targ_critic
      for param_targ, param in zip(target.parameters(), network.parameters()):
        param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)


#https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch
env = simple_adversary_v2.parallel_env(max_cycles=25)

env.reset()
AGENT = MADDPG(env)

episode_num = 30000
num_agents = env.num_agents
print( num_agents, "number of agents")
scores = {agent: np.zeros(episode_num) for agent in env.agents} # reward of each episode of each agent

steps = 0  # global step counter
for episode in range(episode_num):
  obs = env.reset()
  agent_score = {a: 0 for a in env.agents} # agent reward of the current episode
  while env.agents:  
    steps += 1
    #actions = {a: env.action_space(a).sample() for a in env.agents}
    action = AGENT.get_action(obs) 
    next_obs, reward, done, info = env.step(action)
    for agent_id in obs.keys():
      _state  = obs[agent_id]
      _action = action[agent_id]
      _reward = reward[agent_id]
      _nstate = next_obs[agent_id]
      _done   = done[agent_id]
      AGENT.agent_memorys[agent_id].store((_state, _action, _reward, _nstate, _done))

    obs = next_obs
    AGENT.train()
    #print("running")
    #env.render()
    for a, r in reward.items():  # update reward
      agent_score[a] += r
