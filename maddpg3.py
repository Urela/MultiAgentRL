import os
import numpy as np
import collections
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

gamma = 0.99
tau   = 0.005 # for target network soft update

class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000):
    self.states  = np.zeros((length, obs_dim))
    self.actions = np.zeros((length, act_dim))
    self.rewards = np.zeros(length)
    self.nstates = np.zeros((length, obs_dim))
    self.dones   = np.zeros(length, dtype=bool)

    self.idx  = 0
    self.size = length

  def __len__(self): return self.idx

  def store(self, obs, action, reward, next_obs, done):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.nstates[idx] = next_obs
    self.dones[idx] = done

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).float()
    actions = torch.tensor( self.actions[indices], dtype=torch.float).float()
    rewards = torch.tensor( self.rewards[indices], dtype=torch.float).float()
    nstates = torch.tensor( self.nstates[indices], dtype=torch.float).float()
    dones   = torch.tensor( self.dones[indices] ).float()
    return states, actions, rewards, nstates, dones

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims, lr=5e-4):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_dims, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, out_dims)
    #self.net = nn.Sequential(
    #    nn.Linear(in_dims, 64), nn.ReLU(),
    #    nn.Linear(64, 64),      nn.ReLU(),
    #    nn.Linear(64, out_dims),
    #).apply(self.init)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  @staticmethod
  def init(m):
    """init parameter of the module"""
    gain = nn.init.calculate_gain('relu')
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight, gain=gain)
      m.bias.data.fill_(0.01)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    #x = self.net(x)
    return x

class Critic(nn.Module):
  def __init__(self, in_dims, lr=1e-3):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_dims, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 1)
    #self.net = nn.Sequential(
    #    nn.Linear(in_dims, 64), nn.ReLU(),
    #    nn.Linear(64, 64),      nn.ReLU(),
    #    nn.Linear(64, 1),
    #).apply(self.init)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  @staticmethod
  def init(m):
    """init parameter of the module"""
    gain = nn.init.calculate_gain('relu')
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight, gain=gain)
      m.bias.data.fill_(0.01)

  def forward(self, o, a):
    #x = torch.cat([o,a], dim=1) 
    x = torch.cat(o+a, dim=1) # adding lists concates them together
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    #x = self.net(x)
    return x

class DDPG_Agent():
  def __init__(self, in_dims, out_dims):
    self.actor  = Actor(in_dims,  out_dims).to('cpu')
    self.targ_actor  = Actor(in_dims,  out_dims).to('cpu') 
    # intialize the target to match its behaviour network
    self.targ_actor.load_state_dict(self.actor.state_dict()) 

class MADDPG():
  def __init__(self, env):
    self.agents = {}
    self.memory = {}
    self.env = env
    for a in env.agents:
      in_dims  = env.observation_space(a).shape[0] 
      out_dims = env.action_space(a).n 
      self.agents[a] = DDPG_Agent(in_dims, out_dims)
      self.memory[a] = ReplayBuffer(in_dims, out_dims, 50000)

    # Using one Central critic
    critic_input = sum(env.observation_space(a).shape[0]+  
                             env.action_space(a).n for a in env.agents) # gloabal obs+action
    self.critic      = Critic(critic_input).to('cpu')
    self.targ_critic = Critic(critic_input).to('cpu') # target critic
    # intialize the target to match its behaviour network
    self.targ_critic.load_state_dict(self.critic.state_dict())

  def get_action(self, obs):
    actions = {}
    for i, a in enumerate(self.agents):  # each agent select action according to their obs
      state = torch.from_numpy(obs[a])
      action = self.agents[a].actor(state)  # torch.Size([1, action_size])
      actions[a] = action.argmax().item()
    return actions

  def store(self, obs, action, reward, next_obs, done):
    for agent_id in obs.keys():
      _state  = obs[agent_id]
      _reward = reward[agent_id]
      _nstate = next_obs[agent_id]
      _done   = done[agent_id]
      _action = np.zeros(self.env.action_space(agent_id).n)
      _action[action[agent_id]] = 1  # convert action to one-hot encoding
      self.memory[agent_id].store(_state, _action, _reward, _nstate, _done)

  def sample(self, batch_size):
    # NOTE that in MADDPG, we need the obs and actions of all agents
    # but only the reward and done of the current agent is needed in the calculation
    states, actions, rewards, nstates, dones = {}, {}, {}, {}, {}
    next_actions = {}
    for agent_id, buffer in self.memory.items():
      o, a, r, n_o, d = buffer.sample(batch_size)
      states[agent_id] = o
      actions[agent_id] = a
      rewards[agent_id] = r
      nstates[agent_id] = n_o
      dones[agent_id] = d

      # calculate next_action using target_network and next_state
      next_actions[agent_id] = self.agents[agent_id].targ_actor(n_o)

    return states, actions, rewards, nstates, dones, next_actions

  def train(self, batch_size=32):
    for a in self.agents:  
      if(len(self.memory[a]) >= 2000):
        states, actions, rewards, nstates, dones, next_actions = self.sample(32)
        q = self.critic( list(states.values()), list(actions.values()) ).squeeze()
        q_targ = self.targ_critic(list(nstates.values()), list(next_actions.values()) ).squeeze()

        #print(rewards[a].shape, ( q_targ).shape , dones[a].shape)
        q_targ = rewards[a] + gamma*q_targ * (1-dones[a])
        critic_loss = F.smooth_l1_loss(q, q_targ, reduction='mean')

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        out = self.agents[a].actor(states[a])
        actor_loss = -self.critic( list(states.values()), list(actions.values()) ).mean()
        actor_loss_pse = torch.pow(out, 2).mean()
        actor_loss = actor_loss + 1e-3 * actor_loss_pse

        self.agents[a].actor.optimizer.zero_grad()
        actor_loss.backward()
        self.agents[a].actor.optimizer.step()

    self.update_targets()

  def update_targets(self):
      # soft update https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    for a in self.agents:  
      network, target = self.agents[a].actor, self.agents[a].targ_actor
      for param_targ, param in zip(target.parameters(), network.parameters()):
        param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

    network, target = self.critic, self.targ_critic
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

#https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch
env = simple_adversary_v2.parallel_env(max_cycles=25)
env.reset()
AGENT = MADDPG(env)

episode_num = 3000 # 30000
num_agents = env.num_agents
#print( num_agents, "number of agents")
scores = {agent: np.zeros(episode_num) for agent in env.agents} # reward of each episode of each agent
steps = 0  # global step counter
for episode in range(episode_num):
  obs = env.reset()
  agent_score = {a: 0 for a in env.agents} # agent reward of the current episode
  while env.agents:  
    steps += 1
    #action = {a: env.action_space(a).sample() for a in env.agents}
    action = AGENT.get_action(obs) 
    #env.render()
    next_obs, reward, done, info = env.step(action)
    AGENT.store(obs, action, reward, next_obs, done)

    obs = next_obs
    AGENT.train()

    for a, r in reward.items():  # update episodic reward
      agent_score[a] += r

    # episode finishes
  for a, r in agent_score.items():  # record reward
      scores[a][episode] = r

  #if (episode + 1) >2500: env.render()
  if (episode + 1) % 100 == 0:  # print info every 100 episodes
    message = f'episode {episode + 1}, '
    sum_reward = 0
    for agent_id, r in agent_score.items():  # record reward
      message += f'{agent_id}: {r:>4f}; '
      sum_reward += r
    message += f'sum reward: {sum_reward}'
    print(message)

env.close()

from bokeh.plotting import figure, show
p = figure(title="TODO", x_axis_label="Episodes", y_axis_label="Scores")
p.line(np.arange(len(scores["adversary_0"])), scores["adversary_0"],  legend_label="adversary_0", line_color="red", line_width=1)
p.line(np.arange(len(scores["agent_1"])), scores["agent_1"],  legend_label="agent_1", line_color="blue", line_width=2)
p.line(np.arange(len(scores["agent_0"])), scores["agent_0"],  legend_label="agent_0", line_color="green", line_width=1)
show(p) 
