import supersuit as ss
from pettingzoo.butterfly import pistonball_v6

import numpy as np

env = pistonball_v6.parallel_env(
    n_pistons=20,
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
env = ss.frame_stack_v1(env, 3)

print ( "Yeahh" )
env.reset()
for agent in env.agent_iter():

  observation, reward, done, info = env.last()
  action = np.random.uniform(low=-1, high=1, size=1) # random policy for every agent

  #print(agent, action)
  env.step(action)
  env.render()
