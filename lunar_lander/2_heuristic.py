# docs: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

import gym
from gym.envs.box2d.lunar_lander import heuristic

def heuristic_policy(env, observation):
    return heuristic(env.unwrapped, observation)

env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation = env.reset(seed=42)

for _ in range(1000):
   action = heuristic_policy(env, observation)
   observation, reward, done, info = env.step(action)

   if done:
      observation, info = env.reset(return_info=True)

env.close()
