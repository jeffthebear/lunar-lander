# docs: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

from collections import defaultdict
import gym
import math
from scipy import interpolate
import numpy as np
import statistics
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation = env.reset(seed=42)

class QAgent():
   def __init__(self, action_space):
      self.action_space = range(action_space.n)
      self.action_table = defaultdict(list)  # key (action) : value (prev_angle_todo, prev_hover_todo, policy_value)
      self.min_samples = 100

   @staticmethod
   def calc_hover_angle(s):
      angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
      if angle_targ > 0.4:
         angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
      if angle_targ < -0.4:
         angle_targ = -0.4
      hover_targ = 0.55 * np.abs(
         s[0]
      )  # target y should be proportional to horizontal offset

      angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
      hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

      if s[6] or s[7]:  # legs have contact
         angle_todo = 0
         hover_todo = (
            -(s[3]) * 0.5
         )  # override to reduce fall speed, that's all we need after contact

      return angle_todo, hover_todo


   def next_action(self, observation):
      angle_todo, hover_todo = self.calc_hover_angle(observation)

      # calculate results of each action
      action_table_responses = {}
      for action in self.action_space:
         if len(self.action_table.get(action, [])) >= self.min_samples:
            action_tuples = list(zip(*self.action_table[action]))
            prev_angle_todos = action_tuples[0]
            prev_hover_todos = action_tuples[1]
            policy_values = action_tuples[2]

            interp = interpolate.NearestNDInterpolator(list(zip(prev_angle_todos, prev_hover_todos)), policy_values)

            action_table_responses[action] = interp((angle_todo, hover_todo))
         else:
            # haven't seen this action
            action_table_responses[action] = 0

      # if multiple zeros randomly select
      zero_responses = [k for k, v in action_table_responses.items() if v == 0]
      if len(zero_responses) > 1:
         return np.random.choice(zero_responses)

      action_table_responses_sorted = sorted(action_table_responses.items(), 
         key=lambda x: x[1])

      action = action_table_responses_sorted[-1][0]

      # had to add this heuristic or else lander might not land
      if hover_todo < 0.05 and action == 2:
         return 0
         
      return action 

      
   def record_result(self, action, prev_observation, observation, reward):
      """We want to record the result of taking the previous action"""
      prev_angle_todo, prev_hover_todo = self.calc_hover_angle(prev_observation)
      curr_angle_todo, curr_hover_todo = self.calc_hover_angle(observation)
      policy_value = -1 * (abs(curr_angle_todo) + abs(curr_hover_todo))

      self.action_table[action].append((prev_angle_todo, prev_hover_todo, policy_value + reward))
      
q_agent = QAgent(env.action_space)
total_reward = 0

for _ in range(10000):
   prev_observation = observation
   action = q_agent.next_action(prev_observation)
   observation, reward, done, info = env.step(action)
   q_agent.record_result(action, prev_observation, observation, reward / 100. if done else 0)
   total_reward += reward
   print(f'reward: {total_reward}')

   if done:
      if reward > 0:
         print(f'------ Success: {reward} -----')
      observation, info = env.reset(return_info=True)

env.close()
