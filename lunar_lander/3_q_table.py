# docs: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

import gym
import math
import numpy as np
import statistics
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation = env.reset(seed=42)

class QAgent():
   def __init__(self, action_space):
      self.action_space = range(action_space.n)
      self.q_table = {}  # key (action, angle_todo, hover_todo) : value (-1 * (lowest abs(angle_todo) + abs(hover_todo)) + end of episode score)
      self.round_hover = .05
      self.round_angle = .05
      self.round_prec = 2
      self.lr = .5
      self.gamma = .8  # discount of next state
      self.epsilon = .99  # exploration / exploitation ratio
      self.decay = .95  # epsilon decay rate

   @staticmethod
   def calc_hover_angle(s):
      """ Taken from the heuristic implementation.  
          Converts state to angle_todo and hover_todo to make reward function easier to learn.
      """

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

   def round_angle_hover(self, angle_todo, hover_todo):
      """ Rounds angle and hover to specified precision.
          Used for bucketing in the Q-table.
      """
      round_impl = lambda base, x: round(base * round(float(x)/base), self.round_prec)
      return (
         round_impl(self.round_angle, angle_todo),       
         round_impl(self.round_hover, hover_todo)
      )

   def get_all_action_values(self, angle_todo, hover_todo):
      """ Looks up all action values for in Q-table for angle_todo & hover_todo"""
      q_table_responses = {}
      for action in self.action_space:
         # get q_table entry or 0 if we haven't see an action for this angle_todo, hover_todo
         q_table_responses[action] = self.q_table.get((action, angle_todo, hover_todo), 0)
      return q_table_responses

   def next_action(self, observation):
      """ Selects next action """
      choose_random = np.random.choice([True, False], p=[self.epsilon, 1. - self.epsilon])

      if choose_random:
         # explore
         action = np.random.choice(self.action_space)
      else:
         # exploit
         angle_todo, hover_todo = self.calc_hover_angle(observation)
         angle_todo, hover_todo = self.round_angle_hover(angle_todo, hover_todo)
         q_table_action_values = self.get_all_action_values(angle_todo, hover_todo)

         # optional heuristic: upweight down engine
         q_table_action_values[2] = (.95 * q_table_action_values[2]) if q_table_action_values[2] < 0 \
            else q_table_action_values[2]

         # if multiple zeros randomly select
         q_table_zero_responses = [k for k, v in q_table_action_values.items() if v == 0]
         if len(q_table_zero_responses) > 1 and all(q_table_action_values.values()) <= 0:
            return np.random.choice(q_table_zero_responses)

         # take action with the largest value
         q_table_action_values_sorted = sorted(q_table_action_values.items(), key=lambda x: x[1])
         action = q_table_action_values_sorted[-1][0]

         # had to add this heuristic or else lander might not land
         if hover_todo < 0.05 and action == 2:
            return 0
         
      return action

      
   def record_result(self, action, prev_observation, observation, reward):
      """We want to record the result of taking the previous action"""
      prev_angle_todo, prev_hover_todo = self.calc_hover_angle(prev_observation)
      prev_angle_todo_rnd, prev_hover_todo_rnd = self.round_angle_hover(prev_angle_todo, prev_hover_todo)
      curr_angle_todo, curr_hover_todo = self.calc_hover_angle(observation)
      # formula trying to reach 0 in angle_todo and hover todo (add to reward)
      reward += -1 * (abs(curr_angle_todo) + abs(curr_hover_todo))  
      # estimate optimal q of next state
      curr_q_values = self.get_all_action_values(curr_angle_todo, curr_hover_todo)
      curr_q_values_sorted = sorted(curr_q_values.items(), key=lambda x: x[1])
      optimal_curr_q_value = curr_q_values_sorted[-1][1]
      prev_q_value = self.q_table.get((action, prev_angle_todo_rnd, prev_hover_todo_rnd), 0)
      
      # calculate new q-value
      new_q_value = prev_q_value + self.lr * (reward + self.gamma * optimal_curr_q_value - prev_q_value)
      self.q_table[(action, prev_angle_todo_rnd, prev_hover_todo_rnd)] = new_q_value


q_agent = QAgent(env.action_space)
total_reward = 0

for _ in range(10000):
   prev_observation = observation
   action = q_agent.next_action(prev_observation)
   observation, reward, done, info = env.step(action)
   q_agent.record_result(action, prev_observation, observation, reward / 100. if done else 0)
   total_reward += reward
   print(f'q_table length: {len(q_agent.q_table)}, epsilon: {q_agent.epsilon} reward: {total_reward}')

   if done:
      if reward > 0:
         print(f'------ Success: {reward} -----')
      q_agent.epsilon *= q_agent.decay
      observation, info = env.reset(return_info=True)

env.close()
