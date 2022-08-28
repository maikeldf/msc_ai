#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import random
import numpy as np
from gym import spaces

class CloudEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(CloudEnv, self).__init__()

        self.df = df
        self.status = ['FUNCTIONAL', 'DEGRADED']
        self.vm_status = self.status[0]
        self.cpu_mean = 0
        self.cpu_variance = 0
        self.errors = 0
        self.vm_id = -1
        self.obs = 0
        self.last_obs = 0

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,4), dtype=np.float16)

    def _next_observation(self):
        df = self.df[self.df['Vm']==self.vm_id][['CPU1','CPU2','CPU3','CPU4']].copy()
        df = df.reset_index(drop=True)

        current_step = random.randint(
            0, len(df.loc[:, 'CPU1'].values)-1)
        
        self.obs = np.array(df[current_step:current_step + 1])

        return self.obs

    def _take_action(self, action): 
        self.cpu_mean = np.mean([self.obs.min(),self.obs.max()])
        self.cpu_variance = np.var(self.obs)                
        self.vm_status = self.status[action]


    def step(self, action):
        self._take_action(action)
        
        reward = 5
        
        if self.cpu_variance >= 10:
            if self.cpu_mean >= 20: # HIGH MEAN
                if self.vm_status == self.status[0]: # FUNCTIONAL
                    reward = -10
            elif self.cpu_mean < 20: # LOW MEAN
                if self.vm_status == self.status[1]: # DEGRADED
                    reward = -10
        elif self.cpu_variance < 10:
            if self.cpu_mean < 20: # LOW MEAN
                if self.vm_status == self.status[1]: # DEGRADED
                    reward = -10
            elif self.cpu_mean >= 20: # HIGH MEAN
                if self.vm_status == self.status[1]: # DEGRADED
                    reward = -10

        self.errors += 0 if reward == 5 else 1

        done = False
        
        self.vm_id = self.df.iloc[self.current_step]['Vm']

        self.current_step += 1
        if self.current_step >= len(self.df):
            self.current_step = 0
            done = True

        self.last_obs = self.obs
        self.obs = self._next_observation()

        return self.obs, reward, done, {'errors':{
                                        'total':self.errors,
                                        'obs_min':self.last_obs[0].min(),
                                        'obs_max':self.last_obs[0].max(),
                                        'mean':self.cpu_mean,
                                        'variance':self.cpu_variance,
                                        'status':self.vm_status}}

    def reset(self):
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Vm'].values)-1)

        self.vm_id = self.df.iloc[self.current_step]['Vm']
        self.obs = self._next_observation()
        
        return self.obs

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'VM Id: {self.vm_id}')
        print(f'Obs: {self.last_obs}')
        print(f'Cpu Mean: {self.cpu_mean}')
        print(f'Cpu VAriance: {self.cpu_variance}')        
        print(f'VM Status: {self.vm_status}')
        print(f'Errors: {self.errors}\n')
