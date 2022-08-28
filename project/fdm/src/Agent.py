#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from typing import Callable
from stable_baselines3 import PPO
from src.CloudEnv import CloudEnv


import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class Agent:
    def __init__(self, 
        log_dir,
        log_eval,
        model_dir,
        input_file,
        steps):
        
        self.n_steps = 0
        self.total_timesteps=steps
        self.rewards_per_episode = []
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir,"cloud")
        self.log_dir = log_dir
        self.log_eval = log_eval
        self.input_file = input_file        
        self.train_data, self.test_data = self.get_train_test_sample(self.__get_df(input_file))        

        # Debug purposes
        self.error_file = open(os.path.join(self.log_dir,'errors.csv'), 'w')
        self.error_file.write("obs_min,obs_max,mean,variance,status\n")
        self.eval_scalar_name = 'evaluate/'

    def get_train_test_sample(self, df):
        train= df.sample(frac=0.8,random_state=1)
        test= df.drop(train.index)

        test = test.reset_index()
        test['Count'] = range(1, len(test)+1)
        test = test.drop(['index'], axis=1)

        return train.sort_index(), test
        
    def __del__(self):
        self.error_file.close()

    def __callback(self, _locals, _globals):        
        self.rewards_per_episode.append(_locals['rewards'][0])

        # Print stats every 100 calls
        if (self.n_steps) % 100 == 0:
            with self.writer.as_default():
                tf.summary.scalar(self.eval_scalar_name+'avg_rewards', 
                    np.mean(self.rewards_per_episode), step=self.n_steps)

                tf.summary.scalar(self.eval_scalar_name+'errors', 
                    _locals['infos'][0]['err'], step=self.n_steps)

        self.n_steps += 1
        self.writer.flush()
        return True

    def __get_df(self, input_file):
        df = pd.read_csv(input_file)
        return  df.sort_index()


    def linear_schedule(self,
        initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.
        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value
        return func

    def train(self, lin):
        env = DummyVecEnv([lambda: CloudEnv(self.train_data)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        if os.path.exists(self.model_path):
            model = PPO.load(self.model_path, 
                env, 
                verbose=1, 
                tensorboard_log=self.log_dir)
        else:
            model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate=self.linear_schedule(0.001) if lin else 3e-4, 
                tensorboard_log=self.log_dir)

        model.learn(self.total_timesteps
        , reset_num_timesteps=True)

        model.save(self.model_path)
        stats_path = os.path.join(self.model_dir, "env.pkl")
        env.save(stats_path)
        env.close()
        del model, env

    # EVALUATION
    def evaluate(self):
        stats_path = os.path.join(self.model_dir, "env.pkl")

        env = DummyVecEnv([lambda: CloudEnv(self.test_data)])
        env = VecNormalize.load(stats_path, env)

        model = PPO.load(self.model_path, 
                env, 
                verbose=1, 
                tensorboard_log=self.log_dir)

        # do not update them at test time
        env.training = False
        env.norm_reward = True
        env.norm_obs = True
                
        writer = tf.summary.create_file_writer(self.log_eval)

        self.__eval_tensofboard_grap(self.total_timesteps, env, model, writer)
        
        env.close()

    def __eval_tensofboard_grap(self, steps, env, model, writer):
        obs = env.reset()
        rewards_per_episode = []

        for step in range(steps):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            rewards_per_episode.append(reward)

            # Print stats every 100 calls
            if (step) % 100 == 0 or done:    
                with writer.as_default():
                    tf.summary.scalar(self.eval_scalar_name+'avg_rewards', 
                        np.mean(rewards_per_episode), 
                        step=step)

                    tf.summary.scalar(self.eval_scalar_name+'errors', 
                        info[0]['errors']['total'],
                        step=step)
        
            if reward < 0:
                self.error_file.write(
                    f"{info[0]['errors']['obs_min']}, {info[0]['errors']['obs_max']}, {info[0]['errors']['mean']}, {info[0]['errors']['variance']}, {info[0]['errors']['status']}\n")
                env.render()

        writer.flush()
