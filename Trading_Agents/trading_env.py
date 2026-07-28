import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd

import random

class TradingEnv(gym.Env):

    def __init__(self,dataframes,initial_balance=10000,transaction_cost=0.001):

        super().__init__()

        self.dataframes = [df.reset_index(drop=True) for df in dataframes]

        # Start with the first stock
        self.df = self.dataframes[0]

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.current_step = 0

        self.cash = initial_balance

        self.shares = 0

        self.portfolio_value = initial_balance

        self.feature_columns = [
            col for col in self.df.columns
            if col not in ["Date", "Ticker"]
        ]

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) + 3,),
            dtype=np.float32
        )


    
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Randomly choose one stock for this episode
        self.df = random.choice(self.dataframes).reset_index(drop=True)
    
        self.current_step = 0
    
        self.cash = self.initial_balance
    
        self.shares = 0
    
        self.portfolio_value = self.initial_balance
    
        market_features = (
            self.df
            .loc[self.current_step, self.feature_columns]
            .values
            .astype(np.float32)
        )

        portfolio_features = np.array([
        self.cash / self.initial_balance,
        self.shares / 100,
        self.portfolio_value / self.initial_balance
        ], dtype=np.float32)

        observation = np.concatenate([
            market_features,
            portfolio_features
        ])
    
        info = {}
    
        return observation, info


    
    def step(self, action):

        current_price = self.df.loc[self.current_step, "Close"]

        previous_portfolio = self.portfolio_value

        if action == 1:

            # Price including commission
            buy_cost = current_price * (1 + self.transaction_cost)

            if self.cash >= buy_cost:

                self.cash -= buy_cost

                self.shares += 1
            
        elif action == 2:

            if self.shares > 0:

                # Money received after commission
                sell_value = current_price * (1 - self.transaction_cost)
    
                self.cash += sell_value
    
                self.shares -= 1

        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1

        truncated = False

        new_price = self.df.loc[self.current_step, "Close"]

        self.portfolio_value = (

            self.cash +

            self.shares * new_price

        )

        reward = (self.portfolio_value - previous_portfolio) / previous_portfolio

        # Small penalty for every trade
        if action in [1, 2]:
            reward -= 0.001

        market_features = (
            self.df
            .loc[self.current_step, self.feature_columns]
            .values
            .astype(np.float32)
        )

        portfolio_features = np.array([
        self.cash / self.initial_balance,
        self.shares / 100,
        self.portfolio_value / self.initial_balance
        ], dtype=np.float32)

        observation = np.concatenate([
            market_features,
            portfolio_features
        ])

        info = {}

        return observation, reward, terminated, truncated, info
    




    def render(self):

        current_price = self.df.loc[self.current_step, "Close"]
    
        print(f"Step            : {self.current_step}")
        print(f"Current Price   : {current_price:.2f}")
        print(f"Cash            : {self.cash:.2f}")
        print(f"Shares Owned    : {self.shares}")
        print(f"Portfolio Value : {self.portfolio_value:.2f}")