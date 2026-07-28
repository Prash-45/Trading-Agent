import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd
import random


class TradingEnvV2(gym.Env):
    """
    Research-grade Trading Environment

    Action Space
    ------------
    Continuous:
        -1.0  -> Sell all holdings
         0.0  -> Hold
        +1.0  -> Invest all available cash

    Observation
    -----------
    Market Features
    +
    Portfolio Features
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframes,
        initial_balance=10000,
        transaction_cost=0.001,
    ):

        super().__init__()

        # -------------------------
        # Store datasets
        # -------------------------
        self.dataframes = [
            df.reset_index(drop=True)
            for df in dataframes
        ]

        self.df = self.dataframes[0]

        # -------------------------
        # Trading parameters
        # -------------------------
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # -------------------------
        # Portfolio state
        # -------------------------
        self.cash = initial_balance
        self.shares = 0.0
        self.portfolio_value = initial_balance

        self.current_step = 0

        # -------------------------
        # Market features
        # -------------------------
        self.feature_columns = [
            col
            for col in self.df.columns
            if col not in ["Date", "Ticker"]
        ]

        # ===================================================
        # CONTINUOUS ACTION SPACE
        #
        # -1 -> Sell everything
        #  0 -> Hold
        # +1 -> Invest everything
        # ===================================================
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # ===================================================
        # OBSERVATION SPACE
        #
        # Market Features
        # + Cash Ratio
        # + Shares Ratio
        # + Portfolio Ratio
        # ===================================================
        obs_size = len(self.feature_columns) + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )



    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
    
        # -------------------------
        # Randomly select one stock
        # -------------------------
        self.df = random.choice(self.dataframes).reset_index(drop=True)
    
        # -------------------------
        # Reset episode state
        # -------------------------
        self.current_step = 0
    
        self.cash = self.initial_balance
        self.shares = 0.0
        self.portfolio_value = self.initial_balance
    
        # -------------------------
        # Market Features
        # -------------------------
        market_features = (
            self.df
            .loc[self.current_step, self.feature_columns]
            .values
            .astype(np.float32)
        )
    
        # -------------------------
        # Portfolio Features
        # -------------------------
        portfolio_features = np.array([
            self.cash / self.initial_balance,
            self.shares / 100,
            self.portfolio_value / self.initial_balance
        ], dtype=np.float32)
    
        # -------------------------
        # Final Observation
        # -------------------------
        observation = np.concatenate([
            market_features,
            portfolio_features
        ])
    
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
        }
    
        return observation, info

    def step(self, action):

        # Convert numpy array to scalar
        action = float(action[0])
    
        current_price = self.df.loc[self.current_step, "Close"]
    
        previous_portfolio = self.portfolio_value
    
        # =====================================================
        # BUY
        # action > 0
        # =====================================================
        if action > 0:
    
            # Fraction of cash to invest
            invest_amount = self.cash * action
    
            # Commission
            invest_amount_after_fee = invest_amount * (1 - self.transaction_cost)
    
            # Shares purchased
            shares_to_buy = invest_amount_after_fee / current_price
    
            self.cash -= invest_amount
    
            self.shares += shares_to_buy
    
        # =====================================================
        # SELL
        # action < 0
        # =====================================================
        elif action < 0:
    
            # Fraction of holdings to sell
            shares_to_sell = self.shares * abs(action)
    
            proceeds = shares_to_sell * current_price
    
            proceeds_after_fee = proceeds * (1 - self.transaction_cost)
    
            self.cash += proceeds_after_fee
    
            self.shares -= shares_to_sell
    
        # =====================================================
        # HOLD
        # =====================================================
        # action == 0
        # Nothing happens
    
        # -----------------------------------------------------
        # Move to next day
        # -----------------------------------------------------
        self.current_step += 1
    
        terminated = self.current_step >= len(self.df) - 1
    
        truncated = False
    
        new_price = self.df.loc[self.current_step, "Close"]
    
        self.portfolio_value = (
            self.cash +
            self.shares * new_price
        )
    
        # =====================================================
        # Reward
        # =====================================================
        reward = (
            self.portfolio_value - previous_portfolio
        ) / previous_portfolio
    
        # Small penalty for trading
        if abs(action) > 0.05:
            reward -= 0.0005
    
        # =====================================================
        # Observation
        # =====================================================
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
    
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
        }
    
        return observation, reward, terminated, truncated, info

    
    def render(self):

        current_price = self.df.loc[self.current_step, "Close"]
    
        print("=" * 50)
        print(f"Step            : {self.current_step}")
        print(f"Price           : {current_price:.2f}")
        print(f"Cash            : {self.cash:.2f}")
        print(f"Shares          : {self.shares:.4f}")
        print(f"Portfolio Value : {self.portfolio_value:.2f}")
        print("=" * 50)


