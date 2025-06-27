import os
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from enum import IntEnum
from dotenv import load_dotenv
from loaders.scalers import ProcessedAssetData

# Load .env file
load_dotenv()

# Fetch environment variables with type casting and defaults
def get_env_float(var, default=None):
    value = os.getenv(var)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required environment variable: {var}")
    return float(value)

CAPITAL = get_env_float("CAPITAL", default=10000)
COST = get_env_float("COST", default=0.001)
NEG_MUL = get_env_float("NEG_MUL", default=1.5)

class Action(IntEnum):
    SELL = -1
    HOLD = 0
    BUY = 1


class SingleAssetTradingEnvironment:
    """
    A trading environment for a single asset using discrete actions.

    Action Space:
        -1: Sell
         0: Hold
         1: Buy

    State includes normalized price features plus:
        - Capital ratio
        - Running capital ratio
        - Asset value ratio
        - Previous action
    """

    def __init__(
            self,
            asset_data: ProcessedAssetData,
            initial_money: float = CAPITAL,
            trans_cost: float = COST,
            store_flag: bool = True,
            asset_ph: float = 0.0,
            capital_frac: float = 0.2,
            running_thresh: float = 0.1,
            cap_thresh: float = 0.3
    ):
        """
        Initialize the trading environment.

        Args:
            asset_data: Asset data object that supports indexing and has a scaler and frame.
            initial_money: Starting capital.
            trans_cost: Transaction cost rate (e.g. 0.001).
            store_flag: Whether to track and store step-by-step history.
            asset_ph: Initial asset holding (e.g. from previous trading episodes).
            capital_frac: Fraction of capital to invest on each buy.
            running_thresh: Capital threshold under which no investment is allowed.
            cap_thresh: Terminal threshold for capital loss to end episode early.
        """
        self.asset_data = asset_data
        self.scaler = asset_data.scaler
        self.terminal_idx: int = len(asset_data) - 1

        self.initial_capital: float = initial_money
        self.transaction_cost: float = trans_cost
        self.store_flag: bool = store_flag

        self.capital_frac: float = capital_frac
        self.running_thresh: float = running_thresh
        self.cap_thresh: float = cap_thresh

        self.capital: float = 0.0
        self.running_capital: float = 0.0
        self.asset_holdings: float = asset_ph

        self.current_step: int = 0
        self.prev_action: int = 0
        self.current_action: int = 0
        self.current_reward: float = 0.0
        self.current_price: float = 0.0
        self.done: bool = False
        self.current_state: Optional[np.ndarray] = None
        self.next_return: float = 0.0
        self.store: Dict[str, List[float]] = {}

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: The initial observation state for the agent.
        """
        self.capital: float = self.initial_capital
        self.running_capital: float = self.capital
        self.asset_holdings: float = 0.0

        self.current_step: int = 0
        self.prev_action: int = 0
        self.current_action: int = 0
        self.current_reward: float = 0.0
        self.current_price: float = float(self.asset_data.frame.iloc[self.current_step]['Adj Close'])
        self.done: bool = False

        self.next_return, self.current_state = self.get_state(self.current_step)

        if self.store_flag:
            self.store: Dict[str, List[float]] = {
                "action_store": [],
                "reward_store": [],
                "running_capital": [],
                "port_ret": []
            }

        return self.current_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, List[float]]]]:
        """
        Advance the environment by one time step using the given action.

        Args:
            action (int): Action to take. Must be one of {-1, 0, 1}.

        Returns:
            Tuple:
                - np.ndarray: Next state observation.
                - float: Reward for the action taken.
                - bool: Whether the episode is done.
                - Optional[Dict[str, List[float]]]: Optional tracking info if store_flag is True.
        """
        self.current_action = action
        self.current_price = float(self.asset_data.frame.iloc[self.current_step]['Adj Close'])
        self.current_reward = self.calculate_reward()
        self.prev_action = self.current_action

        self.current_step += 1
        self.done = self.check_terminal()

        if not self.done:
            self.next_return, self.current_state = self.get_state(self.current_step)
        else:
            reward_offset: float = 0.0
            if self.store_flag and len(self.store["running_capital"]) > 1:
                ret = (self.store["running_capital"][-1] / self.store["running_capital"][-2]) - 1
                reward_offset += 10 * ret
            if self.current_step < self.terminal_idx:
                reward_offset += -1 * max(0.5, 1 - self.current_step / self.terminal_idx)

            self.current_reward += reward_offset

        info: Optional[Dict[str, List[float]]] = None
        if self.store_flag:
            self.store["action_store"].append(self.current_action)
            self.store["reward_store"].append(self.current_reward)
            self.store["running_capital"].append(self.capital)
            info = self.store

        return self.current_state, self.current_reward, self.done, info

    def calculate_reward(self) -> float:
        """
        Calculate the reward from the current action.

        Includes transaction cost penalty, position update logic,
        and a multiplier for negative rewards to encourage risk aversion.

        Returns:
            float: The calculated reward.
        """
        investment: float = self.running_capital * self.capital_frac
        reward_offset: float = 0.0

        if self.current_action == Action.BUY:
            if self.running_capital > self.initial_capital * self.running_thresh:
                units_bought = investment / self.current_price
                self.asset_holdings += units_bought
                self.running_capital -= investment
                self.current_price *= (1 - self.transaction_cost)

        elif self.current_action == Action.SELL:
            if self.asset_holdings > 0:
                proceeds = self.asset_holdings * self.current_price * (1 - self.transaction_cost)
                self.running_capital += proceeds
                self.asset_holdings = 0

        elif self.current_action == Action.HOLD and self.prev_action == Action.HOLD:
            reward_offset -= 0.1

        prev_capital = self.capital
        self.capital = self.running_capital + self.asset_holdings * self.current_price

        reward = (
            100 * self.next_return * float(self.current_action)
            - abs(float(self.current_action - self.prev_action)) * self.transaction_cost
        )

        if reward < 0:
            reward *= NEG_MUL

        reward += reward_offset

        if self.store_flag:
            self.store['port_ret'].append((self.capital - prev_capital) / prev_capital)

        return reward

    def check_terminal(self) -> bool:
        """
        Check if the episode should terminate.

        Conditions:
            - Reached end of data
            - Capital dropped below threshold

        Returns:
            bool: True if episode is done, False otherwise.
        """
        return (
            self.current_step == self.terminal_idx
            or self.capital <= self.initial_capital * self.cap_thresh
        )

    def get_state(self, idx: int) -> Tuple[float, np.ndarray]:
        """
        Construct the state at a given index.

        Args:
            idx (int): Index in the time series.

        Returns:
            Tuple:
                - float: Next return value (used in reward calculation).
                - np.ndarray: State vector for agent input.
        """
        state_vec: np.ndarray = self.asset_data[idx][1:]
        scaled_state: np.ndarray = self.scaler.transform(state_vec.reshape(1, -1))

        custom_features = np.array([
            self.capital / self.initial_capital,
            self.running_capital / self.capital,
            self.asset_holdings * self.current_price / self.initial_capital,
            self.prev_action
        ]).reshape(1, -1)

        state: np.ndarray = np.concatenate([scaled_state, custom_features], axis=-1)
        next_ret: float = self.asset_data[idx][0]

        return next_ret, state