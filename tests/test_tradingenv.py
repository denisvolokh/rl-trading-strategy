import pytest
import pandas as pd
import numpy as np
from loaders.scalers import ProcessedAssetData
from envs.tradingenv import SingleAssetTradingEnvironment, Action

@pytest.fixture
def dummy_data():
    data = {
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [100, 101, 102, 103, 104],
        "Adj Close": [100, 101, 102, 103, 104],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data)
    df["forward_return"] = df["Close"].pct_change().shift(-1).fillna(0.0)

    return ProcessedAssetData(df)

def test_env_initialization(dummy_data):
    env = SingleAssetTradingEnvironment(asset_data=dummy_data)
    assert env.capital > 0, f"Expected capital to be > 0 but got {env.capital}"
    assert env.current_state is not None, "Expected current_state to be initialized"
    assert env.current_price > 0, f"Expected current_price to be > 0 but got {env.current_price}"

def test_env_reset(dummy_data):
    env = SingleAssetTradingEnvironment(asset_data=dummy_data)
    env.capital -= 500
    state = env.reset()
    assert env.capital == env.initial_capital, (
        f"Expected capital to be reset to initial_capital {env.initial_capital}, but got {env.capital}"
    )
    assert state.shape[1] == env.current_state.shape[1], (
        "State shape mismatch after reset"
    )

def test_env_step_returns(dummy_data):
    env = SingleAssetTradingEnvironment(asset_data=dummy_data)
    state, reward, done, info = env.step(Action.BUY)
    assert isinstance(state, np.ndarray), f"Expected state to be np.ndarray, got {type(state)}"
    assert isinstance(reward, float), f"Expected reward to be float, got {type(reward)}"
    assert isinstance(done, bool), f"Expected done to be bool, got {type(done)}"
    assert isinstance(info, (dict, type(None))), f"Expected info to be dict or None, got {type(info)}"

def test_env_terminal_on_last_step(dummy_data):
    env = SingleAssetTradingEnvironment(asset_data=dummy_data)
    while not env.done:
        _, _, done, _ = env.step(Action.HOLD)
    assert done is True, "Expected environment to be done after reaching terminal index"
    assert env.current_step == env.terminal_idx, (
        f"Expected current_step to be {env.terminal_idx}, got {env.current_step}"
    )

def test_reward_behavior(dummy_data):
    env = SingleAssetTradingEnvironment(asset_data=dummy_data)
    env.step(Action.BUY)
    env.step(Action.SELL)
    rewards = env.store["reward_store"]
    assert len(rewards) >= 2, f"Expected at least 2 rewards, got {len(rewards)}"
    assert any(r <= 0 for r in rewards), f"Expected at least one negative or zero reward, got {rewards}"