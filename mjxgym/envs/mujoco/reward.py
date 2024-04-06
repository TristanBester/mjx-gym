import chex
import jax.numpy as jnp
import numpy as np

from mjxgym.envs.mujoco.types import State
from mjxgym.interfaces.reward import RewardFunction


class DenseRewardFunction(RewardFunction[State]):
    def __call__(self, state: State, action: chex.Array, next_state: State) -> float:
        """Map the (s, a, s') tuple to a scalar reward value.

        The reward is the negative distance between the agent and the goal.
        """
        dist_to_goal = jnp.linalg.norm(
            next_state.mjx_data.qpos - next_state.goal_position
        )
        return -dist_to_goal
