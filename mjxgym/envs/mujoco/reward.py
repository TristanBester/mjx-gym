import chex
import jax.numpy as jnp

from mjxgym.envs.mujoco.constants import EE_SITE_IDX, GOAL_SITE_IDX
from mjxgym.envs.mujoco.types import State
from mjxgym.interfaces.reward import RewardFunction


class DenseRewardFunction(RewardFunction[State]):
    def __call__(self, state: State, action: chex.Array, next_state: State) -> float:
        """Map the (s, a, s') tuple to a scalar reward value.

        The reward is the negative distance between the agent and the goal.
        """
        goal_position = next_state.mjx_data.site_xpos[GOAL_SITE_IDX]
        ee_position = next_state.mjx_data.site_xpos[EE_SITE_IDX]
        dist_to_goal = jnp.linalg.norm(goal_position - ee_position)
        return -dist_to_goal
