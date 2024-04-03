import chex
import jax
import jax.numpy as jnp

from mjxgym.envs.gridworld.types import State
from mjxgym.interfaces.reward import RewardFunction


class SparseReward(RewardFunction[State]):
    def __call__(
        self, state: State, action: chex.Array, next_state: State
    ) -> jnp.float32:
        """Map the (s, a, s') tuple to a scalar reward value.

        Return 1.0 if the agent is in the goal position, 0.0 otherwise.
        """
        solved = jnp.array_equal(
            next_state.agent_pos,
            jnp.array([state.grid.shape[0] - 1, state.grid.shape[1] - 1]),
        )
        return jax.lax.cond(
            solved,
            lambda: jnp.array(1.0, jnp.float32),
            lambda: jnp.array(0.0, jnp.float32),
        )
