import chex
import jax
import jax.numpy as jnp
from mujoco import mjx

from mjxgym.envs.mujoco.constants import ACTIONS
from mjxgym.envs.mujoco.generator import GeneratorReacher2D
from mjxgym.envs.mujoco.reward import DenseRewardFunction
from mjxgym.envs.mujoco.types import Observation, State
from mjxgym.interfaces.environment import Environment
from mjxgym.interfaces.generator import Generator
from mjxgym.interfaces.reward import RewardFunction
from mjxgym.types.timestep import TimeStep
from mjxgym.utils.factory import create_initial_timestep, create_timestep


class Reacher2D(Environment[State, Observation]):
    """2D Reacher environment."""

    def __init__(
        self,
        generator: Generator = GeneratorReacher2D(),
        reward_function: RewardFunction = DenseRewardFunction(),
        discount: float = 0.99,
        max_steps: int = 1000,
    ):
        """Initialize the Reacher2D environment."""
        self.generator = generator
        self.reward_function = reward_function
        self.discount = discount
        self.max_steps = max_steps

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state using the given PRNG key."""
        key, subkey = jax.random.split(key)
        state = self.generator(subkey)
        obs = Observation(
            joint_angles=state.mjx_data.qpos,
            goal_position=state.goal_position,
        )
        timestep = create_initial_timestep(
            observation=obs,
        )
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        # Update the state with the action
        next_q_pos = state.mjx_data.qpos + ACTIONS[action]
        next_mjx_data = state.mjx_data.replace(qpos=next_q_pos)

        # Step the simulation
        next_mjx_data = mjx.step(state.mjx_model, next_mjx_data)

        # Update the system state
        next_state = State(
            key=state.key,
            step_count=state.step_count + 1,
            mjx_model=state.mjx_model,
            mjx_data=next_mjx_data,
            goal_position=state.goal_position,
        )

        # Aggregate timestep information
        reward = self.reward_function(state, action, next_state)
        done = jnp.abs(reward) < 0.05
        truncated = state.step_count >= self.max_steps

        obs = Observation(
            joint_angles=next_state.mjx_data.qpos,
            goal_position=next_state.goal_position,
        )
        next_timestep = create_timestep(
            done=done,
            truncated=truncated,
            observation=obs,
            action=action,
            reward=reward,
            discount=self.discount,
        )
        return next_state, next_timestep
