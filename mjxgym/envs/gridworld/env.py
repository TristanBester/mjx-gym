import chex
import jax
import jax.numpy as jnp

from mjxgym.envs.gridworld.constants import MOVES
from mjxgym.envs.gridworld.generator import GridWorldGenerator
from mjxgym.envs.gridworld.renderer import GridWorldRenderer
from mjxgym.envs.gridworld.reward import SparseReward
from mjxgym.envs.gridworld.types import Observation, State
from mjxgym.interfaces.environment import Environment
from mjxgym.interfaces.generator import Generator
from mjxgym.interfaces.render import Renderer
from mjxgym.interfaces.reward import RewardFunction
from mjxgym.types.timestep import TimeStep
from mjxgym.utils.factory import create_initial_timestep, create_timestep
from mjxgym.wrappers.wrappers import AutoResetWrapper


class GridWorld(Environment[State, Observation]):
    """Simple grid world environment.

    The agent starts at the top-left corner of grid with walls and must
    reach the bottom-right corner.
    """

    def __init__(
        self,
        generator: Generator[State] = GridWorldGenerator(),
        reward_function: RewardFunction[State] = SparseReward(),
        max_steps: int = 100,
    ) -> None:
        """Initialize the GridWorld environment."""
        self.generator = generator
        self.reward_function = reward_function
        self.max_steps = max_steps

    def reset(self, key: chex.Array) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state using the given PRNG key."""
        key, subkey = jax.random.split(key)
        state = self.generator(subkey)
        obs = Observation(agent_pos=state.agent_pos)
        timestep = create_initial_timestep(
            observation=obs,
        )
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        """Execute the given action in the specified environment state."""
        next_agent_pos = self._execute_action(state.grid, state.agent_pos, action)
        solved = jnp.array_equal(
            next_agent_pos,
            jnp.array([state.grid.shape[0] - 1, state.grid.shape[1] - 1]),
        )
        truncated = (state.step_count >= self.max_steps) & (~solved)
        done = solved | truncated

        next_state = State(
            key=state.key,
            step_count=state.step_count + 1,
            grid=state.grid,
            agent_pos=next_agent_pos,
        )
        next_obs = Observation(
            agent_pos=next_agent_pos,
        )
        reward = self.reward_function(state, action, next_state)
        timestep = create_timestep(
            done=done,
            truncated=truncated,
            observation=next_obs,
            reward=reward,
            discount=jnp.array(0.99),
        )
        return next_state, timestep

    def _execute_action(
        self, grid: chex.Array, agent_pos: chex.Array, action: chex.Array
    ) -> chex.Array:
        """Move the agent to the next position based on the action."""
        next_agent_pos = agent_pos + MOVES[action]
        within_bounds = jnp.all(
            (next_agent_pos >= 0) & (next_agent_pos < grid.shape[0])
        )

        # Test if collision valid
        wall_collision = grid[tuple(next_agent_pos)] == 1
        is_valid_move = within_bounds & ~wall_collision
        # Return updated position if valid move, else return current position
        return jax.lax.cond(
            is_valid_move,
            lambda: next_agent_pos,
            lambda: agent_pos,
        )


if __name__ == "__main__":
    env = GridWorld(
        generator=GridWorldGenerator(approx_grid_size=10),
        reward_function=SparseReward(),
    )
    env = AutoResetWrapper(env)
    key = jax.random.PRNGKey(0)
    states = []
    state, timestep = env.reset(key)

    state = state.replace(agent_pos=jnp.array([1, 1]))
    state = state.replace(step_count=1000)
    action = jnp.array(0)
    state, timestep = env.step(state, action)

    print(timestep)
