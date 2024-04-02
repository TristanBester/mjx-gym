import time

import chex
import jax
import jax.numpy as jnp

from mjxgym.constants import MOVES
from mjxgym.generator import GridWorldGenerator
from mjxgym.type import Observation, State, StepType, restart, termination, transition

# Fix typing and small helper functions present in original code


class GridWorld:
    def __init__(self) -> None:
        self.generator = GridWorldGenerator()
        self.max_steps = 1000

    def reset(self, key):
        key, subkey = jax.random.split(key)
        state = self.generator(subkey)
        obs = Observation(agent_pos=state.agent_pos)
        timestep = restart(obs, {})
        return state, timestep

    def step(self, state: State, action: chex.Array):
        next_agent_pos = self._move_agent(state.grid, state.agent_pos, action)
        next_state = State(
            key=state.key,
            step_count=state.step_count + 1,
            grid=state.grid,
            agent_pos=next_agent_pos,
        )
        next_obs = Observation(agent_pos=next_agent_pos)

        solved = jnp.array_equal(next_agent_pos, jnp.array([4, 4]))
        done = solved | (next_state.step_count >= self.max_steps)

        reward = jax.lax.cond(
            solved,
            lambda: jnp.array(1.0),
            lambda: jnp.array(0.0),
        )
        timestep = jax.lax.cond(
            done,
            lambda: termination(reward, next_obs, {}),
            lambda: transition(reward, next_obs, jnp.array(0.99), {}),
        )
        return next_state, timestep

    def _move_agent(self, grid, agent_pos, action):
        new_pos = agent_pos + MOVES[action]
        is_valid_move = jnp.all(
            (new_pos >= 0) & (new_pos < grid.shape[0]) & (grid[tuple(new_pos)] == 0)
        )
        return jax.lax.cond(
            is_valid_move,
            lambda: new_pos,
            lambda: agent_pos,
        )

    def render(self, state: State):
        render_grid = state.grid.at[tuple(state.agent_pos)].set(2)
        print(render_grid)
        print()


# if __name__ == "__main__":
#     env = AutoResetWrapper(GridWorld())

#     key = jax.random.PRNGKey(0)
#     state, timestep = env.reset(key)
#     state, timestep = env.step(state, jnp.array(0))

#     print(state, timestep)
