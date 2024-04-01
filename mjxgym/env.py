import chex
import jax
import jax.numpy as jnp

from mjxgym.constants import MOVES
from mjxgym.generator import GridWorldGenerator
from mjxgym.type import Observation, State, restart, termination, transition


class GridWorld:
    def __init__(self) -> None:
        self.generator = GridWorldGenerator()

    def reset(self, key):
        key, subkey = jax.random.split(key)
        state = self.generator(subkey)
        obs = Observation(agent_pos=state.agent_pos)
        timestep = restart(obs, {})
        return state, timestep

    def step(self, state: State, action: chex.Array):
        next_agent_pos = self._move_agent(state.grid, state.agent_pos, action)
        done = jnp.array_equal(next_agent_pos, jnp.array([4, 4]))
        next_state = State(
            key=state.key,
            step_count=state.step_count + 1,
            grid=state.grid,
            agent_pos=next_agent_pos,
        )
        next_obs = Observation(agent_pos=next_agent_pos)
        reward = jax.lax.cond(
            done,
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


if __name__ == "__main__":
    env = GridWorld()
    key = jax.random.PRNGKey(0)

    # jit_reset = jax.jit(env.reset)
    # jit_step = jax.jit(env.step)

    jit_reset = jax.jit(env.reset, static_argnums=(0,))
    jit_step = jax.jit(env.step, static_argnums=(0,))

    vmap_reset = jax.vmap(jit_reset)
    vmap_step = jax.vmap(jit_step)

    keys = jax.random.split(key, 5)

    states, timesteps = vmap_reset(keys)

    print(states.agent_pos)

    states, timesteps = vmap_step(states, jnp.array([0, 0, 0, 0, 0]))

    print(states.agent_pos)
