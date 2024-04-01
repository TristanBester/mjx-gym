from functools import cached_property

import jax
import jax.numpy as jnp

from mjxgym.constants import WALLS
from mjxgym.type import State


class GridWorldGenerator:
    def __init__(self, grid_size=5) -> None:
        self.grid_size = grid_size

    @cached_property
    def grid(self):
        """Grid is static, so we can cache it."""
        grid = jnp.zeros((5, 5), dtype=jnp.int32)
        grid = grid.at[tuple(WALLS)].set(1)
        return grid

    def __call__(self, key) -> State:
        def cond_fun(grid, agent_pos):
            return grid[tuple(agent_pos)] == 1

        def loop_body(key, grid, agent_pos):
            key, subkey = jax.random.split(key)
            agent_pos = jax.random.randint(
                key=subkey,
                shape=(2,),
                minval=0,
                maxval=5,
                dtype=jnp.int32,
            )
            return key, grid, agent_pos

        key, subkey = jax.random.split(key)
        _, _, agent_pos = jax.lax.while_loop(
            cond_fun=lambda args: cond_fun(args[1], args[2]),
            body_fun=lambda args: loop_body(*args),
            init_val=(
                key,
                self.grid,
                jax.random.randint(
                    key=subkey,
                    shape=(2,),
                    minval=0,
                    maxval=5,
                    dtype=jnp.int32,
                ),
            ),
        )
        return State(
            key=key,
            step_count=0,
            grid=self.grid,
            agent_pos=agent_pos,
        )
