import chex
import jax
import jax.numpy as jnp

from mjxgym.envs.gridworld.types import State
from mjxgym.interfaces.generator import Generator


class GridWorldGenerator(Generator[State]):
    """Generate initial state for gridworld environment."""

    def __init__(self, approx_grid_size: int = 5):
        if approx_grid_size % 2 != 1:
            # Force grid size to be odd
            self.grid_size = approx_grid_size + 1
        else:
            self.grid_size = approx_grid_size
        self.grid = self._generate_grid()

    def __call__(self, key: chex.PRNGKey) -> State:
        """Sample initial state distribution using given PRNG key."""

        def cond_fun(
            grid: chex.Array,
            goal_pos: chex.Array,
            agent_pos: chex.Array,
        ) -> bool:
            """Return if the agent position is valid."""
            pos_invalid = grid[tuple(agent_pos)] == 1
            is_goal = jnp.array_equal(agent_pos, goal_pos)
            return pos_invalid | is_goal

        def body_fun(
            key: chex.PRNGKey,
            grid: chex.Array,
            goal_pos: chex.Array,
            agent_pos: chex.Array,
        ) -> chex.Array:
            """Sample a new agent position."""
            key, subkey = jax.random.split(key)
            agent_pos = jax.random.randint(
                subkey, minval=0, maxval=grid.shape[0], shape=(2,)
            )
            return key, grid, goal_pos, agent_pos

        # Sample random agent starting position
        key, subkey = jax.random.split(key)
        _, _, _, agent_pos = jax.lax.while_loop(
            cond_fun=lambda args: cond_fun(args[1], args[2], args[3]),
            body_fun=lambda args: body_fun(*args),
            init_val=(
                key,
                self.grid,
                jnp.array([self.grid_size - 1, self.grid_size - 1]),
                jax.random.randint(
                    subkey, minval=0, maxval=self.grid.shape[0], shape=(2,)
                ),
            ),
        )
        return State(
            key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            grid=self.grid,
            agent_pos=agent_pos,
        )

    def _generate_grid(self):
        """Generate a grid with walls and doors."""
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        mid_row, mid_col = self.grid_size // 2, self.grid_size // 2

        # Create walls
        grid = grid.at[mid_row, :].set(1)
        grid = grid.at[:, mid_col].set(1)

        # Create doors
        grid = grid.at[mid_row, 0].set(0)
        grid = grid.at[mid_row, -1].set(0)
        grid = grid.at[0, mid_col].set(0)
        grid = grid.at[-1, mid_col].set(0)
        return grid
