import jax.numpy as jnp

UP = jnp.array([-1, 0])
DOWN = jnp.array([1, 0])
LEFT = jnp.array([0, -1])
RIGHT = jnp.array([0, 1])
MOVES = jnp.array([RIGHT, LEFT, UP, DOWN])
WALLS = jnp.array(
    [
        [1, 2],
        [2, 2],
        [3, 2],
        [2, 3],
        [2, 1],
    ]
).T.reshape(2, -1)
