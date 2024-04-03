import jax.numpy as jnp

# Position modifiers for the four cardinal directions
_UP = jnp.array([-1, 0])
_DOWN = jnp.array([1, 0])
_LEFT = jnp.array([0, -1])
_RIGHT = jnp.array([0, 1])

# Move array for all four cardinal directions indexed by agent
MOVES = jnp.array(
    [
        _RIGHT,
        _LEFT,
        _UP,
        _DOWN,
    ]
)
