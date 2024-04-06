import jax.numpy as jnp

XML_PATH = "assets/reacher.xml"

ACTIONS = jnp.array(
    [
        [-0.1, 0.1],
        [0.1, -0.1],
        [0.0, 0.1],
        [0.1, 0.0],
        [-0.1, 0.0],
        [0.0, -0.1],
        [-0.1, -0.1],
        [0.1, 0.1],
        [0.0, 0.0],
    ]
)
