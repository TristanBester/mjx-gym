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

# Index of goal site in MuJoCo site data arrays
GOAL_SITE_IDX = 0
# Index of end effector site in MuJoCo site data arrays
EE_SITE_IDX = 1
