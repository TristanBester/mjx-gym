import chex
import jax
import jax.numpy as jnp
import mujoco
from chex._src.pytypes import PRNGKey
from mujoco import mjx

from mjxgym.envs.mujoco.constants import XML_PATH
from mjxgym.envs.mujoco.types import State
from mjxgym.interfaces.generator import Generator


class GeneratorReacher2D(Generator[State]):
    def __init__(self):
        # Load using the MuJoCo entrypoint
        self.mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Send to accelataor
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

    def __call__(self, key: chex.PRNGKey) -> State:
        # Here we will move the target position to a random location
        goal_position = jnp.array([0.3, 0.3])
        key, subkey = jax.random.split(key)
        return State(
            key=subkey,
            step_count=0,
            mjx_model=self.mjx_model,
            mjx_data=self.mjx_data,
            goal_position=goal_position,
        )
