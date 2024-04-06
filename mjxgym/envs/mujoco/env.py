import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
from jax_tqdm import loop_tqdm
from mujoco import mjx

from mjxgym.envs.mujoco.constants import ACTIONS
from mjxgym.envs.mujoco.generator import GeneratorReacher2D
from mjxgym.envs.mujoco.renderer import Reacher2DRenderer
from mjxgym.envs.mujoco.reward import DenseRewardFunction
from mjxgym.envs.mujoco.types import Observation, State
from mjxgym.interfaces.environment import Environment
from mjxgym.interfaces.generator import Generator
from mjxgym.interfaces.reward import RewardFunction
from mjxgym.types.timestep import TimeStep
from mjxgym.utils.factory import create_initial_timestep, create_timestep


class Reacher2D(Environment[State, Observation]):
    """2D Reacher environment."""

    def __init__(
        self,
        generator: Generator = GeneratorReacher2D(),
        reward_function: RewardFunction = DenseRewardFunction(),
        discount: float = 0.99,
        max_steps: int = 1000,
    ):
        """Initialize the Reacher2D environment."""
        self.generator = generator
        self.reward_function = reward_function
        self.discount = discount
        self.max_steps = max_steps

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state using the given PRNG key."""
        key, subkey = jax.random.split(key)
        state = self.generator(subkey)
        obs = Observation(
            joint_1_angle=state.mjx_data.qpos[0],
            joint_2_angle=state.mjx_data.qpos[1],
            goal_position=state.goal_position,
        )
        timestep = create_initial_timestep(
            observation=obs,
        )
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        # Update the state with the action
        next_q_pos = state.mjx_data.qpos + ACTIONS[action]
        next_mjx_data = state.mjx_data.replace(qpos=next_q_pos)

        # Step the simulation
        next_mjx_data = mjx.step(state.mjx_model, next_mjx_data)

        # Update the system state
        next_state = State(
            key=state.key,
            step_count=state.step_count + 1,
            mjx_model=state.mjx_model,
            mjx_data=next_mjx_data,
            goal_position=state.goal_position,
        )

        # Aggregate timestep information
        reward = self.reward_function(state, action, next_state)
        done = self.max_steps <= state.step_count
        obs = Observation(
            joint_1_angle=next_state.mjx_data.qpos[0],
            joint_2_angle=next_state.mjx_data.qpos[1],
            goal_position=next_state.goal_position,
        )
        next_timestep = create_timestep(
            done=done,
            truncated=False,
            observation=obs,
            action=action,
            reward=reward,
            discount=self.discount,
        )
        return next_state, next_timestep


if __name__ == "__main__":
    # mj_model_one = mujoco.MjModel.from_xml_path("assets/reacher.xml")
    # mj_model_two = mujoco.MjModel.from_xml_path("assets/reacher.xml")

    # mj_data = mujoco.MjData(mj_model_one)
    # renderer = mujoco.Renderer(mj_model_two)

    # mujoco.mj_resetData(mj_model_one, mj_data)
    # mjx_data = mjx.put_data(mj_model_one, mj_data)
    # mjx_model = mjx.put_model(mj_model_one)

    # mjx_data = mjx.step(mjx_model, mjx_data)

    # mj_data = mjx.get_data(mj_model_two, mjx_data)
    # renderer.update_scene(data=mj_data)
    # pixels = renderer.render()
    # plt.imshow(pixels)
    # plt.show()

    #########################################

    env = Reacher2D()
    state, timestep = env.reset(jax.random.PRNGKey(0))

    @loop_tqdm(10_000_000)
    def fori_body(i, val):
        key, state, timestep = val

        key, subkey = jax.random.split(key)
        action = jax.random.randint(subkey, (), 0, 4)
        state, timestep = env.step(state, action)
        return key, state, timestep

    key, state, timestep = jax.lax.fori_loop(
        0, 10_000_000, fori_body, (jax.random.PRNGKey(0), state, timestep)
    )
