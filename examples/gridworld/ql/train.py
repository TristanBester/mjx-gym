import chex
import jax
import jax.numpy as jnp
from jax_tqdm import loop_tqdm

from examples.gridworld.ql.qlearning import EpsilonGreedy, QLearning
from examples.gridworld.ql.utils import plot_policy, plot_returns, plot_value_function
from mjxgym.envs.gridworld.env import GridWorld
from mjxgym.envs.gridworld.generator import GridWorldGenerator
from mjxgym.envs.gridworld.renderer import GridWorldRenderer
from mjxgym.types.timestep import TimeStep
from mjxgym.wrappers.wrappers import AutoResetWrapper


class ReturnsManager:
    def init(self, max_episodes: int) -> tuple[int, chex.Array]:
        returns = jnp.full((max_episodes,), 0.0, dtype=jnp.float32)
        ep_counter = 0
        return ep_counter, returns

    def update_returns(
        self, returns_buffer: chex.Array, ep_counter: int, timestep: TimeStep
    ) -> tuple[int, chex.Array]:
        curr_ep_return = returns_buffer[ep_counter]
        updated_ep_return = curr_ep_return + timestep.reward
        returns_buffer = returns_buffer.at[ep_counter].set(updated_ep_return)
        ep_counter = jax.lax.cond(
            timestep.is_last(),
            lambda c: c + 1,
            lambda c: c,
            ep_counter,
        )
        return ep_counter, returns_buffer


@jax.jit
def train_agent(key: chex.PRNGKey, n_steps: int = 200000) -> chex.Array:
    """Train agent using Q-learning algorithm."""
    generator = GridWorldGenerator(approx_grid_size=10)
    env = AutoResetWrapper(GridWorld(generator, max_steps=100))
    qlearning = QLearning(learning_rate=0.1)
    epsilon_greedy = EpsilonGreedy(eps=0.1, n_actions=4)

    grid_size = env._env.generator.grid_size
    q_values = jnp.zeros((grid_size, grid_size, 4))

    returns_manager = ReturnsManager()
    ep_counter, returns_buffer = returns_manager.init(max_episodes=n_steps)

    @loop_tqdm(n_steps)
    def fori_body(_, val):
        q_values, key, state, timestep, ep_counter, returns_buffer = val

        key, subkey = jax.random.split(key)
        action = epsilon_greedy(subkey, q_values, state.agent_pos)
        next_state, next_timestep = env.step(state, action)
        q_values = qlearning.update(q_values, timestep, action, next_timestep)

        ep_counter, returns_buffer = returns_manager.update_returns(
            returns_buffer, ep_counter, next_timestep
        )
        return (
            q_values,
            key,
            next_state,
            next_timestep,
            ep_counter,
            returns_buffer,
        )

    state, timestep = env.reset(key)
    q_values, _, _, _, ep_counter, returns_buffer = jax.lax.fori_loop(
        0,
        n_steps,
        fori_body,
        (q_values, key, state, timestep, ep_counter, returns_buffer),
    )
    return q_values, ep_counter, returns_buffer


def visualise_trajectory(q_values: chex.Array) -> None:
    """Visualise the trajectory of the agent."""
    key = jax.random.PRNGKey(2)
    states = []

    renderer = GridWorldRenderer()
    generator = GridWorldGenerator(approx_grid_size=10)
    env = GridWorld(generator, max_steps=100)

    state, timestep = env.reset(key)
    states.append(state)

    while not timestep.is_last():
        action = jnp.argmax(q_values[tuple(state.agent_pos)])
        state, timestep = env.step(state, action)
        states.append(state)
    renderer.animate_trajectory(states, "results/trajectory.mp4")


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    q_values, ep_counter, returns_buffer = train_agent(key)
    returns = returns_buffer[:ep_counter]

    plot_value_function(q_values, "results/value_function.png")
    plot_policy(q_values, "results/policy.png")
    plot_returns(returns, "results/returns.png")
    visualise_trajectory(q_values)
