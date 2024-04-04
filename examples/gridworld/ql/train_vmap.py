import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax_tqdm import loop_tqdm

from examples.gridworld.ql.qlearning import EpsilonGreedy, QLearning, ReturnsManager
from examples.gridworld.ql.utils import plot_policy, plot_value_function
from mjxgym.envs.gridworld.env import GridWorld
from mjxgym.envs.gridworld.generator import GridWorldGenerator
from mjxgym.types.timestep import StepType
from mjxgym.wrappers.wrappers import VmapAutoResetWrapper


def train_agents(n_agents: int, n_steps: int = 200000):
    generator = GridWorldGenerator(approx_grid_size=11)
    env = VmapAutoResetWrapper(GridWorld(generator, max_steps=100))
    eps_greedy = EpsilonGreedy(eps=0.1, n_actions=4)
    q_learning = QLearning(learning_rate=0.1)
    returns_manager = ReturnsManager()

    eps_greedy_vmap = jax.vmap(eps_greedy.__call__)
    q_learning_vmap = jax.vmap(q_learning.update)
    update_returns_vmap = jax.vmap(returns_manager.update_returns)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_agents)
    q_values = jnp.zeros((n_agents, 11, 11, 4))
    returns_buffers = jnp.full((n_agents, n_steps), 0.0, dtype=jnp.float32)
    ep_counters = jnp.zeros(n_agents, dtype=jnp.int32)

    @loop_tqdm(n_steps)
    def fori_body(_: int, val: tuple) -> tuple:
        q_values, key, states, timesteps, ep_counters, returns_buffers = val

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, n_agents)
        actions = eps_greedy_vmap(subkeys, q_values, states.agent_pos)
        next_states, next_timesteps = env.step(states, actions)
        q_values = q_learning_vmap(q_values, timesteps, actions, next_timesteps)
        ep_counters, returns_buffers = update_returns_vmap(
            returns_buffers, ep_counters, next_timesteps
        )

        return (
            q_values,
            key,
            next_states,
            next_timesteps,
            ep_counters,
            returns_buffers,
        )

    states, timesteps = env.reset(keys)
    q_values, _, _, _, ep_counters, returns_buffers = jax.lax.fori_loop(
        0,
        n_steps,
        fori_body,
        (q_values, key, states, timesteps, ep_counters, returns_buffers),
    )
    return q_values, ep_counters, returns_buffers


if __name__ == "__main__":
    q_values, ep_counters, returns_buffers = train_agents(n_agents=10, n_steps=1000000)
    plot_policy(q_values[0], "results/vmap_policy.png")
    plot_value_function(q_values[0], "results/vmap_value_function.png")

    all_returns = []
    max_ep_count = jnp.max(ep_counters)
    for i in range(len(returns_buffers)):
        agent_returns = returns_buffers[i, : ep_counters[i]]
        agent_returns = jnp.pad(
            agent_returns,
            (0, max_ep_count - len(agent_returns)),
            mode="constant",
            constant_values=agent_returns[-1],
        )
        all_returns.append(agent_returns)
    all_returns = jnp.stack(all_returns)

    def moving_average(arr, window_size):
        if window_size % 2 == 0:
            # window size is required to be odd
            window_size += 1

        weights = jnp.ones(window_size) / window_size
        result = jnp.convolve(arr, weights, mode="valid")

        # Adjust the length of the result to match the original array
        padding = (window_size - 1) // 2
        result = jnp.concatenate([arr[:padding], result, arr[-padding:]])
        return result

    window_size = 2000

    for returns in all_returns:
        returns_ma = moving_average(returns, window_size)
        returns_ma = returns_ma[window_size // 2 : -window_size // 2]

        plt.plot(returns_ma)

    plt.title("Returns over time")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()
