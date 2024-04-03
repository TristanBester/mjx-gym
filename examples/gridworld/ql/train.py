import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax_tqdm import loop_tqdm

from examples.gridworld.ql.qlearning import EpsilonGreedy, QLearning
from mjxgym.envs.gridworld.env import GridWorld
from mjxgym.envs.gridworld.generator import GridWorldGenerator
from mjxgym.wrappers.wrappers import AutoResetWrapper


@jax.jit
def train_agent(key: chex.PRNGKey, n_steps: int = 200000) -> chex.Array:
    """Train agent using Q-learning algorithm."""
    generator = GridWorldGenerator(approx_grid_size=10)
    env = AutoResetWrapper(GridWorld(generator, max_steps=100))
    qlearning = QLearning(learning_rate=0.1)
    epsilon_greedy = EpsilonGreedy(eps=0.1, n_actions=4)

    grid_size = env._env.generator.grid_size
    q_values = jnp.zeros((grid_size, grid_size, 4))
    returns = jnp.full((n_steps,), -1.0)
    ep_counter = 0

    @loop_tqdm(n_steps)
    def fori_body(_, val):
        q_values, key, state, timestep, ep_counter, returns = val

        key, subkey = jax.random.split(key)
        action = epsilon_greedy(subkey, q_values, state.agent_pos)
        next_state, next_timestep = env.step(state, action)
        q_values = qlearning.update(q_values, timestep, action, next_timestep)

        ep_counter, returns = jax.lax.cond(
            next_timestep.is_last(),
            lambda c, a, t: (c + 1, a.at[c].set(t.reward)),
            lambda c, a, _: (c, a),
            ep_counter,
            returns,
            next_timestep,
        )

        return (
            q_values,
            key,
            next_state,
            next_timestep,
            ep_counter,
            returns,
        )

    state, timestep = env.reset(key)
    q_values, _, _, _, ep_counter, returns = jax.lax.fori_loop(
        0,
        n_steps,
        fori_body,
        (q_values, key, state, timestep, ep_counter, returns),
    )
    return q_values, ep_counter, returns


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    q_values, ep_counter, returns = train_agent(key)
    print(q_values.max())
    print(ep_counter)
    print(returns[ep_counter - 100 : ep_counter + 10])

    print(q_values.max())

    for row in range(11):
        for col in range(11):
            print(f"{jnp.max(q_values[row, col, :]):.3f}", end=" ")
        print()

    print()
    directions = ["→", "←", "↑", "↓"]
    for row in range(11):
        for col in range(11):
            mx_val = jnp.max(q_values[row, col, :])
            d = jnp.argmax(q_values[row, col, :])

            if mx_val == 0:
                print("x", end="  ")
            else:
                print(directions[d], end="  ")
        print()

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

    returns = returns[:ep_counter]
    window_size = 100
    returns_ma = moving_average(returns, window_size)
    returns_ma = returns_ma[window_size // 2 : -window_size // 2]

    plt.plot(returns_ma)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()
