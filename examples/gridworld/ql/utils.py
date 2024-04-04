import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_value_function(q_values: chex.Array, path: str) -> None:
    """Plot the Q-value function of the gridworld."""
    q_values = jnp.max(q_values, axis=-1)
    grid_size = q_values.shape[0]

    fig = plt.gcf()
    ax = fig.gca()
    ax.clear()
    ax.set_xticks(jnp.arange(-0.5, grid_size - 1, 1), minor=True)
    ax.set_yticks(jnp.arange(-0.5, grid_size - 1, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()

    # Render the grid
    for row in range(grid_size):
        for col in range(grid_size):
            tile_value = q_values[row, col]
            rect = plt.Rectangle(
                [col - 0.5, row - 0.5], 1, 1, color=plt.cm.cool(tile_value)
            )
            ax.add_patch(rect)
            ax.text(col, row, f"{tile_value:.2f}", ha="center", va="center")
    ax.set_title("Value Function")
    plt.savefig(path, dpi=300)
    print("Saved value function plot to", path)


def plot_policy(q_values: chex.Array, path: str) -> None:
    """Plot the policy of the gridworld."""
    actions = jnp.argmax(q_values, axis=-1)
    q_values = jnp.max(q_values, axis=-1)
    grid_size = q_values.shape[0]
    directions = ["→", "←", "↑", "↓"]

    fig = plt.gcf()
    ax = fig.gca()
    ax.clear()
    ax.set_xticks(jnp.arange(-0.5, grid_size - 1, 1), minor=True)
    ax.set_yticks(jnp.arange(-0.5, grid_size - 1, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()

    # Render the grid
    for row in range(grid_size):
        for col in range(grid_size):

            if q_values[row, col] == 0:
                rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color="black")
                ax.add_patch(rect)
            else:
                tile_value = directions[actions[row, col]]
                rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color="white")
                ax.add_patch(rect)
                ax.text(col, row, f"{tile_value}", ha="center", va="center")
    ax.set_title("Policy")
    plt.savefig(path, dpi=300)
    print("Saved policy plot to", path)


def plot_returns(returns: chex.Array, path: str):
    """Plot the returns over time."""

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

    window_size = 100
    returns_ma = moving_average(returns, window_size)
    returns_ma = returns_ma[window_size // 2 : -window_size // 2]

    plt.gca().clear()
    plt.plot(returns_ma)
    plt.title("Returns over time")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(path, dpi=300)
    print("Saved returns plot to", path)
