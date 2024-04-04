import chex
import jax
import jax.numpy as jnp

from mjxgym.types.timestep import TimeStep


class QLearning:
    """Q-Learning algorithm."""

    def __init__(self, learning_rate: float):
        """Initialize Q-Learning algorithm parameters."""
        self.learning_rate = learning_rate

    def update(
        self,
        q_values: chex.Array,
        timestep: TimeStep,
        action: int,
        next_timestep: TimeStep,
    ) -> chex.Array:
        """Update the Q-values using Q-Learning rule."""
        state_action_idx = jnp.append(timestep.observation.agent_pos, action)
        curr_val = q_values[tuple(state_action_idx)]

        # This handles the last step as discount is 0 if next_step is terminal
        td_target = next_timestep.reward + next_timestep.discount * jnp.max(
            q_values[tuple(next_timestep.observation.agent_pos)]
        )
        td_error = td_target - curr_val
        updated_val = curr_val + self.learning_rate * td_error
        q_values = q_values.at[tuple(state_action_idx)].set(updated_val)
        return q_values


class EpsilonGreedy:
    """Epsilon-greedy policy."""

    def __init__(self, eps: float, n_actions: int):
        """Initialize epsilon-greedy policy parameters."""
        self.eps = eps
        self.n_actions = n_actions

    def __call__(
        self, key: chex.Array, q_values: chex.Array, state: chex.Array
    ) -> chex.Array:
        """Select action using epsilon-greedy policy."""

        def _random_action(key: chex.Array) -> chex.Array:
            """Select a random action."""
            return jax.random.randint(key, shape=(), minval=0, maxval=self.n_actions)

        def _greedy_action(key: chex.PRNGKey) -> chex.Array:
            """Select the greedy action."""
            # Mask for actions with maximum Q-value
            q_max_mask = jnp.equal(
                q_values[tuple(state)], jnp.max(q_values[tuple(state)])
            )
            # Assign equal probability to all actions with maximum Q-value
            proba_max = jnp.divide(q_max_mask, jnp.sum(q_max_mask))
            # Select action with maximum Q-value
            action = jax.random.choice(key, jnp.arange(self.n_actions), p=proba_max)
            return action

        key, eps_key, act_key = jax.random.split(key, 3)
        explore = jax.random.uniform(eps_key) < self.eps
        action = jax.lax.cond(
            explore,
            _random_action,
            _greedy_action,
            act_key,
        )
        return action


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
