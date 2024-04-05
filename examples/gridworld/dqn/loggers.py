from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

# consider having an episodic metric logger and a step metric logger


@chex.dataclass(frozen=True)
class LoggerState:
    """State of the logger."""

    export_freq: int
    env_count: int
    ep_counters: chex.Array
    ep_returns: chex.Array
    returns_buffer: chex.Array
    log_counter: int = 0
    total_episodes: int = 0


writer = SummaryWriter("logs")


class Logger:
    def init(self, export_freq: int, env_count: int) -> LoggerState:
        return LoggerState(
            export_freq=export_freq,
            env_count=env_count,
            ep_counters=jnp.zeros(env_count, dtype=jnp.int32),
            ep_returns=jnp.zeros(env_count, dtype=jnp.float32),
            returns_buffer=jnp.zeros((env_count, export_freq), dtype=jnp.float32),
        )

    def _handle_export(self, logger_state: LoggerState) -> None:
        """Export the data to disk."""
        returns = []
        for i, c in enumerate(logger_state.ep_counters):
            returns.append(logger_state.returns_buffer[i, :c])
        returns = jnp.hstack(returns)
        writer.add_scalar("returns", returns.mean(), logger_state.total_episodes)

        # for ep, r in zip(eps, returns):
        #     writer.add_scalar("returns", r, ep)

    def _export(self, logger_state: LoggerState) -> LoggerState:
        logger_state = logger_state.replace(
            total_episodes=logger_state.total_episodes + logger_state.ep_counters.sum()
        )

        jax.debug.callback(self._handle_export, logger_state)

        # jax.debug.print("Exporting")

        # Start logging from the beginning of the returns buffer
        ep_counters = logger_state.ep_counters.at[:].set(0)
        # Reset the returns buffer
        returns_buffer = logger_state.returns_buffer.at[:, :].set(0.0)
        # Note: Do not overwrite ep_returns to preserve rewards from currently active episodes
        # Update the logger state
        updated_logger_state = logger_state.replace(
            ep_counters=ep_counters,
            returns_buffer=returns_buffer,
        )
        return updated_logger_state

    def log(
        self, logger_state: LoggerState, rewards: chex.Array, dones: chex.Array
    ) -> LoggerState:
        # Unpack the logger state
        curr_ep_returns = logger_state.ep_returns
        ep_counters = logger_state.ep_counters
        returns_buffer = logger_state.returns_buffer

        # Compute new logger state
        curr_ep_returns += rewards
        dones = jax.lax.convert_element_type(dones, jnp.float32)
        buffer_modifiers = jnp.multiply(curr_ep_returns, dones)
        buffer_idx = (jnp.arange(4, dtype=jnp.int32), ep_counters)
        updated_values = jnp.add(returns_buffer[buffer_idx], buffer_modifiers)
        returns_buffer = returns_buffer.at[buffer_idx].set(updated_values)
        ep_counters += dones.astype(jnp.int32)
        curr_ep_returns = jnp.multiply(curr_ep_returns, jnp.subtract(1.0, dones))
        updated_logger_state = LoggerState(
            export_freq=logger_state.export_freq,
            env_count=logger_state.env_count,
            ep_counters=ep_counters,
            ep_returns=curr_ep_returns,
            returns_buffer=returns_buffer,
            log_counter=logger_state.log_counter + 1,
            total_episodes=logger_state.total_episodes,
        )

        export_required = jnp.equal(
            jnp.mod(updated_logger_state.log_counter, updated_logger_state.export_freq),
            0,
        )

        logger_state = jax.lax.cond(
            export_required,
            lambda state: self._export(state),
            lambda state: state,
            updated_logger_state,
        )
        return logger_state


# if __name__ == "__main__":
#     logger = Logger()
#     log_state = logger.init(env_count=4, export_freq=10)

#     # rewards = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
#     # dones = jnp.array([False, False, False, False], dtype=jnp.bool_)
#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)

#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)
#     # log_state = logger.log(log_state, rewards, dones)
#     # rewards = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
#     # dones = jnp.array([True, True, True, True], dtype=jnp.bool_)
#     # log_state = logger.log(log_state, rewards, dones)

#     for i in range(100):
#         rewards = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
#         dones = jnp.array([False, False, False, False], dtype=jnp.bool_)
#         log_state = logger.log(log_state, rewards, dones)

#         rewards = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
#         rewards = jax.random.uniform(jax.random.PRNGKey(0), (4,), dtype=jnp.float32)
#         dones = jnp.array([True, True, True, True], dtype=jnp.bool_)
#         log_state = logger.log(log_state, rewards, dones)

#     # ep_counts = jnp.array([3, 2, 1, 0])
#     # arr = jnp.array(
#     #     [
#     #         [1, 1, 1, 0],
#     #         [1, 1, 0, 0],
#     #         [1, 0, 0, 0],
#     #         [0, 0, 0, 0],
#     #     ]
#     # )

#     # print(ep_counts)
#     # print(arr)

#     # returns = []
#     # for i, c in enumerate(ep_counts):
#     #     returns.append(arr[i, :c])
#     # returns = np.hstack(returns)
#     # print(returns)
