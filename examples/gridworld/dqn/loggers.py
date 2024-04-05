from functools import partial

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class LoggerState:
    """State of the logger."""

    export_freq: int
    env_count: int
    ep_counters: chex.Array
    ep_returns: chex.Array
    returns_buffer: chex.Array
    log_counter: int = 0


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
        print("Exporting")
        print(logger_state.returns_buffer)
        print("Done")

    def _export(self, logger_state: LoggerState) -> LoggerState:
        ep_counters = logger_state.ep_counters.at[:].set(0)
        returns_buffer = logger_state.returns_buffer.at[:, :].set(0.0)
        ep_returns = logger_state.ep_returns.at[:].set(0.0)

        updated_logger_state = logger_state.replace(
            ep_counters=ep_counters,
            returns_buffer=returns_buffer,
            ep_returns=ep_returns,
        )

        jax.debug.callback(self._handle_export, logger_state)
        return updated_logger_state

    def log(
        self, logger_state: LoggerState, rewards: chex.Array, dones: chex.Array
    ) -> LoggerState:
        # Unpack the logger state
        curr_ep_returns = logger_state.ep_returns
        ep_counters = logger_state.ep_counters
        returns_buffer = logger_state.returns_buffer
        env_count = logger_state.env_count

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


if __name__ == "__main__":
    logger = Logger()
    log_state = logger.init(env_count=4, export_freq=10)

    for i in range(20):
        rewards = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
        dones = jnp.array([False, False, False, False], dtype=jnp.bool_)
        log_state = logger.log(log_state, rewards, dones)

        rewards = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
        dones = jnp.array([True, True, True, True], dtype=jnp.bool_)
        log_state = logger.log(log_state, rewards, dones)
