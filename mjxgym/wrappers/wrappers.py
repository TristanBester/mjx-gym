from typing import TypeVar

import chex
import jax

from mjxgym.interfaces.environment import Environment
from mjxgym.types.timestep import TimeStep

State = TypeVar("State")
Observation = TypeVar("Observation")


class Wrapper(Environment[State, Observation]):
    """A base class for environment wrappers."""

    def __init__(self, env: Environment[State, Observation]):
        self._env = env


class AutoResetWrapper(Wrapper[State, Observation]):
    """An environment wrapper that automatically resets the environment when it is done."""

    def reset(self, key: chex.Array) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment."""
        return self._env.reset(key)

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        """Exexute the given action in the wrapped environment."""
        state, timestep = self._env.step(state, action)
        # Reset the environment if necessary
        state, timestep = jax.lax.cond(
            timestep.is_last(),
            self._auto_reset,
            lambda s, t: (s, t),
            state,
            timestep,
        )
        return state, timestep

    def _auto_reset(self, state, timestep):
        """Reset the environment & change the timestep observation."""
        _, subkey = jax.random.split(state.key)
        state, reset_timestep = self._env.reset(subkey)
        timestep = timestep.replace(observation=reset_timestep.observation)
        return state, timestep


class VmapAutoResetWrapper(AutoResetWrapper[State, Observation]):
    """An environment wrapper that applies vmap to the reset method."""

    def __init__(self, env: Environment[State, Observation]):
        super().__init__(env)
        self.jit_reset = jax.jit(jax.vmap(self._env.reset))
        self.jit_step = jax.jit(jax.vmap(self._env.step))

    @property
    def unwrapped(self) -> Environment[State, Observation]:
        """Return the unwrapped environment."""
        return self._env

    def reset(self, key: chex.Array) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment."""
        return self.jit_reset(key)

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        """Exexute the given action in the wrapped environment."""
        # Vmap homogeneous computation (parallelizable).
        return self.jit_step(state, action)

    # def step(
    #     self, state: State, action: chex.Array
    # ) -> tuple[State, TimeStep[Observation]]:
    #     """Exexute the given action in the wrapped environment."""
    #     # Vmap homogeneous computation (parallelizable).
    #     state, timestep = self.jit_step(state, action)
    #     # Map heterogeneous computation (non-parallelizable).
    #     state, timestep = jax.lax.map(
    #         lambda args: self._reset_if_required(*args),
    #         (state, timestep),
    #     )
    #     return state, timestep

    # def _reset_if_required(self, state, timestep):
    #     """Reset the environment if the timestep is the last."""
    #     state, timestep = jax.lax.cond(
    #         timestep.is_last(),
    #         self._auto_reset,
    #         lambda s, t: (s, t),
    #         state,
    #         timestep,
    #     )
    #     return state, timestep
