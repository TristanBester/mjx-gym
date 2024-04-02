import chex
import jax

from mjxgym.type import State, StepType


class Wrapper:
    def __init__(self, env):
        self._env = env

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._env)})"

    def __getattr__(self, name: str):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        """Returns the wrapped env."""
        return self._env.unwrapped

    def reset(self, key: chex.PRNGKey):
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """
        return self._env.reset(key)

    def step(self, state: State, action: chex.Array):
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        return self._env.step(state, action)

    def render(self, state: State):
        """Compute render frames during initialisation of the environment.

        Args:
            state: State object containing the dynamics of the environment.
        """
        return self._env.render(state)


class AutoResetWrapper(Wrapper):
    def __init__(self, env, next_obs_in_extras=True):
        super().__init__(env)
        self.next_obs_in_extras = next_obs_in_extras

        if self.next_obs_in_extras:

            def add_obs_to_extras(timestep):
                extras = timestep.extras
                extras["next_obs"] = timestep.observation
                return timestep._replace(extras=extras)

            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep

    def reset(self, key: chex.PRNGKey):
        state, timestep = super().reset(key)
        timestep = self._maybe_add_obs_to_extras(timestep)
        return state, timestep

    def step(self, state, action):
        state, timestep = self._env.step(state, action)
        state, timestep = jax.lax.cond(
            timestep.is_last(),
            self._auto_reset,
            lambda s, t: (s, self._maybe_add_obs_to_extras(t)),
            state,
            timestep,
        )
        return state, timestep

    def _auto_reset(self, state, timestep):
        _, subkey = jax.random.split(state.key)
        state, reset_timestep = self._env.reset(subkey)

        timestep = self._maybe_add_obs_to_extras(timestep)
        timestep = timestep._replace(observation=reset_timestep.observation)
        return state, timestep
