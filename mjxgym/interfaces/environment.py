from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import chex

from mjxgym.types.state import EnvironmentState
from mjxgym.types.timestep import TimeStep

State = TypeVar("State", bound=EnvironmentState)
Observation = TypeVar("Observation")


class Environment(ABC, Generic[State, Observation]):
    """Interface for a reinforcement learning environment."""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state using the given PRNG key."""

    @abstractmethod
    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        """Execute the given action in the specified environment state."""
