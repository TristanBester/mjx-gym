from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import chex

State = TypeVar("State")


class Generator(ABC, Generic[State]):
    """Interface for generators which create initial states for environments."""

    @abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Sample initial state distribution using given PRNG key."""
