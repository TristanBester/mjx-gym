from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import chex
import jax.numpy as jnp

State = TypeVar("State")


class RewardFunction(ABC, Generic[State]):
    """Interface for reward functions."""

    @abstractmethod
    def __call__(
        self, state: State, action: chex.Array, next_state: State
    ) -> jnp.float32:
        """Map the (s, a, s') tuple to a scalar reward value."""
