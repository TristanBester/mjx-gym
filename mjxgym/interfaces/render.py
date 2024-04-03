from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

State = TypeVar("State")


class Renderer(ABC, Generic[State]):
    """Interface for environment renderers."""

    @abstractmethod
    def __call__(self, state: State) -> None:
        """Render the given environment state."""

    @abstractmethod
    def render_pixels(self, state: State) -> np.array:
        """Render the given environment state as a pixel array."""

    @abstractmethod
    def animate_trajectory(self, states: list[State], path: str) -> None:
        """Animate the trajectory of the agent in the environment."""
