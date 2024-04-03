from typing import Generic, TypeVar

import chex
import jax.numpy as jnp

Observation = TypeVar("Observation")


class StepType(jnp.int8):
    """StepType is an enum of the different types of time steps."""

    FIRST = jnp.array(0, jnp.int8)
    MID = jnp.array(1, jnp.int8)
    LAST = jnp.array(2, jnp.int8)


@chex.dataclass
class TimeStep(Generic[Observation]):
    """Contains the information associated with an environmental interaction."""

    step_type: StepType
    observation: Observation
    reward: chex.Array
    discount: chex.Array
    info: dict

    def is_first(self) -> bool:
        """Returns True if the step is the first step in an episode."""
        return self.step_type == StepType.FIRST

    def is_mid(self) -> bool:
        """Returns True if the step is a mid-episode step."""
        return self.step_type == StepType.MID

    def is_last(self) -> bool:
        """Returns True if the step is the last step in an episode."""
        return self.step_type == StepType.LAST
