from functools import cached_property
from typing import NamedTuple

import chex
import jax.numpy as jnp


class State(NamedTuple):
    """This is a dataclass as it is required to be modifiable."""

    key: chex.PRNGKey
    step_count: jnp.int32
    grid: chex.Array  # (5, 5)
    agent_pos: chex.Array  # (2,)


class Observation(NamedTuple):
    """This is a NamedTuple as it is immutable."""

    agent_pos: chex.Array  # (2,)


class StepType(jnp.int8):
    """Defines the status of a `TimeStep` within a sequence.

    First: 0
    Mid: 1
    Last: 2
    """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = jnp.array(0, jnp.int8)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = jnp.array(1, jnp.int8)
    # Denotes the last `TimeStep` in a sequence.
    LAST = jnp.array(2, jnp.int8)


class TimeStep(NamedTuple):
    """Contains the information of an environmental interaction.

    Attributes:
        step_type: A 'StepType' enum value.
        reward: Transition reward.
        discount: Transition discount. None if first step.
        observation: An observation object.
        extras: Extra information.
    """

    step_type: StepType
    reward: chex.Array
    discount: chex.Array
    observation: Observation
    extras: dict

    def is_first(self):
        return self.step_type == StepType.FIRST

    def is_mid(self):
        return self.step_type == StepType.MID

    def is_last(self):
        return self.step_type == StepType.LAST

    def __repr__(self):
        return (
            f"TimeStep(step_type={self.step_type}, "
            f"reward={self.reward}, "
            f"discount={self.discount}, "
            f"observation={self.observation}, "
            f"extras={self.extras})"
        )


def restart(observation, extras) -> TimeStep:
    """Return a timestep with steptype as FIRST"""
    return TimeStep(
        step_type=StepType.FIRST,
        reward=jnp.array(0.0),
        discount=jnp.array(1.0),
        observation=observation,
        extras=extras,
    )


def transition(reward, observation, discount, extras) -> TimeStep:
    """Return a timestep with steptype as MID"""
    return TimeStep(
        step_type=StepType.MID,
        reward=reward,
        discount=discount,
        observation=observation,
        extras=extras,
    )


def termination(reward, observation, extras) -> TimeStep:
    """Return a timestep with steptype as LAST"""
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=jnp.array(0.0),
        observation=observation,
        extras=extras,
    )


def truncation(reward, observation, discount, extras) -> TimeStep:
    """Return a timestep with steptype as LAST"""
    discount = discount or jnp.array(1, dtype=jnp.float32)
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=discount,
        observation=observation,
        extras=extras,
    )
