from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from mjxgym.types.timestep import StepType, TimeStep

Observation = TypeVar("Observation")


def create_initial_timestep(
    observation: Observation, info: dict | None = None
) -> TimeStep[Observation]:
    """Factory function for creating a TimeStep object with FIRST step type."""
    info = info or {"truncated": False}
    return TimeStep(
        step_type=StepType.FIRST,
        observation=observation,
        reward=jnp.array(0.0, dtype=jnp.float32),
        discount=jnp.array(1.0, dtype=jnp.float32),
        info=info,
    )


def create_timestep(
    done: bool,
    truncated: bool,
    observation: Observation,
    reward: chex.Array,
    discount: chex.Array,
    info: dict | None = None,
) -> TimeStep[Observation]:
    """Factory function for creating a TimeStep object based on the given parameters."""
    dn = jax.lax.convert_element_type(done, jnp.int8)
    tr = jax.lax.convert_element_type(truncated, jnp.int8)
    branch_idx = dn + tr

    # If not done, then not truncated, branch 0
    # If done and not trancted, branch 1
    # If done and truncated, branch 2
    info = info or {"truncated": False}
    return jax.lax.switch(
        branch_idx,
        (
            _create_transition_timestep,
            _create_terminated_timestep,
            _create_truncated_timestep,
        ),
        observation,
        reward,
        discount,
        info,
    )


def _create_transition_timestep(
    observation: Observation,
    reward: chex.Array,
    discount: chex.Array,
    info: dict,
) -> TimeStep[Observation]:
    """Factory function for creating a TimeStep object with MID step type."""
    info["truncated"] = False
    return TimeStep(
        step_type=StepType.MID,
        observation=observation,
        reward=reward,
        discount=discount,
        info=info,
    )


def _create_terminated_timestep(
    observation: Observation,
    reward: chex.Array,
    discount: chex.Array,
    info: dict,
) -> TimeStep[Observation]:
    """Factory function for creating a TimeStep object for transitions into terminal states.

    Discount factor is equal to zero to support simplified q-value updates
    which do not require conditional statements.
    """
    info["truncated"] = False
    return TimeStep(
        step_type=StepType.LAST,
        observation=observation,
        reward=reward,
        discount=jnp.array(0.0, dtype=jnp.float32),
        info=info,
    )


def _create_truncated_timestep(
    observation: Observation,
    reward: chex.Array,
    discount: chex.Array,
    info: dict,
) -> TimeStep[Observation]:
    """Factory function for creating a TimeStep object for truncated trajectories.

    Discount factory is not set to zero to support q-value bootstrapping.
    """
    info["truncated"] = True
    return TimeStep(
        step_type=StepType.LAST,
        observation=observation,
        reward=reward,
        discount=discount,
        info=info,
    )
