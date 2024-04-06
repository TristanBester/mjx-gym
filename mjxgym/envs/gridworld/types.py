import chex
import jax.numpy as jnp

from mjxgym.types.state import EnvironmentState


@chex.dataclass(frozen=True)
class State(EnvironmentState):
    """Model of GridWorld environment state."""

    grid: chex.Array
    agent_pos: chex.Array


@chex.dataclass(frozen=True)
class Observation:
    """Model of the agent-observable properties of the environment state."""

    agent_pos: chex.Array
