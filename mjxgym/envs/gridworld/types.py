import chex
import jax.numpy as jnp

from mjxgym.types.state import EnvironmentState


@chex.dataclass
class State(EnvironmentState):
    """Model of GridWorld environment state."""

    step_count: jnp.int32
    grid: chex.Array
    agent_pos: chex.Array


@chex.dataclass
class Observation:
    """Model of the agent-observable properties of the environment state."""

    agent_pos: chex.Array
