import chex
from mujoco.mjx._src.types import Data, Model

from mjxgym.types.state import EnvironmentState


@chex.dataclass(frozen=True)
class State(EnvironmentState):
    """Model of Reacher2D environment state."""

    mjx_model: Model
    mjx_data: Data
    goal_position: chex.Array


@chex.dataclass(frozen=True)
class Observation:
    """Model of the agent-observable properties of the environment state.

    Attributes:
        joint_1_angle: The angle of the first joint. This is link_1 in the XML.
        joint_2_angle: The angle of the second joint. This is link_2 in the XML.
        goal_position: The position of the goal in the environment.
    """

    joint_1_angle: chex.Array
    joint_2_angle: chex.Array
    goal_position: chex.Array
