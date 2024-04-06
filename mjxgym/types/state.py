import chex


@chex.dataclass(frozen=True)
class EnvironmentState:
    """Base class for environment states."""

    key: chex.PRNGKey
    step_count: int
