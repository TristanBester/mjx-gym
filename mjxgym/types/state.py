import chex


@chex.dataclass
class EnvironmentState:
    """Base class for environment states."""

    key: chex.PRNGKey
