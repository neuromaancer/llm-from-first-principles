"""Return utilities shared by reinforcement-learning lessons."""

from collections.abc import Sequence


def discounted_returns(
    rewards: Sequence[float],
    gamma: float,
) -> list[float]:
    """Compute the discounted return from every time step.

    Args:
        rewards: Rewards ordered from earliest to latest.
        gamma: Discount factor in the closed interval [0, 1].

    Returns:
        One discounted return for every reward position.

    Raises:
        ValueError: If gamma lies outside [0, 1].
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be between 0 and 1."
        )

    returns: list[float] = [
        0.0
        for _ in rewards
    ]

    running_return: float = 0.0

    for time_step in reversed(
        range(len(rewards))
    ):
        running_return = (
            float(rewards[time_step])
            + gamma * running_return
        )

        returns[time_step] = running_return

    return returns


__all__ = [
    "discounted_returns",
]
