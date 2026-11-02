"""SM-2 spaced repetition. Pure arithmetic: today is an argument, so one input gives one date."""

from datetime import date, timedelta
from typing import Any

# SM-2's floor. Below roughly 1.3 the interval growth collapses and the item is
# shown so often that the algorithm stops being spaced repetition.
MIN_EASE = 1.3
DEFAULT_EASE = 2.5


def sm2(
    *, quality: int, ease: float, interval: int, repetitions: int, today: date
) -> dict[str, Any]:
    """One SM-2 step. Pure arithmetic over its arguments.

    Args:
        quality: recall quality, 0 (blank) to 5 (perfect).
        ease: current ease factor.
        interval: current interval in days.
        repetitions: consecutive successful reviews so far.
        today: the date the review happened.

    Returns:
        The updated schedule. Same input, same output, always.
    """
    if quality < 3:
        # A lapse resets the ladder. The item is not "slightly harder"; it was
        # not recalled, and the next interval has to start over.
        repetitions = 0
        next_interval = 1
    else:
        repetitions += 1
        if repetitions == 1:
            next_interval = 1
        elif repetitions == 2:
            next_interval = 6
        else:
            next_interval = round(interval * ease)

    # The ease factor updates on every review including lapses -- that is
    # canonical SM-2, and the variant that only updates on success lets a
    # repeatedly-failed item keep an ease it has not earned.
    delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    next_ease = max(MIN_EASE, ease + delta)

    return {
        "interval": next_interval,
        # Rounded so two runs cannot differ in the sixteenth decimal place and
        # fail an equality assertion that is morally true.
        "ease": round(next_ease, 4),
        "repetitions": repetitions,
        "due_date": (today + timedelta(days=next_interval)).isoformat(),
        "lapsed": quality < 3,
    }
