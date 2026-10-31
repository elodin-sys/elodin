"""Pure exit aggregation for independent Betaflight qualification contracts."""


def scenario_exit_code(
    *,
    hold_exit_code: int | None = None,
    c0_passed: bool | None = None,
    referee_exit_code: int | None = None,
    axis_passed: bool | None = None,
    camera_accepted: bool | None = None,
) -> int:
    """Preserve each scenario's nonzero failure, including simultaneous B + E failures.

    An unselected check is ``None``, not a pass. The ordered result lines are
    printed separately before this function is called.
    """
    if hold_exit_code:
        return hold_exit_code
    if c0_passed is False:
        return 1
    if referee_exit_code:
        return referee_exit_code
    if axis_passed is False or camera_accepted is False:
        return 1
    return 0
