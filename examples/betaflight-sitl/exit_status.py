"""Pure exit aggregation and callback-execution evidence for SITL scenarios."""


class CallbackExecution:
    """Sticky F adapter failure plus proof of a full normal callback sequence.

    The SDK may log/swallow Python post-step errors. The adapter must record
    errors before re-raising, and mark completion only after the final callback
    body (including cleanup) returned successfully. Physical evidence alone
    cannot establish that execution reached its expected normal end.
    """

    def __init__(self, expected_ticks: int):
        if (
            isinstance(expected_ticks, bool)
            or not isinstance(expected_ticks, int)
            or expected_ticks < 1
        ):
            raise ValueError("expected_ticks must be a positive integer")
        self.expected_ticks = expected_ticks
        self._last_started = -1
        self._last_completed = -1
        self._fatal_reason: str | None = None
        self.normal_completion = False

    def begin(self, tick: int) -> None:
        if tick != self._last_started + 1 or self._last_completed != self._last_started:
            self._fail("callback_sequence_gap")
        if not 0 <= tick < self.expected_ticks:
            self._fail("callback_tick_out_of_range")
        self._last_started = tick

    def completed(self, tick: int) -> None:
        if tick != self._last_started:
            self._fail("callback_sequence_mismatch")
        self._last_completed = tick
        if tick == self.expected_ticks - 1 and self._fatal_reason is None:
            self.normal_completion = True

    def record_error(self, error: BaseException) -> None:
        self._fail(f"callback_error_{type(error).__name__}")

    def _fail(self, reason: str) -> None:
        if self._fatal_reason is None:
            self._fatal_reason = reason
        self.normal_completion = False

    @property
    def failure_reason(self) -> str | None:
        return self._fatal_reason or (
            None if self.normal_completion else "callback_sequence_incomplete"
        )

    @property
    def exit_code(self) -> int:
        return int(self.failure_reason is not None)


def scenario_exit_code(
    *,
    course_exit_code: int | None = None,
    execution_exit_code: int | None = None,
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
    if execution_exit_code:
        return execution_exit_code
    if course_exit_code:
        return course_exit_code
    if hold_exit_code:
        return hold_exit_code
    if c0_passed is False:
        return 1
    if referee_exit_code:
        return referee_exit_code
    if axis_passed is False or camera_accepted is False:
        return 1
    return 0
