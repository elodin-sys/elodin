import pytest
import elodin


def test_sum_as_string():
    assert elodin.sum_as_string(1, 1) == "2"
