import pytest
import nox_py


def test_sum_as_string():
    assert nox_py.sum_as_string(1, 1) == "2"
