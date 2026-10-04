import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full closed-loop SITL runs (~1 min each)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m", default=""):
        return
    skip_slow = pytest.mark.skip(reason="slow SITL test; run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
