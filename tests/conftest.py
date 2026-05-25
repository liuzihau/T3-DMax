# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end tests that build the model (need T3-D modeling deps importable)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked `slow` (end-to-end model tests).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
