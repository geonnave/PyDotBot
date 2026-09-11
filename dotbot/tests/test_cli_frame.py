# SPDX-FileCopyrightText: 2026-present Inria
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the frame resolution ladder.

A frame names a site, so the package must not ship one: the neutral default
is what a fresh install gets, and a real name comes from the config.
"""

from dotbot.calibration.lighthouse2 import FRAME_NAME_DEFAULT
from dotbot.cli._frame import resolve_frame
from dotbot.config import load_config_text, select_deployment


def test_no_config_falls_back_to_a_neutral_package_frame():
    assert FRAME_NAME_DEFAULT == "default"
    assert resolve_frame(environ={}) == ("default", "the default")


def test_the_config_names_the_frame():
    config = load_config_text('frame = "inria-aio-c"')
    assert resolve_frame(config=config, environ={}) == (
        "inria-aio-c",
        "the config file",
    )


def test_the_flag_wins_over_the_config():
    config = load_config_text('frame = "inria-aio-c"')
    assert resolve_frame(config=config, flag="taped-square", environ={}) == (
        "taped-square",
        "the command line",
    )


def test_the_environment_wins_over_the_config_and_loses_to_the_flag():
    config = load_config_text('frame = "inria-aio-c"')
    environ = {"DOTBOT_FRAME": "bench"}
    assert resolve_frame(config=config, environ=environ) == (
        "bench",
        "DOTBOT_FRAME",
    )
    assert resolve_frame(config=config, flag="taped", environ=environ)[0] == "taped"


def test_a_deployment_carries_its_own_frame():
    config = load_config_text(
        'frame = "inria-aio-c"\n'
        'default_deployment = "limerick"\n'
        '[deployment.limerick]\nframe = "limerick-hall"\n'
    )
    deployment, _ = select_deployment(config)
    assert resolve_frame(config=config, deployment=deployment, environ={}) == (
        "limerick-hall",
        "the config file",
    )
