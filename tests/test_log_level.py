"""MEMORA_LOG_LEVEL: opt-in stderr logging so absorb's INFO audit lines
(profile, supersede, downgrade) are not dropped by Python's default
WARNING-level last-resort handler."""

import logging

import pytest

from memora import server


@pytest.fixture(autouse=True)
def _restore_memora_logger():
    pkg = logging.getLogger("memora")
    saved = (pkg.level, pkg.propagate, list(pkg.handlers))
    yield
    pkg.setLevel(saved[0])
    pkg.propagate = saved[1]
    pkg.handlers[:] = saved[2]


def test_unset_changes_nothing():
    pkg = logging.getLogger("memora")
    before = (pkg.level, pkg.propagate, list(pkg.handlers))
    assert server._configure_memora_logging({}) is None
    assert (pkg.level, pkg.propagate, list(pkg.handlers)) == before


def test_info_reaches_stderr_once(capsys):
    assert server._configure_memora_logging({"MEMORA_LOG_LEVEL": "info"}) == logging.INFO
    server._configure_memora_logging({"MEMORA_LOG_LEVEL": "INFO"})  # idempotent
    logging.getLogger("memora.storage").info("absorb supersede: audit line")
    captured = capsys.readouterr()
    assert captured.err.count("absorb supersede: audit line") == 1
    assert captured.out == ""  # never stdout (stdio transport)


def test_unknown_level_is_ignored(capsys):
    assert server._configure_memora_logging({"MEMORA_LOG_LEVEL": "chatty"}) is None
    assert "ignoring unknown MEMORA_LOG_LEVEL" in capsys.readouterr().err
