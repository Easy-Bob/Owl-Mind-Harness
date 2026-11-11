"""Shared for every test."""

import pytest


@pytest.fixture(autouse=True)
def orchestrated_turns(monkeypatch):
    """Tests are written against the orchestrated path. The M4-D savings that bypass it (the
    direct route and quiz reuse) are on in the service, and their own tests turn them on."""
    monkeypatch.setenv("DIRECT_SINGLE_TUTOR", "false")
    monkeypatch.setenv("QUIZ_REUSE", "false")
