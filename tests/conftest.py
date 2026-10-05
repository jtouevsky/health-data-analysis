from __future__ import annotations

import pytest

from healthlab.config import load_config


@pytest.fixture
def cfg():
    """Project configuration with demo mode off (tests always pass explicit DB paths)."""
    return load_config(demo_mode=False)
