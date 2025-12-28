"""Pytest configuration and fixtures for Fairy2 tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Get test device (GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_complex_weights():
    """Generate random complex weights for testing."""
    def _generate(shape=(10, 10)):
        w_re = torch.randn(shape)
        w_im = torch.randn(shape)
        return w_re, w_im
    return _generate


@pytest.fixture
def simple_linear():
    """Create a simple nn.Linear for testing."""
    import torch.nn as nn
    linear = nn.Linear(64, 128, bias=False)
    return linear
