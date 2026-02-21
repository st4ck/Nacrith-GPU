import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_wrapper import ModelWrapper


@pytest.fixture(scope="session")
def model_wrapper():
    return ModelWrapper(verbose=False)
