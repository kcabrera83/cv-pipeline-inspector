import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_features():
    return [120.0, 35.0, 0.3, 4.5, 150.0, 90.0, 60.0, 0.55, 1800.0, 30.0, 110.0, 0.2, 45.0, 0.7]
