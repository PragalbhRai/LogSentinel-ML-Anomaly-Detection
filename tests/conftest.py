from pathlib import Path

import pytest

from logsentinel import InferenceEngine, load_models, load_replay

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def models():
    return load_models()


@pytest.fixture(scope="session")
def engine(models):
    return InferenceEngine(models)


@pytest.fixture(scope="session")
def replay():
    return load_replay()
