import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from make_fixture import build  # noqa: E402


@pytest.fixture(scope="session")
def mcpas_csv(tmp_path_factory) -> Path:
    """A synthetic McPAS-like export carrying the real file's defects."""
    path = tmp_path_factory.mktemp("data") / "mcpas_like.csv"
    build(path, n_rows=600, seed=0)
    return path


@pytest.fixture(scope="session")
def raw_frame(mcpas_csv):
    from tcrga.data import load_mcpas

    return load_mcpas(mcpas_csv)


@pytest.fixture()
def clean_frame(raw_frame):
    from tcrga.data import normalise

    frame, _ = normalise(raw_frame)
    return frame
