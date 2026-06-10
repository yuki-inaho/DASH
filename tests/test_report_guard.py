"""Static-text check for the copy-paste guard bug in train.py training_report.

training_report requires full GPU training to execute, so per workdoc 手順15
this test reads train.py as TEXT (it does NOT import it) and asserts the
structural properties of the fixed guards.
"""
import os

TRAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")


def _read_src():
    with open(TRAIN_PY, "r", encoding="utf-8") as f:
        return f.read()


def test_static_guard_used_once():
    src = _read_src()
    assert src.count("if static_indices.any():") == 1


def test_dynamic_guard_present():
    src = _read_src()
    assert "if dynamic_indices.any():" in src


def test_dynamic_image_none_else_present():
    src = _read_src()
    assert "dynamic_image = None" in src


def test_dynamic_image_logging_guarded():
    src = _read_src()
    assert "if dynamic_image is not None:" in src
