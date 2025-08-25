import pytest
import importlib

def test_import_train_rythm_model():
    assert importlib.util.find_spec('train_rythm_model') is not None
