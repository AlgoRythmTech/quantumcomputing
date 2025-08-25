import pytest
import importlib

def test_import_advanced_training_system():
    assert importlib.util.find_spec('advanced_training_system') is not None
