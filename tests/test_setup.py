import pytest
import importlib

def test_import_setup():
    assert importlib.util.find_spec('setup') is not None
