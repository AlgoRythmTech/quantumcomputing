import pytest
import importlib

def test_import_photonai_backend():
    assert importlib.util.find_spec('photonai_backend') is not None
