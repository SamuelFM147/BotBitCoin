from python.scripts.check_versions import get_versions, is_python_supported

def test_python_version_supported():
    assert is_python_supported() is True

def test_core_libs_present():
    v = get_versions()
    assert v["numpy"] is not None
    assert v["pandas"] is not None
    assert v["torch"] is not None
    assert v["stable_baselines3"] is not None
    assert v["gymnasium"] is not None