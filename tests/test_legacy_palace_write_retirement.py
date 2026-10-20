import importlib.util


def test_direct_palace_write_module_is_retired() -> None:
    assert importlib.util.find_spec("lethyx.memory.palace") is None
