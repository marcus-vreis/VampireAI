"""Smoke tests — garantem que os módulos importam sem erro."""

import src


def test_package_imports() -> None:
    assert src is not None
