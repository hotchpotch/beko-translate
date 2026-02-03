from __future__ import annotations


def test_main_is_callable() -> None:
    from neko_translate import main

    assert callable(main)
