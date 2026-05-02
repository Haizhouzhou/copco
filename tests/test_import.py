from copco_eye_bench import __version__


def test_version_is_defined() -> None:
    assert isinstance(__version__, str)
    assert __version__
