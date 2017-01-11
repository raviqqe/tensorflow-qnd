import sys

from .flag import FlagAdder


def test_flag_adder():
    sys.argv = ["command", "--foo", "baz"]

    adder = FlagAdder()
    adder.add_flag("foo", dest="bar")
    assert adder.flags["bar"] == "baz"
