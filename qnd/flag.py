import gargparse


FLAGS = gargparse.ARGS
_FLAG_NAMES = set()


def add_flag(name, *args, **kwargs):
    """Add a flag.

    Added flags can be accessed by `FLAGS` module variable.
    (e.g. `FLAGS.my_flag_name`)

    - Args
        - `name`: Flag name. Real flag name will be `"--{}".format(name)`.
        - `*args`, `**kwargs`: The rest arguments are the same as
            `argparse.ArgumentParser.add_argument()`.
    """
    global _FLAG_NAMES

    if name not in _FLAG_NAMES:
        _FLAG_NAMES.add(name)
        gargparse.add_argument("--" + name, *args, **kwargs)


def add_required_flag(name, *args, **kwargs):
    """Add a required flag.

    Its interface is the same as `add_flag()` but `required=True` is set by
    default.
    """
    add_flag(name, *args, required=True, **kwargs)


class FlagAdder:
    """Manage addition of flags."""

    def __init__(self):
        """Create a `FlagAdder` instance."""
        self._flags = []

    def add_flag(self, name, *args, **kwargs):
        """Add a flag.

        See `add_flag()`.
        """
        add_flag(name, *args, **kwargs)
        self._flags.append(name)

    def add_required_flag(self, name, *args, **kwargs):
        """Add a required flag.

        See `add_required_flag()`.
        """
        self.add_flag(name, *args, required=True, **kwargs)

    @property
    def flags(self):
        """Get added flags.

        - Returns
            - `dict` of flag names to values added by a `FlagAdder` instance.
        """
        return {flag: getattr(FLAGS, flag) for flag in self._flags}


def add_output_dir_flag():
    add_flag("output_dir",
             default="output",
             help="Directory where checkpoint and event files are stored")
