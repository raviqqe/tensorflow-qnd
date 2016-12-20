import gargparse



_FLAGS = set()

def add_flag(name, *args, **kwargs):
  if name not in _FLAGS:
    global _FLAGS
    _FLAGS.add(name)
    gargparse.add_argument("--" + name, *args, **kwargs)


def add_required_flag(name, *args, **kwargs):
  add_flag(name, *args, required=True, **kwargs)


class FlagAdder:
  def __init__(self):
    self._flags = []

  def add_flag(self, name, *args, **kwargs):
    add_flag(name, *args, **kwargs)
    self._flags.append(name)

  def add_required_flag(self, name, *args, **kwargs):
    self.add_flag(name, *args, required=True, **kwargs)

  @property
  def flags(self):
    return self._flags


# flag types

def str_list(string):
  return string.split(',')


def tf_module_attr(name):
  return getattr(tf, name)
