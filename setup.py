import setuptools
import sys



if not sys.version_info >= (3, 5):
  exit("Sorry, Python must be later than 3.5.")


setuptools.setup(
  name="tensorflow-qnd",
  version="0.0.2",
  description="Quick and Distributed TensorFlow command framework",
  long_description=open("README.md").read(),
  license="Public Domain",
  author="Yota Toyama",
  author_email="raviqqe@gmail.com",
  url="http://github.com/raviqqe/tensorflow-qnd/",
  packages=["qnd"],
  install_requires=["gargparse"],
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: Public Domain",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: System :: Networking",
  ],
)
