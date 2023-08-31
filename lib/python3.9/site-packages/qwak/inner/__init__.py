"""Top-level package for qwak-proto."""

# fmt: off
__author__ = '''Qwak'''
__email__ = 'info@qwak.com'
__version__ = '0.0.315.dev0'
# fmt: on

from .di_configuration import wire_dependencies

_container = wire_dependencies()
