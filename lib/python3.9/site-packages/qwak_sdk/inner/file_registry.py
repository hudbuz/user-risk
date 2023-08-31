import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from yaspin.core import Yaspin


def list_qwak_python_files(path: Path, sp: Yaspin):
    """
    Helper function which finds python files with qwak imports in a given module

    Args:
        path: path to a directory
        sp: spinner object, used to print
    Returns:
        List of python files which use the Qwak SDK
    """

    file_paths = []
    if Path.is_dir(path):
        for object_path in path.glob("**/*.py"):
            try:
                if not object_path.is_file():
                    continue
                mod_name, file_ext = os.path.splitext(object_path.name)
                if file_ext != ".py" or mod_name == "setup":
                    continue

                with object_path.open("rb") as fp:
                    content = fp.read()
                    contains_qwak_feature = all(
                        [token in content for token in (b"qwak")]
                    )

                if not contains_qwak_feature:
                    continue

                with _add_to_python_path(object_path):
                    # Test that the file can be imported cleanly
                    spec = importlib.util.spec_from_file_location(mod_name, object_path)
                    if not spec:
                        sp.write(f"Could not load file: {object_path}. Skipping.")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    file_paths.append((mod_name, object_path))
            except Exception as e:
                sp.write(
                    f"Got an error trying to handle the file {object_path}, error is: {e}. Skipping."
                )
    return file_paths


def extract_class_objects(python_files, clazz):
    """
    Helper function which given a list of python files extracts objects of the type `clazz`

    Args
        python_files: list of python files
        clazz: class object which its instances should be extracted
    Returns
        list of extracted objects of type clazz from the given python files
    """
    clazz_objects = []
    for mod, file in python_files:
        with _add_to_python_path(Path(file)):
            spec = importlib.util.spec_from_file_location(mod, file)
            if not spec:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for feature in list(module.__dict__.values()):
                if isinstance(feature, clazz):
                    clazz_objects.append((feature, file))

    return clazz_objects


@contextmanager
def _add_to_python_path(path: Path):
    """
    Helper function - context adding path to python path and then remove it

    Args
        path: path of file or folder to add to python path, in case of file adds the containing folder
    Returns
    """
    try:
        folder = str(path.parent) if path.is_file() else str(path)
        sys.path.append(folder)
        yield
    finally:
        sys.path.remove(folder)
