import importlib.util
import os
import shutil
import subprocess  # nosec
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from qwak_sdk.commands.models.build._logic.constant.temp_dir import TEMP_LOCAL_MODEL_DIR
from qwak_sdk.commands.models.build._logic.context import DependencyManagerType
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.build._logic.util.step_decorator import (
    build_failure_handler,
)
from qwak_sdk.exceptions import QwakSuggestionException

TEMP_LOCAL_EDITABLE_FOLDER = "editable"


@dataclass
class DependencyFileObject:
    dep_file_name: List[str]
    lock_file_name: str = field(default="")


DEPS_MANAGER_FILE_MAP = {
    DependencyManagerType.PIP: DependencyFileObject(dep_file_name=["requirements.txt"]),
    DependencyManagerType.POETRY: DependencyFileObject(
        dep_file_name=["pyproject.toml"], lock_file_name="poetry.lock"
    ),
    DependencyManagerType.CONDA: DependencyFileObject(
        dep_file_name=["conda.yml", "conda.yaml"]
    ),
}


class PostFetchValidationStep(Step):
    def description(self) -> str:
        return "Post model fetch validation"

    @build_failure_handler()
    def execute(self) -> None:
        self.validate_dependencies()
        self.configure_base_docker_image()
        self.create_development_wheels()

    def validate_dependencies(self):
        if (
            not Path(self.config.build_properties.model_uri.uri).is_dir()
            or self.config.build_env.python_env.dependency_file_path
        ):
            self.notifier.debug("Validating dependency file exists")
            model_uri, main_dir = (
                self.context.host_temp_local_build_dir / TEMP_LOCAL_MODEL_DIR,
                self.config.build_properties.model_uri.main_dir,
            )
            (
                self.context.dependency_manager_type,
                self.context.model_relative_dependency_file,
                self.context.model_relative_dependency_lock_file,
            ) = find_dependency_files(model_uri, main_dir, self.notifier)

            if (
                self.context.dependency_manager_type
                and self.context.model_relative_dependency_file
            ):
                return

            self.notifier.error("Dependency file wasn't found, failing...")
            raise QwakSuggestionException(
                message="Dependency file isn't found",
                suggestion="Make sure your model include one of dependencies manager: pip/poetry/conda",
            )

    def configure_base_docker_image(self):
        base_image = self.config.build_env.docker.base_image
        if (not base_image) and self.config.build_env.remote.resources.gpu_type:
            base_image = "qwakai/qwak:gpu-py39"
        self.context.base_image = base_image

    def create_development_wheels(self):
        if not os.getenv("QWAK_DEBUG"):
            return

        (module_location,) = importlib.util.find_spec(
            "qwak_sdk"
        ).submodule_search_locations
        source_dir = Path(module_location).parent
        pyproject_toml = source_dir / "pyproject.toml"

        if pyproject_toml.is_file():
            self.notifier.info(
                "Detected non-PyPI-released qwak-sdk installed, creating local qwak-runtime "
                "and qwak-core wheel files to pass to build process"
            )
            runtime_dir = source_dir.parent / "qwak-runtime"
            core_dir = source_dir.parent / "qwak-core"
            target_path = self.context.host_temp_local_build_dir
            self.context.custom_runtime_wheel = self._create_wheel(
                runtime_dir, target_path
            )
            self.context.custom_core_wheel = self._create_wheel(core_dir, target_path)

    def _create_wheel(self, package_dir, target_path):
        dist_dir = package_dir / "dist"
        shutil.rmtree(dist_dir, ignore_errors=True)
        output = subprocess.check_output(["make", "install"], cwd=package_dir)  # nosec
        for line in output.decode().split("\n"):
            self.notifier.debug(f">>> {line}")
        output = subprocess.check_output(["poetry", "build"], cwd=package_dir)  # nosec
        for line in output.decode().split("\n"):
            self.notifier.debug(f">>> {line}")
        wheel_file = next(dist_dir.glob("*.whl"), None)
        editable_folder = target_path / TEMP_LOCAL_EDITABLE_FOLDER
        editable_folder.mkdir(exist_ok=True)
        copied_wheel_file = shutil.move(wheel_file, editable_folder / wheel_file.name)
        shutil.rmtree(dist_dir, ignore_errors=True)
        self.notifier.info(f"Created wheel for {package_dir} successfully")
        return copied_wheel_file


def find_file_location(model_uri, main_dir, filename) -> Path:
    file_locations: List[Path] = [
        model_uri / filename,
        model_uri / main_dir / filename,
    ]
    for file in file_locations:
        if file.is_file():
            return file


def find_dependency_files(model_uri, main_dir, notifier):
    dependency_manager_type = None
    model_relative_dependency_file = None
    model_relative_dependency_lock_file = None
    for dep_type, dependency_file_object in DEPS_MANAGER_FILE_MAP.items():
        for filename in dependency_file_object.dep_file_name:
            dep_file_path = find_file_location(model_uri, main_dir, filename)
            if dep_file_path:
                notifier.info(
                    f"Found dependency type: {dep_type.name} by file: {dep_file_path}"
                )
                dependency_manager_type = dep_type
                model_relative_dependency_file = dep_file_path.relative_to(model_uri)
                if dependency_file_object.lock_file_name:
                    dep_lock_file_path = find_file_location(
                        model_uri,
                        main_dir,
                        dependency_file_object.lock_file_name,
                    )
                    if dep_lock_file_path:
                        notifier.info(
                            f"Found dependency lock file {dep_lock_file_path}"
                        )
                        model_relative_dependency_lock_file = (
                            dep_lock_file_path.relative_to(model_uri)
                        )
                break

    return (
        dependency_manager_type,
        model_relative_dependency_file,
        model_relative_dependency_lock_file,
    )


def get_possible_dependency_lock_paths(dependency_path: Path):
    paths = []
    for _, dependency_file_object in DEPS_MANAGER_FILE_MAP.items():
        if dependency_file_object.lock_file_name:
            lock_file_path = (
                dependency_path.parent / dependency_file_object.lock_file_name
            )
            paths.append(lock_file_path)
    return paths
