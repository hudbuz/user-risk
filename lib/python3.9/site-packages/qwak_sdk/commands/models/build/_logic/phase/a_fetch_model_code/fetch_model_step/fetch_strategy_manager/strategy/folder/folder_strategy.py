import shutil
import re
from pathlib import Path
from typing import List, Optional

from qwak_sdk.exceptions import QwakSuggestionException
from qwak_sdk.tools.files import copytree
from qwak_sdk.commands.models.build._logic.phase.a_fetch_model_code.post_fetch_validation_step import get_possible_dependency_lock_paths

from ...common import get_git_commit_id
from ..strategy import Strategy, get_ignore_pattern


class FolderStrategy(Strategy):
    def fetch(
        self,
        src: str,
        dest: str,
        custom_dependencies_path: Optional[str],
        main_dir: str,
        dependency_path: str,
        lock_dependency_path: str,
        dependency_required_folders: List[str],
        **kwargs,
    ) -> Optional[str]:
        try:
            self.notifier.debug(f"Fetching Model code from local directory -  {src}")

            ignore_patterns, patterns_for_printing = get_ignore_pattern(
                src, main_dir, self.notifier
            )
            self.notifier.debug(
                f"Will ignore the following files: {patterns_for_printing}."
            )

            copytree(
                src=Path(src) / main_dir,
                dst=Path(dest) / main_dir,
                ignore=ignore_patterns,
            )

            self._copy_dependency_required_folders(dependency_required_folders, src, dest, ignore_patterns)

            if (Path(src) / "tests").exists():
                copytree(
                    src=Path(src) / "tests",
                    dst=Path(dest) / "tests",
                    ignore=ignore_patterns,
                )
            if dependency_path:
                shutil.copy(
                    src=Path(src) / dependency_path, dst=Path(dest) / dependency_path
                )
            if lock_dependency_path:
                if (Path(src) / lock_dependency_path).is_file():
                    shutil.copy(
                        src=Path(src) / lock_dependency_path,
                        dst=Path(dest) / lock_dependency_path,
                    )
                else:
                    self.notifier.warning(
                        "No lock dependency file found in model directory."
                    )

            self._copy_custom_dependencies(custom_dependencies_path, dest)
            # Get git commit id if exists
            return get_git_commit_id(src, self.notifier)
        except Exception as e:
            if isinstance(e, QwakSuggestionException):
                raise e  # Propagate suggestions without changes
            main_dir_path = Path(src) / main_dir
            if not main_dir_path.exists():
                message = f"""Seems like you're running the 'qwak models build' command from the wrong location.
Please make sure you run the command at the parent directory of your '{main_dir}' folder"""
            else:
                message = f"Please make sure that {src} has read permissions.",

            raise QwakSuggestionException(
                message="Unable to copy model",
                src_exception=e,
                suggestion=message,
            )

    def _copy_dependency_required_folders(self, dependency_required_folders, src, dest, ignore_patterns):
        for folder in dependency_required_folders:
            destination_folder = folder
            while destination_folder.startswith(".."):
                destination_folder = re.sub(r"^\.\./", "", destination_folder)

            if (Path(dest) / destination_folder).exists():
                raise QwakSuggestionException(
                    message="Unable to copy model",
                    suggestion=f"It's not possible to copy directory `{folder}` into `{destination_folder}` because the target `{destination_folder}` already exist. Mixing files from multiple sources is not allowed."
                )
            if (Path(src) / folder).exists():
                copytree(
                    src=Path(src) / folder,
                    dst=Path(dest) / destination_folder,
                    ignore=ignore_patterns,
                )
            else:
                self.notifier.warning(
                    'Folder "{}" does not exist. Skipping it.'.format(folder)
                )

    def _copy_custom_dependencies(self, custom_dependencies_path, dest):
        if custom_dependencies_path and Path(custom_dependencies_path).is_file():
            shutil.copy(
                src=custom_dependencies_path,
                dst=Path(dest) / Path(custom_dependencies_path).name,
            )
            possible_dependency_lock_paths = get_possible_dependency_lock_paths(Path(custom_dependencies_path))
            for path in possible_dependency_lock_paths:
                if path.is_file():
                    self.notifier.info("Found dependency lock file: {}".format(path))
                    shutil.copy(
                        src=path,
                        dst=Path(dest) / path.name,
                    )
