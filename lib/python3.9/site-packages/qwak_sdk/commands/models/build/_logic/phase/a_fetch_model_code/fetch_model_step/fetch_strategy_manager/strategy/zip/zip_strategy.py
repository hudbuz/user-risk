import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

from qwak_sdk.exceptions import QwakSuggestionException
from qwak_sdk.tools.files import copytree

from ...common import get_git_commit_id
from ..strategy import Strategy, get_ignore_pattern


class ZipStrategy(Strategy):
    def fetch(
        self,
        src: Union[str, Path],
        dest: str,
        custom_dependencies_path: Optional[str],
        main_dir: str,
        **kwargs,
    ) -> Optional[str]:
        self.notifier.info(f"Fetching Model code from local zip file -  {src}")
        try:
            with tempfile.TemporaryDirectory() as temp_extraction_target:
                with ZipFile(src) as zip_file:
                    zip_file.extractall(path=temp_extraction_target)

                    git_commit_id = get_git_commit_id(
                        temp_extraction_target, self.notifier
                    )

                    ignore_patterns, patterns_for_printing = get_ignore_pattern(
                        temp_extraction_target,
                        main_dir,
                        self.notifier,
                    )
                    self.notifier.info(
                        f"Will ignore the following files: {patterns_for_printing}."
                    )

                    copytree(
                        src=temp_extraction_target,
                        dst=dest,
                        ignore=ignore_patterns,
                    )

                    # Copy custom dependencies path
                    if (
                        custom_dependencies_path
                        and Path(custom_dependencies_path).is_file()
                    ):
                        shutil.copy(
                            src=custom_dependencies_path,
                            dst=Path(dest) / Path(custom_dependencies_path).name,
                        )

                return git_commit_id
        except Exception as e:
            raise QwakSuggestionException(
                message="Unable to unzip zipped file",
                src_exception=e,
                suggestion=f"Please make sure that {src} has read permissions",
            )
