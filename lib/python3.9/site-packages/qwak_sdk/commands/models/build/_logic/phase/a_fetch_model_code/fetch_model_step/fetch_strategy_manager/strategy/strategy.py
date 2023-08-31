from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

_IGNORED_PATTERNS = [r"\..*", r"__pycache__"]
QWAK_IGNORE_FILE_NAME = ".qwakignore"


if TYPE_CHECKING:
    from qwak_sdk.commands.models.build import Notifier


def load_patterns_from_ignore_file(notifier: Notifier, ignore_file_path: Path):
    ignore_file_path = Path(ignore_file_path)
    if ignore_file_path.is_file():
        notifier.info(
            f"Found a Qwak ignore file {ignore_file_path} - will ignore listed patterns"
        )

        with ignore_file_path.open("r") as ignore_file:
            patterns_to_ignore = [
                pattern.strip() for pattern in ignore_file.readlines()
            ]
            notifier.debug(
                f"Patterns from Qwak ignore file detected - {str(patterns_to_ignore)}"
            )
            return patterns_to_ignore

    notifier.debug("no Qwak ignore file was found, skipping")
    return []


def get_ignore_pattern(
    src: str, main_dir: str, notifier: Notifier
) -> Tuple[Callable[[Any, list[str]], set[str]], list[str]]:
    if (Path(src) / main_dir / QWAK_IGNORE_FILE_NAME).is_file():
        ignore_file_path = Path(src) / main_dir / QWAK_IGNORE_FILE_NAME
    else:
        ignore_file_path = Path(src) / QWAK_IGNORE_FILE_NAME

    ignored_patterns = (
        load_patterns_from_ignore_file(
            notifier=notifier, ignore_file_path=ignore_file_path
        )
        + _IGNORED_PATTERNS
    )

    return shutil.ignore_patterns(*ignored_patterns), ignored_patterns


class Strategy(ABC):
    def __init__(self, notifier: Notifier):
        self.notifier = notifier

    @abstractmethod
    def fetch(
        self,
        src: Union[str, Path],
        dest: str,
        git_branch: str,
        git_credentials: str,
        model_id: str,
        build_id: str,
        custom_dependency_path: Optional[str],
    ) -> Optional[str]:
        pass
