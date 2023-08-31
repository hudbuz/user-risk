from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from qwak_sdk.exceptions import QwakSuggestionException

from .common import is_git_uri, is_local_dir, is_zip_uri
from .strategy.folder.folder_strategy import FolderStrategy
from .strategy.git.git_strategy import GitStrategy
from .strategy.strategy import Strategy
from .strategy.zip.zip_strategy import ZipStrategy

URI_CHECK_STRATEGY_MAPPING = {
    is_local_dir: FolderStrategy,
    is_git_uri: GitStrategy,
    is_zip_uri: ZipStrategy,
}

if TYPE_CHECKING:
    from qwak_sdk.commands.models.build import Notifier


class FetchStrategyManager:
    def __init__(self, uri: str, notifier: Notifier) -> None:
        self._strategy: Optional[Strategy] = None
        self._uri = uri
        for check, clazz in URI_CHECK_STRATEGY_MAPPING.items():
            if check(uri):
                self._strategy = clazz(notifier)

        if self._strategy is None:
            raise QwakSuggestionException(
                message=f"Model URI {uri} is no valid.",
                suggestion=f"Please make sure that model path: {uri} is a zip file/git uri/local folder.",
            )

    def fetch(
        self,
        dest: str,
        git_branch: str,
        git_credentials: str,
        model_id: str,
        build_id: str,
        custom_dependencies_path: str,
        main_dir: str,
        dependency_path: str,
        lock_dependency_path: str,
        dependency_required_folders: List[str],
    ) -> str:
        return self._strategy.fetch(
            src=self._uri,
            dest=dest,
            git_branch=git_branch,
            git_credentials=git_credentials,
            model_id=model_id,
            build_id=build_id,
            custom_dependencies_path=custom_dependencies_path,
            main_dir=main_dir,
            dependency_path=dependency_path,
            lock_dependency_path=lock_dependency_path,
            dependency_required_folders=dependency_required_folders,
        )
