import re
from pathlib import Path
from typing import Union

_GIT_URI_REGEX = re.compile(r"^[^/]*:")
_FILE_URI_REGEX = re.compile(r"^file://.+")
_ZIP_URI_REGEX = re.compile(r".+\.zip")


def is_zip_uri(uri: Union[str, Path]) -> bool:
    return uri and _ZIP_URI_REGEX.match(str(uri)) is not None


def is_local_dir(uri: Union[str, Path]) -> bool:
    return uri and not _GIT_URI_REGEX.match(str(uri)) and Path(uri).is_dir()


def is_git_uri(uri: Union[str, Path]) -> bool:
    return uri and _GIT_URI_REGEX.match(str(uri)) is not None


def get_git_commit_id(path: Union[str, Path], notifier) -> str:
    try:
        import git

        repo = git.Repo(path=path, search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        notifier.warning(f"Failed to get git commit with error: {e}")
        return ""
