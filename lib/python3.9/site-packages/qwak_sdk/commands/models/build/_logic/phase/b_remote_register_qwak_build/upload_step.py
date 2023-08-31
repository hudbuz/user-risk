from __future__ import annotations

import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import requests
from qwak.exceptions import QwakException

from qwak_sdk.commands.models.build._logic.constant.temp_dir import TEMP_LOCAL_MODEL_DIR
from qwak_sdk.commands.models.build._logic.constant.upload_tag import (
    BUILD_CONFIG_TAG,
    MODEL_CODE_TAG,
    QWAK_CORE_WHEEL_TAG,
    QWAK_RUNTIME_WHEEL_TAG,
    QWAK_SDK_VERSION_TAG,
    SKINNY_MODEL_CODE_TAG,
)
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.build._logic.util.step_decorator import (
    build_failure_handler,
)
from qwak_sdk.exceptions import QwakGeneralBuildException
from qwak_sdk.tools.files import copytree

if TYPE_CHECKING:
    from qwak_sdk.commands.models.build._logic import Notifier

from qwak_sdk import __version__ as qwak_sdk_version

_MAX_FILE_SIZE_BYTES = 10000000
_IGNORED_PATTERNS = [r"\..*", r"__pycache__"]
_LOG_PREFIX = "Saving Qwak Model"
QWAK_IGNORE_FILE_NAME = ".qwakignore"

# Hidden directories setting
HIDDEN_FILES_PREFIX = "."
HIDDEN_DIRS_TO_INCLUDE = [".dvc"]


class UploadStep(Step):
    def description(self) -> str:
        return _LOG_PREFIX

    @build_failure_handler()
    def execute(self) -> None:
        files_tag_iterator = self.create_files_to_upload()
        files_total_size = sum(
            file.stat().st_size for (file, tag) in files_tag_iterator
        )
        upload_so_far = 0
        for file, tag in files_tag_iterator:
            if file.exists():
                pre_signed_url = self.upload_file(
                    file=file,
                    tag=tag,
                    all_files_size_to_upload=files_total_size,
                    read_so_far=upload_so_far,
                )
                upload_so_far += file.stat().st_size

                if tag == MODEL_CODE_TAG:
                    self.context.model_code_remote_url = str(pre_signed_url).split("?")[
                        0
                    ]

    def create_files_to_upload(self) -> Tuple[List[Tuple[Path, Any]], int]:
        ignored_patterns = (
            load_patterns_from_ignore_file(
                notifier=self.notifier,
                ignore_file_path=self.context.host_temp_local_build_dir
                / TEMP_LOCAL_MODEL_DIR
                / self.config.build_properties.model_uri.main_dir
                / QWAK_IGNORE_FILE_NAME,
            )
            + _IGNORED_PATTERNS
        )

        # copy 'main' and 'tests' directories
        dirs_to_include = [self.config.build_properties.model_uri.main_dir, "tests"]
        deps_folders = []
        for (
            folder
        ) in self.config.build_properties.model_uri.dependency_required_folders:
            destination_folder = folder
            while destination_folder.startswith(".."):
                destination_folder = re.sub(r"^\.\./", "", destination_folder)
            deps_folders.append(destination_folder)
        if deps_folders:
            self.notifier.debug(
                f"Adding dependency folders to model code: {deps_folders}"
            )
            dirs_to_include += deps_folders

        self.notifier.debug("Zipping skinny model code")
        skinny_size_zip_file = zip_model(
            build_dir=self.context.host_temp_local_build_dir,
            dependency_file=self.context.model_relative_dependency_file,
            deps_lock_file=self.context.model_relative_dependency_lock_file,
            dirs_to_include=dirs_to_include,
            zip_name="skinny_size_model_code",
            ignored_patterns=ignored_patterns,
            max_bytes=_MAX_FILE_SIZE_BYTES,
        )

        # Full size model
        self.notifier.debug("Zipping full model code")
        full_size_zip_file = zip_model(
            build_dir=self.context.host_temp_local_build_dir,
            dependency_file=self.context.model_relative_dependency_file,
            deps_lock_file=self.context.model_relative_dependency_lock_file,
            dirs_to_include=dirs_to_include,
            zip_name="full_size_model_code",
            ignored_patterns=ignored_patterns,
        )

        # Dump config file for upload
        config_file_temp = self.context.host_temp_local_build_dir / "build.conf"
        config_file_temp.write_text(self.config.to_yaml())

        # Dump qwak-sdk version for upload
        qwak_sdk_version_temp = self.context.host_temp_local_build_dir / "VERSION"
        qwak_sdk_version_temp.write_text(qwak_sdk_version)

        files_tag_iterator = [
            (full_size_zip_file, MODEL_CODE_TAG),
            (skinny_size_zip_file, SKINNY_MODEL_CODE_TAG),
            (qwak_sdk_version_temp, QWAK_SDK_VERSION_TAG),
            (config_file_temp, BUILD_CONFIG_TAG),
        ]

        if self.context.custom_runtime_wheel:
            files_tag_iterator.append(
                (self.context.custom_runtime_wheel, QWAK_RUNTIME_WHEEL_TAG)
            )

        if self.context.custom_core_wheel:
            files_tag_iterator.append(
                (self.context.custom_core_wheel, QWAK_CORE_WHEEL_TAG)
            )

        return files_tag_iterator

    def upload_file(
        self, file: Path, tag: str, all_files_size_to_upload: int, read_so_far: int
    ):
        self.notifier.debug(f"Upload file {file}")

        pre_signed_url = self.get_pre_signed_upload_url(tag=tag)
        self.upload_file_to_s3(
            upload_url=pre_signed_url,
            file=file,
            all_files_size_to_upload=all_files_size_to_upload,
            read_so_far=read_so_far,
        )

        self.notifier.debug(f"Upload file {file} completed")

        return pre_signed_url

    def get_pre_signed_upload_url(self, tag: str) -> str:
        try:
            self.notifier.debug(f"Getting pre-signed url for S3 upload - tag {tag}")

            pre_signed_url = (
                self.context.client_builds_orchestrator.get_build_versioning_upload_url(
                    build_id=self.context.build_id,
                    model_id=self.context.model_id,
                    tag=tag,
                ).upload_url
            )

            self.notifier.debug("Pre-signed url generated successfully")

            return pre_signed_url
        except QwakException as e:
            raise QwakGeneralBuildException(
                message="Unable to get pre-signed url for uploading model",
                src_exception=e,
            )

    def upload_file_to_s3(
        self,
        upload_url: str,
        file: Path,
        all_files_size_to_upload: int,
        read_so_far: int,
    ):
        try:
            self.notifier.debug(f"Upload file {file} to Qwak storage")

            http_response = requests.put(  # nosec B113
                url=upload_url,
                data=UploadInChunks(
                    file=file,
                    notifier=self.notifier,
                    chunk_size_bytes=10,
                    all_files_size_to_upload=all_files_size_to_upload,
                    read_so_far=read_so_far,
                ),
                headers={"content-type": "text/plain"},
            )
            if http_response.status_code != 200:
                raise QwakException(
                    f"Status: [{http_response.status_code}], "
                    f"reason: [{http_response.reason}]"
                )

            self.notifier.debug(f"File {file} uploaded to Qwak storage successfully")
        except Exception as e:
            raise QwakGeneralBuildException(
                message="Fail uploading model to S3 storage.",
                src_exception=e,
            )


def load_patterns_from_ignore_file(notifier: Notifier, ignore_file_path: Path):
    if Path(ignore_file_path).is_file():
        notifier.info("Found a Qwak ignore file - will ignore listed patterns")

        with open(ignore_file_path, "r") as igonre_file:
            patterns_to_ignore = [
                pattern.strip() for pattern in igonre_file.readlines()
            ]
            notifier.debug(
                f"Patterns from Qwak igonre file detected - {str(patterns_to_ignore)}"
            )
            return patterns_to_ignore

    notifier.debug("no Qwak ignore file was found, skipping")
    return []


def _replace_large_files_with_too_large_file_message(
    filtered_model: Path, max_bytes: Optional[int]
):
    def does_exceed_size(file: Path):
        file_size = file.lstat().st_size
        return file_size > max_bytes, file.lstat().st_size

    if max_bytes is None:
        return

    for root, dirs, files in os.walk(filtered_model):
        for file in files:
            file_path = Path(os.path.join(root, file))
            replace_content, file_size = does_exceed_size(file_path)
            if replace_content:
                Path(file_path).write_text(
                    f"File is too big to display. Size: {file_size} bytes"
                )


def zip_model(
    build_dir: Path,
    dependency_file: Path,
    dirs_to_include: list[str],
    zip_name: str,
    ignored_patterns: Iterable[str],
    max_bytes: Optional[int] = None,
    deps_lock_file: Optional[Path] = None,
):
    try:
        filtered_model = build_dir / zip_name
        ignored_patterns = get_files_to_ignore(
            directory=build_dir / TEMP_LOCAL_MODEL_DIR, patterns=ignored_patterns
        )

        for included_dir in dirs_to_include:
            dir_to_copy = build_dir / TEMP_LOCAL_MODEL_DIR / included_dir
            if dir_to_copy.is_dir():
                copytree(
                    src=build_dir / TEMP_LOCAL_MODEL_DIR / included_dir,
                    dst=filtered_model / included_dir,
                    ignore=shutil.ignore_patterns(*ignored_patterns),
                    dirs_exist_ok=True,
                )

        deps_file = build_dir / TEMP_LOCAL_MODEL_DIR / dependency_file
        shutil.copy(deps_file, filtered_model / dependency_file)

        if deps_lock_file:
            deps_lock_file_full_path = build_dir / TEMP_LOCAL_MODEL_DIR / deps_lock_file
            shutil.copy(deps_lock_file_full_path, filtered_model / deps_lock_file)

        _replace_large_files_with_too_large_file_message(filtered_model, max_bytes)

        zip_path = Path(
            shutil.make_archive(
                base_name=str(filtered_model),
                format="zip",
                root_dir=filtered_model,
            )
        )

        shutil.rmtree(filtered_model)
        return zip_path

    except Exception as e:
        raise QwakGeneralBuildException(
            message="Unable to zip model before upload",
            src_exception=e,
        )


def get_files_to_ignore(directory: Path, patterns: Iterable[str] = ()):
    def ignore_hidden(file: Path, exclusions: List[str]):
        name = os.path.basename(os.path.abspath(file))
        is_hidden = name.startswith(HIDDEN_FILES_PREFIX) and (
            name != QWAK_IGNORE_FILE_NAME and name not in exclusions
        )
        return is_hidden

    def is_ignore_by_pattern(file: Path):
        return (
            len(
                [
                    pattern
                    for pattern in patterns
                    if re.search(fnmatch.translate(pattern), str(file))
                ]
            )
            != 0
        )

    return [
        file.name
        for file in Path(directory).rglob("*")
        if is_ignore_by_pattern(file)
        or ignore_hidden(file, exclusions=HIDDEN_DIRS_TO_INCLUDE)
    ]


class UploadInChunks(object):
    def __init__(
        self,
        file: Path,
        notifier: Notifier,
        all_files_size_to_upload: int,
        read_so_far: int = 0,
        chunk_size_bytes: int = 1 << 13,
    ):
        self._file = file
        self._chunk_size = chunk_size_bytes
        self._total_size = self._file.stat().st_size
        self._read_so_far = read_so_far
        self._notifier = notifier
        self._all_files_size_to_upload = (
            all_files_size_to_upload  # Used for calculating percentage for both files
        )
        self._last_percent_update = 0

    def __iter__(self):
        with self._file.open("rb") as file:
            while True:
                data = file.read(self._chunk_size)
                if not data:
                    break
                self._read_so_far += len(data)
                percent = self._read_so_far * 1e2 / self._all_files_size_to_upload
                msg = "{percent:3.0f}%".format(percent=percent)
                if int(percent / 10) > self._last_percent_update:
                    self._notifier.info(msg)  # Updating only after 10 percent change
                    self._last_percent_update = int(percent / 10)
                self._notifier.spinner_text(line=msg)
                yield data

    def __len__(self):
        return self._total_size
