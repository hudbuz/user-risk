import logging
from abc import ABC, abstractmethod
from pathlib import Path

from qwak.exceptions import QwakException
from yaspin.core import Yaspin

from qwak_sdk.tools.colors import Color

from .messages import (
    FAILED_CONTACT_QWAK_SUPPORT,
    FAILED_CONTACT_QWAK_SUPPORT_PROGRAMMATIC,
)
from .utils import zip_logs


class BuildRunHandler(ABC):
    @abstractmethod
    def handle_phase_in_progress(self, phase_id: str):
        pass

    @abstractmethod
    def handle_phase_finished_successfully(
        self, phase_id: str, duration_in_seconds: int
    ):
        pass

    @abstractmethod
    def handle_contact_support_error(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        pass

    @abstractmethod
    def handle_remote_build_error(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        pass

    @abstractmethod
    def handle_keyboard_interrupt(
        self, build_id: str, phase_id: str, duration_in_seconds: int
    ):
        pass

    @abstractmethod
    def handle_pipeline_exception(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        pass

    @abstractmethod
    def handle_pipeline_quiet_exception(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        pass

    def build_phase_to_human_readable_string(self, phase_id: str):
        return phase_id[len("BuildPhase.") :].replace("_", " ").capitalize()


class CLIBuildRunner(BuildRunHandler):
    def __init__(self, sp: Yaspin, log_path: Path):
        self.sp = sp
        self.log_path = str(log_path)

    def handle_phase_in_progress(self, phase_id: str):
        logging.debug(
            f"Build phase in progress: {self.build_phase_to_human_readable_string(phase_id)}"
        )

    def handle_phase_finished_successfully(
        self, phase_id: str, duration_in_seconds: int
    ):
        if self.sp:
            self.sp.ok("✅")

        logging.debug(
            f"Phase successfully finished: {self.build_phase_to_human_readable_string(phase_id)} after {duration_in_seconds} seconds"
        )

    def __report_failure(self, phase_id: str, duration_in_seconds: int):
        logging.debug(
            f"Build phase failed: {self.build_phase_to_human_readable_string(phase_id)} after {duration_in_seconds} seconds"
        )

    def handle_contact_support_error(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        print(f"\n{ex}")
        print(
            FAILED_CONTACT_QWAK_SUPPORT.format(
                build_id=build_id,
                log_file=Path(self.log_path).parent / build_id,
            )
        )
        zip_logs(log_path=self.log_path, build_id=build_id)
        self.__report_failure(phase_id, duration_in_seconds)
        exit(1)

    def handle_remote_build_error(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        if self.sp:
            self.sp.fail("💥")
            print(f"\n{Color.RED}{ex}")
        else:
            print(f"\n{ex}")
        self.__report_failure(phase_id, duration_in_seconds)
        exit(1)

    def handle_keyboard_interrupt(
        self, build_id: str, phase_id: str, duration_in_seconds: int
    ):
        print(f"\n{Color.RED}Stopping Qwak build (ctrl-c)")
        zip_logs(log_path=self.log_path, build_id=build_id)
        self.__report_failure(phase_id, duration_in_seconds)
        exit(1)

    def handle_pipeline_exception(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        if self.sp:
            self.sp.fail("💥")
            print(f"\n{Color.RED}{ex}")
        zip_logs(
            log_path=self.log_path,
            build_id=build_id,
        )
        self.__report_failure(phase_id, duration_in_seconds)
        exit(1)

    def handle_pipeline_quiet_exception(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        if self.sp:
            self.sp.ok("‼️")
            print(f"\n{Color.RED}{ex}")
        self.__report_failure(phase_id, duration_in_seconds)
        exit(1)


class ProgrammaticBuildRunner(BuildRunHandler):
    def handle_phase_in_progress(self, phase_id: str):
        logging.debug(
            f"Build phase in progress: {self.build_phase_to_human_readable_string(phase_id)}"
        )

    def handle_phase_finished_successfully(
        self, phase_id: str, duration_in_seconds: int
    ):
        logging.debug(
            f"Phase successfully finished: {self.build_phase_to_human_readable_string(phase_id)} after {duration_in_seconds} seconds"
        )

    def __report_failure(self, phase_id: str, duration_in_seconds: int):
        logging.debug(
            f"Build phase failed: {self.build_phase_to_human_readable_string(phase_id)} after {duration_in_seconds} seconds"
        )

    def handle_contact_support_error(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        print(
            FAILED_CONTACT_QWAK_SUPPORT_PROGRAMMATIC.format(
                build_id=build_id,
            )
        )
        self.__report_failure(phase_id, duration_in_seconds)
        raise QwakException(str(ex))

    def handle_keyboard_interrupt(
        self, build_id: str, phase_id: str, duration_in_seconds: int
    ):
        self.__report_failure(phase_id, duration_in_seconds)

    def handle_pipeline_exception(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        self.__report_failure(phase_id, duration_in_seconds)
        raise QwakException(str(ex))

    def handle_pipeline_quiet_exception(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        self.__report_failure(phase_id, duration_in_seconds)

    def handle_remote_build_error(
        self, build_id: str, phase_id: str, ex: BaseException, duration_in_seconds: int
    ):
        self.handle_pipeline_exception(build_id, phase_id, ex, duration_in_seconds)
