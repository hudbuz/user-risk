from typing import Tuple

from qwak.exceptions import QuietError

from qwak_sdk.commands.models.build._logic.context import Context
from qwak_sdk.exceptions import QwakGeneralBuildException
from qwak_sdk.exceptions.qwak_remote_build_failed import QwakRemoteBuildFailedException

from ..build_steps import StepsPipeline, remote_build_steps
from ..config.config_v1 import ConfigV1
from ..interface.notifier_interface import Notifier
from .build_run_handlers import BuildRunHandler, CLIBuildRunner, ProgrammaticBuildRunner
from .logger import build_logger
from .messages import SUCCESS_MSG_REMOTE, SUCCESS_MSG_REMOTE_WITH_DEPLOY
from .notifier_impl import NotifierImpl
from .spinner import spinner
from .time_source import Stopwatch, SystemClockTimeSource, TimeSource
from .utils import zip_logs


def execute_build_pipeline(
    config: ConfigV1,
    json_logs: bool,
    programmatic: bool = False,
    use_git_credentials: str = None,
):
    with build_logger(config=config, json_logs=json_logs) as (
        logger,
        log_path,
    ):
        pipeline, success_msg = create_pipeline(config)
        if use_git_credentials:
            pipeline.context.git_credentials = use_git_credentials
        time_source = SystemClockTimeSource()
        for phase in pipeline.phases:
            phase_details = phase.get_phase_details()
            notifier = NotifierImpl(
                logger,
                prefix="" if json_logs else phase_details.get_description(),
                phase_details=phase_details,
            )

            if programmatic:
                build_runner = ProgrammaticBuildRunner()
                phase_run(phase, pipeline.context, build_runner, notifier, time_source)
            else:
                with spinner(
                    text=phase_details.get_description(),
                    show=(config.verbose == 0 and not json_logs),
                ) as sp:
                    notifier.set_spinner(sp)
                    build_runner = CLIBuildRunner(sp, log_path)
                    phase_run(
                        phase, pipeline.context, build_runner, notifier, time_source
                    )

    print(
        success_msg.format(
            build_id=pipeline.context.build_id,
            model_id=pipeline.context.model_id,
            project_uuid=pipeline.context.project_uuid,
        )
    )
    zip_logs(log_path=log_path, build_id=pipeline.context.build_id)

    return pipeline.context.build_id


def phase_run(
    phase: StepsPipeline,
    context: Context,
    build_runner: BuildRunHandler,
    notifier: Notifier,
    time_source: TimeSource,
):
    build_id = context.build_id
    current_step_phase = None
    stop_watch = Stopwatch(time_source)
    try:
        for step in phase.steps:
            current_step_phase = str(step.build_phase)
            step.set_notifier(notifier)
            build_runner.handle_phase_in_progress(current_step_phase)

            step.execute()

        phase_duration = stop_watch.elapsed_time_in_seconds()
        build_runner.handle_phase_finished_successfully(
            current_step_phase, phase_duration
        )
    except QwakGeneralBuildException as e:
        phase_duration = stop_watch.elapsed_time_in_seconds()
        build_runner.handle_contact_support_error(
            build_id, current_step_phase, e, phase_duration
        )
    except QwakRemoteBuildFailedException as e:
        phase_duration = stop_watch.elapsed_time_in_seconds()
        build_runner.handle_remote_build_error(
            build_id, current_step_phase, e, phase_duration
        )
    except KeyboardInterrupt:
        phase_duration = stop_watch.elapsed_time_in_seconds()
        build_runner.handle_keyboard_interrupt(
            build_id, current_step_phase, phase_duration
        )
    except BaseException as e:
        notifier.exception("Failed", e)
        phase_duration = stop_watch.elapsed_time_in_seconds()
        if not isinstance(e, QuietError):
            build_runner.handle_pipeline_exception(
                build_id, current_step_phase, e, phase_duration
            )
        else:
            build_runner.handle_pipeline_quiet_exception(
                build_id, current_step_phase, e, phase_duration
            )


def create_pipeline(
    config: ConfigV1,
) -> Tuple[StepsPipeline, str]:
    success_message = (
        SUCCESS_MSG_REMOTE_WITH_DEPLOY if config.deploy else SUCCESS_MSG_REMOTE
    )
    pipeline = remote_build_steps(config)

    return pipeline, success_message
