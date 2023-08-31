from __future__ import annotations

from qwak_sdk.commands.models.build._logic.constant.temp_dir import TEMP_LOCAL_MODEL_DIR
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.build._logic.phase.a_fetch_model_code.fetch_model_step.fetch_strategy_manager.fetch_strategy_manager import (
    FetchStrategyManager,
)
from qwak_sdk.commands.models.build._logic.util.step_decorator import (
    build_failure_handler,
)


class FetchModelStep(Step):
    def description(self) -> str:
        return "Fetch model code"

    @build_failure_handler()
    def execute(self) -> None:
        fetch_strategy_manager = FetchStrategyManager(
            uri=self.config.build_properties.model_uri.uri,
            notifier=self.notifier,
        )
        self.notifier.debug("Fetching model code")
        git_commit_id = fetch_strategy_manager.fetch(
            dest=self.context.host_temp_local_build_dir / TEMP_LOCAL_MODEL_DIR,
            git_branch=self.config.build_properties.model_uri.git_branch,
            git_credentials=self.context.git_credentials,
            model_id=self.config.build_properties.model_id,
            build_id=self.context.build_id,
            custom_dependencies_path=self.config.build_env.python_env.dependency_file_path,
            main_dir=self.config.build_properties.model_uri.main_dir,
            dependency_path=self.context.model_relative_dependency_file,
            lock_dependency_path=self.context.model_relative_dependency_lock_file,
            dependency_required_folders=self.config.build_properties.model_uri.dependency_required_folders,
        )
        if git_commit_id:
            self.context.git_commit_id = git_commit_id
            self.notifier.debug(f"Git commit ID identified - {git_commit_id}")

        self.notifier.debug(
            f"Model code stored in {self.context.host_temp_local_build_dir / TEMP_LOCAL_MODEL_DIR}"
        )
        self.notifier.info("Successfully fetched model code")
