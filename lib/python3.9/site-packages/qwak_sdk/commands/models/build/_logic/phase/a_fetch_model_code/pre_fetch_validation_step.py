import uuid
from pathlib import Path

from qwak.clients.secret_service import SecretServiceClient

from qwak_sdk.commands._logic.instance_template import verify_template_id
from qwak_sdk.commands.models.build._logic.constant.host_resource import (
    HOST_TEMP_BUILD_DIR,
)
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.build._logic.phase.a_fetch_model_code.post_fetch_validation_step import (
    find_dependency_files,
)
from qwak_sdk.commands.models.build._logic.util.step_decorator import (
    build_failure_handler,
)
from qwak_sdk.commands.models.build._logic.util.text import snake_case
from qwak_sdk.exceptions import QwakSuggestionException


class PreFetchValidationStep(Step):
    DEFAULT_CPUS = 2
    DEFAULT_MEMORY = "4Gi"
    VALIDATION_FAILURE_CPU_AND_GPU_MESSAGE = (
        "CPU and GPU configured together, Invalid configuration."
    )
    VALIDATION_FAILURE_CPU_AND_GPU_SUGGESTION = "Please configure only CPU or GPU"
    VALIDATION_FAILURE_CPU_AND_INSTANCE_MESSAGE = (
        "CPU and instance configured together, Invalid configuration."
    )
    VALIDATION_FAILURE_CPU_AND_INSTANCE_SUGGESTION = (
        "Please configure only CPU or instance"
    )
    VALIDATION_FAILURE_GPU_AND_INSTANCE_MESSAGE = (
        "GPU and instance configured together, Invalid configuration."
    )
    VALIDATION_FAILURE_GPU_AND_INSTANCE_SUGGESTION = (
        "Please configure only GPU or instance"
    )

    def description(self) -> str:
        return "Pre model fetch validation"

    def execute(self) -> None:
        self.validate_build_properties()
        self.create_build_id()
        self.collect_model_id()
        self.create_build_dir()
        self.collect_git_credentials()
        self.validate_dependencies()
        self.validate_resources()
        self.validate_deployment_resources()

    def validate_build_properties(self):
        if not self.config.build_properties.model_uri.uri:
            error_message = "Model uri wasn't set"
            self.notifier.error(f"{error_message}, failing...")

            raise QwakSuggestionException(
                message=error_message,
                suggestion="Make sure your build properties object contains model uri argument",
            )

    @build_failure_handler()
    def collect_model_id(self):
        model_id = self.config.build_properties.model_id
        model = self.context.client_models_management.get_model(
            model_id=model_id,
            exception_on_missing=False,
        )
        if not model:
            suggestion = f"Create model {model_id} or check model ID spelling"
            snake_case_model_id = snake_case(model_id)
            if self.context.client_models_management.is_model_exists(
                snake_case_model_id
            ):
                suggestion = f"Try using model ID {snake_case_model_id} instead"
            raise QwakSuggestionException(
                message=f"Model ID {model_id} isn't found",
                suggestion=suggestion,
            )
        self.context.project_uuid = model.project_id
        self.context.model_uuid = model.uuid
        self.context.model_id = model_id

    def create_build_id(self):
        self.context.build_id = str(uuid.uuid4())
        self.notifier.info(f"Generated build ID - {self.context.build_id}")

    def create_build_dir(self):
        build_dir = HOST_TEMP_BUILD_DIR / self.context.model_id / self.context.build_id
        build_dir.mkdir(exist_ok=True, parents=True)
        self.context.host_temp_local_build_dir = build_dir
        self.notifier.debug(f"Build directory created - {build_dir}")

    def collect_git_credentials(self):
        if not self.context.git_credentials:
            if self.config.build_properties.model_uri.git_credentials:
                self.context.git_credentials = (
                    self.config.build_properties.model_uri.git_credentials
                )
            elif self.config.build_properties.model_uri.git_credentials_secret:
                self.context.git_credentials = SecretServiceClient().get_secret(
                    self.config.build_properties.model_uri.git_credentials_secret
                )
            else:
                self.notifier.debug("Git credentials isn't configured")

    def validate_dependencies(self):
        if (
            Path(self.config.build_properties.model_uri.uri).is_dir()
            and not self.config.build_env.python_env.dependency_file_path
        ):
            model_uri, main_dir = (
                Path(self.config.build_properties.model_uri.uri),
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

    def validate_resources(self):
        gpu_configured = (
            self.config.build_env.remote.resources.gpu_type
            or self.config.build_env.remote.resources.gpu_amount
        )
        cpu_configured = (
            self.config.build_env.remote.resources.cpus
            or self.config.build_env.remote.resources.memory
        )
        instance_configured = self.config.build_env.remote.resources.instance

        if cpu_configured and gpu_configured:
            raise QwakSuggestionException(
                message=self.VALIDATION_FAILURE_CPU_AND_GPU_MESSAGE,
                suggestion=self.VALIDATION_FAILURE_CPU_AND_GPU_SUGGESTION,
            )
        if cpu_configured and instance_configured:
            raise QwakSuggestionException(
                message=self.VALIDATION_FAILURE_CPU_AND_INSTANCE_MESSAGE,
                suggestion=self.VALIDATION_FAILURE_CPU_AND_INSTANCE_SUGGESTION,
            )
        if gpu_configured and instance_configured:
            raise QwakSuggestionException(
                message=self.VALIDATION_FAILURE_GPU_AND_INSTANCE_MESSAGE,
                suggestion=self.VALIDATION_FAILURE_GPU_AND_INSTANCE_SUGGESTION,
            )

        if instance_configured:
            verify_template_id(
                self.config.build_env.remote.resources.instance,
                self.context.client_instance_template,
            )

        if not (cpu_configured or gpu_configured or instance_configured):
            self.config.build_env.remote.resources.cpus = self.DEFAULT_CPUS
            self.config.build_env.remote.resources.memory = self.DEFAULT_MEMORY

    def validate_deployment_resources(self):
        if self.config.deploy and self.config.deployment_instance:
            verify_template_id(
                self.config.deployment_instance, self.context.client_instance_template
            )
