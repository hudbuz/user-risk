import itertools
import uuid
from collections import defaultdict
from datetime import datetime

from _qwak_proto.qwak.automation.v1.automation_management_service_pb2 import (
    CreateAutomationRequest,
    CreateAutomationResponse,
    DeleteAutomationRequest,
    DeleteAutomationResponse,
    GetAutomationByNameRequest,
    GetAutomationByNameResponse,
    GetAutomationRequest,
    GetAutomationResponse,
    ListAutomationExecutionsRequest,
    ListAutomationExecutionsResponse,
    ListAutomationsRequest,
    ListAutomationsResponse,
    RegisterAutomationExecutionRequest,
    RegisterAutomationExecutionResponse,
    RunAutomationRequest,
    RunAutomationResponse,
    ToggleAutomationActivityRequest,
    ToggleAutomationActivityResponse,
    UpdateAutomationExecutionRequest,
    UpdateAutomationExecutionResponse,
    UpdateAutomationRequest,
    UpdateAutomationResponse,
)
from _qwak_proto.qwak.automation.v1.automation_management_service_pb2_grpc import (
    AutomationManagementServiceServicer,
)
from _qwak_proto.qwak.automation.v1.automation_pb2 import Automation
from qwak.automations.automation_executions import (
    AutomationExecution,
    ExecutionRunDetails,
    ExecutionStatus,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class AutomationManagementServiceMock(AutomationManagementServiceServicer):
    QWAK_ENVIRONMENT = "test_environment"

    def __init__(self):
        self.automations = dict()
        self.automation_executions = defaultdict(list)

    def CreateAutomation(
        self, request: CreateAutomationRequest, context
    ) -> CreateAutomationResponse:
        """Create an automation"""
        try:
            automation_id = str(uuid.uuid4())
            automation = Automation(
                automation_id=automation_id,
                automation_spec=request.automation_spec,
                qwak_environment_id=self.QWAK_ENVIRONMENT,
            )
            self.automations[automation_id] = automation
            return CreateAutomationResponse(automation_id=automation_id)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def UpdateAutomation(
        self, request: UpdateAutomationRequest, context
    ) -> UpdateAutomationResponse:
        """Update an automation"""
        try:
            automation = Automation(
                automation_id=request.automation_id,
                automation_spec=request.automation_spec,
                qwak_environment_id=self.QWAK_ENVIRONMENT,
            )
            self.automations[request.automation_id] = automation
            return UpdateAutomationResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeleteAutomation(
        self, request: DeleteAutomationRequest, context
    ) -> DeleteAutomationResponse:
        """Delete an automation"""
        try:
            del self.automations[request.automation_id]
            return DeleteAutomationResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetAutomation(
        self, request: GetAutomationRequest, context
    ) -> GetAutomationResponse:
        """Get an automation By ID"""
        try:
            return GetAutomationResponse(
                automation=self.automations[request.automation_id]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetAutomationByName(
        self, request: GetAutomationByNameRequest, context
    ) -> GetAutomationByNameResponse:
        """Get an automation by name"""
        try:
            return GetAutomationByNameResponse(
                automation=list(
                    filter(
                        lambda a: a.automation_spec.automation_name
                        == request.automation_name,
                        self.automations.values(),
                    )
                )[0]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ListAutomations(
        self, request: ListAutomationsRequest, context
    ) -> ListAutomationsResponse:
        """List all automation"""
        try:
            return ListAutomationsResponse(automations=list(self.automations.values()))
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ListAutomationExecutions(
        self, request: ListAutomationExecutionsRequest, context
    ) -> ListAutomationExecutionsResponse:
        """List automation's executions"""
        try:
            return ListAutomationExecutionsResponse(
                automation_executions=self.automation_executions[request.automation_id]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def RegisterAutomationExecution(
        self, request: RegisterAutomationExecutionRequest, context
    ) -> RegisterAutomationExecutionResponse:
        """Register an execution"""
        try:
            run_details = ExecutionRunDetails(
                start_time=datetime.now(), status=ExecutionStatus.RUNNING
            )
            execution = AutomationExecution(
                execution_id=request.execution_id, run_details=run_details
            )
            self.automation_executions[request.automation_id].append(execution)
            return RegisterAutomationExecutionResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def UpdateAutomationExecution(
        self, request: UpdateAutomationExecutionRequest, context
    ) -> UpdateAutomationExecutionResponse:
        """Update an execution"""
        try:
            existing_executions = list(
                itertools.chain(*self.automation_executions.values())
            )
            relevant_execution = list(
                filter(
                    lambda ex: ex.execution_id == request.execution_id,
                    existing_executions,
                )
            )[0]
            self.__merge_run_details(
                relevant_execution.run_details, request.run_details
            )
            return UpdateAutomationExecutionResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ToggleAutomationActivity(
        self, request: ToggleAutomationActivityRequest, context
    ) -> ToggleAutomationActivityResponse:
        """Toggle whether an automation is enabled or disabled"""
        try:
            self.automations[request.automation_id].automation_spec = request.is_enabled
            return ToggleAutomationActivityResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def RunAutomation(
        self, request: RunAutomationRequest, context
    ) -> RunAutomationResponse:
        """Run an automation"""
        if self.automations[request.automation_id]:
            return RunAutomationResponse()
        else:
            raise_internal_grpc_error(
                context,
                ValueError(f"Automation with ID {request.automation_id} not found"),
            )

    @staticmethod
    def __merge_run_details(
        run_details_1: ExecutionRunDetails, run_details_2: ExecutionRunDetails
    ):
        run_details_1.start_time = (
            run_details_2.start_time
            if run_details_2.start_time
            else run_details_1.start_time
        )
        run_details_1.end_time = (
            run_details_2.end_time if run_details_2.end_time else run_details_1.end_time
        )
        run_details_1.error_details = (
            run_details_2.error_details
            if run_details_2.error_details
            else run_details_1.error_details
        )
        run_details_1.status = (
            run_details_2.status if run_details_2.status else run_details_1.status
        )
        run_details_1.task = (
            run_details_2.task if run_details_2.task else run_details_1.task
        )
        run_details_1.finish_cause = (
            run_details_2.finish_cause
            if run_details_2.finish_cause
            else run_details_1.finish_cause
        )

        return run_details_1
