import uuid
from typing import List, Optional

from _qwak_proto.qwak.automation.v1.automation_management_service_pb2 import (
    CreateAutomationRequest,
    DeleteAutomationRequest,
    GetAutomationByNameRequest,
    GetAutomationRequest,
    ListAutomationExecutionsRequest,
    ListAutomationsRequest,
    RegisterAutomationExecutionRequest,
    RunAutomationRequest,
    UpdateAutomationExecutionRequest,
    UpdateAutomationRequest,
)
from _qwak_proto.qwak.automation.v1.automation_management_service_pb2_grpc import (
    AutomationManagementServiceStub,
)
from _qwak_proto.qwak.automation.v1.automation_pb2 import Automation as AutomationProto
from dependency_injector.wiring import Provide
from grpc import RpcError, StatusCode
from qwak.automations.automation_executions import (
    AutomationExecution,
    ExecutionRunDetails,
)
from qwak.automations.automations import Automation
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class AutomationsManagementClient:
    """
    Used for interacting with Automations Registry endpoints
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._service_stub = AutomationManagementServiceStub(grpc_channel)

    def get_automation_by_name(self, automation_name: str) -> Optional[Automation]:
        """

        Args:
            automation_name: automation name to get
        Returns:
            Automation
        """
        try:
            automation: AutomationProto = self._service_stub.GetAutomationByName(
                GetAutomationByNameRequest(automation_name=automation_name)
            ).automation
            return Automation.from_proto(automation)
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                return None
            else:
                raise QwakException(
                    f"Failed to fetch automation by name [{automation_name}], error code: [{e.details()}]"
                )

    def get_automation_by_id(self, automation_id: str) -> Optional[Automation]:
        """

        Args:
            automation_id: automation name to get
        Returns:
            Automation
        """
        try:
            automation: AutomationProto = self._service_stub.GetAutomation(
                GetAutomationRequest(automation_id=automation_id)
            ).automation
            return Automation.from_proto(automation)
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                return None
            else:
                raise QwakException(
                    f"Failed to fetch automation with ID [{automation_id}], error code: [{e.details()}]"
                )

    def update_automation(
        self, automation_id: str, automation: AutomationProto
    ) -> None:
        """

        Args:
            automation: automation to update
        """
        automation.automation_spec.is_sdk_v1 = True
        try:
            self._service_stub.UpdateAutomation(
                UpdateAutomationRequest(
                    automation_id=automation_id,
                    automation_spec=automation.automation_spec,
                )
            )
            return Automation.from_proto(automation)
        except RpcError as e:
            raise QwakException(
                f"Failed to update automation with ID [{automation.automation_id}], error code: [{e.details()}]"
            )

    def create_automation(self, automation: AutomationProto) -> str:
        """

        Args:
            automation: automation to create
        Returns:
            Automation id of created automation
        """
        automation.automation_spec.is_sdk_v1 = True
        try:
            return self._service_stub.CreateAutomation(
                CreateAutomationRequest(automation_spec=automation.automation_spec)
            ).automation_id
        except RpcError as e:
            raise QwakException(
                f"Failed to update automation, error code: [{e.details()}]"
            )

    def run_automation(self, automation_id: str) -> str:
        """
        Trigger an automation to run

        Args:
            automation_id: automation id to trigger
        """
        try:
            return self._service_stub.RunAutomation(
                RunAutomationRequest(automation_id=automation_id)
            )
        except RpcError as e:
            raise QwakException(
                f"Failed to execute a run automation command, error code: [{e.details()}]"
            )

    def list_automations(
        self,
    ) -> Optional[List[Automation]]:
        """

        Returns:
            List of client automations
        """
        try:
            automations: List[AutomationProto] = self._service_stub.ListAutomations(
                ListAutomationsRequest()
            ).automations
            return list(
                map(lambda automation: Automation.from_proto(automation), automations)
            )
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                return None
            else:
                raise QwakException(
                    f"Failed to fetch automations, error code: [{e.details()}]"
                )

    def delete_automation(self, automation_id: str) -> bool:
        """

        Args:
            automation_id: The automation id to delete

        Returns:
            boolean whether the automation was deleted or not
        """
        try:
            self._service_stub.DeleteAutomation(
                DeleteAutomationRequest(automation_id=automation_id)
            )
            return True
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                return False
            else:
                raise QwakException(
                    f"Failed to fetch automations, error code: [{e.details()}]"
                )

    def register_execution(self, automation_id: str) -> str:
        """

        Args:
            automation_id: The automation id to register

        Returns:
            The execution id for the automation execution

        """
        try:
            execution_id = str(uuid.uuid4())
            self._service_stub.RegisterAutomationExecution(
                RegisterAutomationExecutionRequest(
                    automation_id=automation_id, execution_id=execution_id
                )
            )
            return execution_id
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                raise QwakException(
                    f"Failed to register automation execution, automation {automation_id} was not found"
                )
            else:
                raise QwakException(
                    f"Failed to register automation execution, error code: [{e.details()}]"
                )

    def update_execution(
        self, execution_id: str, run_details: ExecutionRunDetails
    ) -> None:
        """

        Args:
            execution_id: The execution id to update
            run_details: The details to update on the execution

        """
        try:
            self._service_stub.UpdateAutomationExecution(
                UpdateAutomationExecutionRequest(
                    execution_id=execution_id, run_details=run_details.to_proto()
                )
            )
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                raise QwakException(
                    f"Failed to update execution, execution {execution_id} was not found"
                )
            else:
                raise QwakException(
                    f"Failed to update execution, error code: [{e.details()}]"
                )

    def list_executions(self, automation_id: str) -> List[AutomationExecution]:
        """

        Args:
            automation_id: The automation to list its executions

        Returns:
            A list of executions for the specified automation

        """
        try:
            executions = self._service_stub.ListAutomationExecutions(
                ListAutomationExecutionsRequest(automation_id=automation_id)
            ).automation_executions
            return list(
                map(
                    lambda execution: AutomationExecution.from_proto(execution),
                    executions,
                )
            )
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                raise QwakException(
                    f"Failed to list executions for automation, automation {automation_id} was not found"
                )
            else:
                raise QwakException(
                    f"Failed to list executions for automation, error code: [{e.details()}]"
                )
