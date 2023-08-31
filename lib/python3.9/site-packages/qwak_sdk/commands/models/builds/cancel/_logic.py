from qwak.clients.build_orchestrator import BuildOrchestratorClient


def execute_cancel_build(build_id):
    return BuildOrchestratorClient().cancel_build_model(build_id=build_id)
