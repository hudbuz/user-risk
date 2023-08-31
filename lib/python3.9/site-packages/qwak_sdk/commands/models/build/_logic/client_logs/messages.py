BUILD_LOGS_URL = (
    "https://app.qwak.ai/projects/{project_uuid}/{model_id}/build/{build_id}"
)

SUCCESS_MSG_REMOTE = """
Build ID \033[4m{build_id}\033[0m triggered remotely

########### To follow build logs using CLI
qwak models builds logs -b {build_id} --follow

########### To follow build logs using Qwak platform
https://app.qwak.ai/projects/{project_uuid}/{model_id}/build/{build_id}
"""

SUCCESS_MSG_REMOTE_WITH_DEPLOY = """
Build ID \033[4m{build_id}\033[0m finished successfully and deployed

########### To view the model using Qwak platform
https://app.qwak.ai/projects/{project_uuid}/{model_id}
"""

FAILED_CONTACT_QWAK_SUPPORT = """
Build ID \033[4m{build_id}\033[0m failed!!
You can share the logs from \033[4m{log_file}.zip\033[0m with Qwak support.
"""

FAILED_CONTACT_QWAK_SUPPORT_PROGRAMMATIC = """
Build ID \033[4m{build_id}\033[0m failed!!
"""

FAILED_REMOTE_BUILD_SUGGESTION = """
Your build failed. You can check the reason for the failure in the Qwak Platform:
https://app.qwak.ai/projects/{project_uuid}/{model_id}/build/{build_id}
"""

FAILED_DEPLOY_BUILD_SUGGESTION = """
Deploying the build Failed. You can check the reason for the failure in the Qwak Platform:
https://app.qwak.ai/projects/{project_uuid}/{model_id}?tabId=1
Since the build finished successfully, you can try and redeploy it either from the platform or from the CLI
"""
