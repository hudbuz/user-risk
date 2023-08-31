import logging
import platform
import sys

import requests

from qwak_sdk import __version__ as sdk_version

logger = logging.getLogger(__name__)


def _amplitude_url():
    return "https://api2.amplitude.com/2/httpapi"


def _platform_info():
    return platform.platform()


def _python_version():
    return "{major}.{minor}.{micro}".format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        micro=sys.version_info.micro,
    )


def log_event(properties, user_id):
    properties["sdk_version"] = sdk_version
    properties["python_version"] = _python_version()
    properties["platform_info"] = _platform_info()

    event = [
        {
            "event_type": "sdk-event",
            "user_id": user_id,
            "event_properties": properties,
            "ip": "$remote",
        }
    ]

    event_data = {"events": event}

    try:
        return requests.post(_amplitude_url(), json=event_data, timeout=1)
    except Exception as e:
        logger.debug(str(e))
