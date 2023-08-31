import json
from enum import Enum
from json import JSONDecodeError

from qwak_inference.constants import QwakConstants
from qwak_inference.realtime_client.rest_helpers import RestSession

try:
    from qwak.exceptions import QwakHTTPException
except ImportError:
    from qwak_inference.exceptions import QwakHTTPException

_TRAFFIC_SPLIT_HEADER = "x-realtime-route-mode"


class InferenceOutputFormat(Enum):
    PANDAS = 1
    DICTIONARY = 2


DEFAULT_ENVIRONMENT_ERROR_MESSAGE = (
    "Failed to get default environment name. This might be due to connectivity "
    "issues or missing account configuration."
)


class RealTimeClient(object):
    def __init__(
        self,
        model_id: str,
        environment: str = None,
        variation: str = None,
        model_api: str = None,
    ):
        """

        :param model_id: The model id to invoke against. If not provided, will attempt to extract the model ID from the
        `QWAK_MODEL_ID` environment variable
        """

        self.r_session = RestSession()

        if not environment:
            environment = _get_default_environment_name(self.r_session)

        self.model_id = model_id
        self.content_type = "application/json; format=pandas-split"
        self.model_api = (
            model_api if model_api else _get_model_url(model_id, environment, variation)
        )

    def predict(
        self,
        feature_vectors,
        output_format=InferenceOutputFormat.DICTIONARY,
        metadata: dict = {},
    ):
        """
        Perform a prediction request against a Qwak based model

        :param feature_vectors: A list of feature vectors to predict against. Each feature vector is modeled as a python
         dictionary, or a pandas Dataframe
        :param output_format: A list of feature vectors to predict against. Each feature vector is modeled as a python
         dictionary, or a pandas Dataframe
        :param metadata: metadata to split traffic based on
        :return: Prediction response from the model
        """

        if feature_vectors.__class__.__name__ == "DataFrame":
            feature_vectors = feature_vectors.to_json(orient="split")

        if isinstance(feature_vectors, dict) or isinstance(feature_vectors, list):
            feature_vectors = json.dumps(feature_vectors)

        try:
            rule_based_traffic = {}
            if metadata:
                rule_based_traffic = metadata.copy()
                rule_based_traffic[_TRAFFIC_SPLIT_HEADER] = "true"

            response = self.r_session.post(
                self.model_api, data=feature_vectors, headers=rule_based_traffic
            )

            if response.status_code >= 400:
                exception_class_name = response.headers.get("X-Exception-Class")
                msg = f"{response.status_code}: {response.text}"
                raise QwakHTTPException(response.status_code, msg, exception_class_name)

            elif response.status_code != 200:
                raise QwakHTTPException(response.status_code, response.content)

            if output_format == InferenceOutputFormat.DICTIONARY:
                dict_response = json.loads(response.content)
                return dict_response
            elif output_format == InferenceOutputFormat.PANDAS:
                try:
                    import pandas as pd
                except ImportError:
                    raise ImportError(
                        "Pandas is not installed. Please install pandas to use the pandas output format"
                    )
                dict_response = json.loads(response.content)
                return pd.DataFrame(dict_response)
            else:
                raise ValueError(
                    f"Non supported inference output parameter {output_format}"
                )
        except QwakHTTPException as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to make a prediction request. Error is: {e}")


def _get_default_environment_name(r_session) -> str:
    try:
        response = r_session.post(QwakConstants.QWAK_AUTHENTICATED_USER_ENDPOINT)
        account_details = response.json()["authenticatedUserContext"]["user"][
            "accountDetails"
        ]
        default_environment_id = account_details["defaultEnvironmentId"]
        return account_details["environmentById"][default_environment_id]["name"]
    except JSONDecodeError:
        raise RuntimeError(DEFAULT_ENVIRONMENT_ERROR_MESSAGE)
    except Exception:
        raise RuntimeError(DEFAULT_ENVIRONMENT_ERROR_MESSAGE + " Error is: {e}")


def _get_model_url(model_id, environment, variation=None) -> str:
    model_route = f"{model_id}/{variation}" if variation else model_id
    return f"https://models.{environment}.qwak.ai/v1/{model_route}/predict"
