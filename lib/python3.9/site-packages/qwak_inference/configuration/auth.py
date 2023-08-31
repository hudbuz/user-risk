import base64
import configparser
import json
from datetime import datetime

import requests

from qwak_inference.configuration.session import Session
from qwak_inference.constants import QwakConstants
from qwak_inference.exceptions import QwakLoginException


class AuthClient:
    _TOKENS_FIELD = "TOKENS"

    def __init__(
        self,
        api_key,
        auth_file=None,
    ):
        if auth_file:
            self._auth_file = auth_file
        else:
            self._auth_file = QwakConstants.QWAK_AUTHORIZATION_FILE

        self._config = configparser.ConfigParser()
        self._environment = Session().get_environment()
        self.token = None
        self.api_key = api_key

    # Returns Non if token is expired
    def get_token(self):
        try:
            if not self.token:
                self._config.read(self._auth_file)
                self.token = json.loads(
                    self._config.get(
                        section=self._environment, option=self._TOKENS_FIELD
                    )
                )

            # Check that token isn't expired
            if datetime.utcnow() >= self.token_expiration():
                self.login()
                return self.token
            else:
                return self.token
        except configparser.NoSectionError:
            self.login()
            return self.token

    def login(self):
        try:
            response = requests.post(
                QwakConstants.QWAK_AUTHENTICATION_URL,
                json={"qwakApiKey": self.api_key},
                timeout=30,
            )
            if response.status_code == 200:
                self.token = response.json()["accessToken"]
            else:
                raise QwakLoginException(f"Error: {response.reason}")

            from pathlib import Path

            Path(self._auth_file).parent.mkdir(parents=True, exist_ok=True)
            self._config.read(self._auth_file)

            with open(self._auth_file, "w") as configfile:
                self._config[self._environment] = {
                    self._TOKENS_FIELD: json.dumps(self.token)
                }

                self._config.write(configfile)
        except Exception as e:
            raise e

    def token_expiration(self) -> datetime:
        if not self.token:
            self.login()
        tokenSplit = self.token.split(".")
        decoded_token = json.loads(_base64url_decode(tokenSplit[1]).decode("utf-8"))
        return datetime.fromtimestamp(decoded_token["exp"])


def _base64url_decode(input):
    rem = len(input) % 4
    if rem > 0:
        input += "=" * (4 - rem)

    return base64.urlsafe_b64decode(input)
