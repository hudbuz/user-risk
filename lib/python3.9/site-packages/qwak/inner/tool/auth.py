import warnings

from qwak.inner.di_configuration.session import Session

warnings.filterwarnings(action="ignore", module=".*jose.*")

import configparser  # noqa E402
import json  # noqa E402

import requests  # noqa E402
from jose import jwt  # noqa E402
from qwak.exceptions import QwakLoginException  # noqa E402
from qwak.inner.const import QwakConstants  # noqa E402


class Auth0ClientBase:
    _TOKENS_FIELD = "TOKENS"

    def __init__(
        self,
        api_key,
        auth_file=QwakConstants.QWAK_AUTHORIZATION_FILE,
        audience=QwakConstants.TOKEN_AUDIENCE,
    ):
        self._auth_file = auth_file
        self._config = configparser.ConfigParser()
        self._environment = Session().get_environment()
        self.jwks = requests.get(QwakConstants.AUTH0_JWKS_URI, timeout=60).json()
        self.token = None
        self.audience = audience
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

            # Test that token isn't expired
            self.get_claims()
            return self.token
        except configparser.NoSectionError:
            self.login()
            return self.token
        except jwt.ExpiredSignatureError:
            self.login()
            return self.token

    def login(self):
        from qwak.clients.administration import AuthenticationClient

        self.token = AuthenticationClient().authenticate(self.api_key).access_token

        from pathlib import Path

        Path(self._auth_file).parent.mkdir(parents=True, exist_ok=True)
        self._config.read(self._auth_file)

        with open(self._auth_file, "w") as configfile:
            self._config[self._environment] = {
                self._TOKENS_FIELD: json.dumps(self.token)
            }

            self._config.write(configfile)

    def get_claims(self):
        try:
            if not self.token:
                self.get_token()
            unverified_header = jwt.get_unverified_header(self.token)
            rsa_key = {}
            for key in self.jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "use": key["use"],
                        "n": key["n"],
                        "e": key["e"],
                    }
            if rsa_key:
                payload = jwt.decode(
                    self.token,
                    rsa_key,
                    algorithms=QwakConstants.AUTH0_ALGORITHMS,
                    audience=self.audience,
                )
                claims = {}
                token_prefix = QwakConstants.TOKEN_AUDIENCE
                claims["exp"] = payload["exp"]
                for key in payload:
                    if key.startswith(token_prefix):
                        claims[key.split(token_prefix)[1]] = payload[key]
                return claims
            raise QwakLoginException()
        except jwt.ExpiredSignatureError as e:
            raise e
        except Exception:
            raise QwakLoginException()
