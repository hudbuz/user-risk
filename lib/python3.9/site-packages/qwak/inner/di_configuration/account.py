import configparser
import errno
import os
from dataclasses import dataclass
from typing import Optional

from qwak.exceptions import QwakLoginException
from qwak.inner.const import QwakConstants
from qwak.inner.di_configuration.session import Session
from qwak.inner.tool.auth import Auth0ClientBase


@dataclass
class UserAccount:
    """
    User Account Configuration
    """

    # Assigned API Key
    api_key: str

    # Assigned username
    username: Optional[str] = None


class UserAccountConfiguration:
    USER_FIELD = "user"

    API_KEY_FIELD = "api_key"

    def __init__(
        self,
        config_file=QwakConstants.QWAK_CONFIG_FILE,
        auth_file=QwakConstants.QWAK_AUTHORIZATION_FILE,
        auth_client=Auth0ClientBase,
    ):
        self._config_file = config_file
        self._auth_file = auth_file
        self._config = configparser.ConfigParser()
        self._auth = configparser.ConfigParser()
        self._auth_client = auth_client
        self._environment = Session().get_environment()

    def configure_user(self, user_account: UserAccount):
        """
        Write user account to the given config file in an ini format. Configuration will be written under the 'default'
        section
        :param user_account: user account properties to be written
        """
        self._auth.read(self._auth_file)
        self._auth.remove_section(self._environment)
        with self._safe_open(self._auth_file) as authfile:
            self._auth.write(authfile)

        self._auth_client(
            api_key=user_account.api_key,
            auth_file=self._auth_file,
        ).login()

        self._config.read(self._config_file)

        with self._safe_open(self._config_file) as configfile:
            self._config[self._environment] = {}
            if user_account.username:
                self._config[self._environment][self.USER_FIELD] = user_account.username
            self._config[self._environment][self.API_KEY_FIELD] = user_account.api_key
            self._config.write(configfile)

    @staticmethod
    def _mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
                raise

    @staticmethod
    def _safe_open(path):
        UserAccountConfiguration._mkdir_p(os.path.dirname(path))
        return open(path, "w")

    def get_user_config(self):
        """
        Get persisted user account from config file
        :return:
        """
        try:
            username = os.environ.get("QWAK_USERNAME")
            api_key = os.environ.get("QWAK_API_KEY")
            if api_key:
                Session().set_environment(api_key)
                return UserAccount(username=username, api_key=api_key)
            else:
                self._config.read(self._config_file)
                return UserAccount(
                    username=self._config.get(
                        section=self._environment, option=self.USER_FIELD, fallback=None
                    ),
                    api_key=self._config.get(
                        section=self._environment, option=self.API_KEY_FIELD
                    ),
                )

        except FileNotFoundError:
            raise QwakLoginException(
                f"Could not read user configuration from {self._config_file}. "
                f"Please make sure one has been set using `qwak configure` command"
            )

        except configparser.NoSectionError:
            raise QwakLoginException(
                f"Could not read user configuration from {self._config_file}. "
                f"Please make sure one has been set using `qwak configure` command"
            )

    def get_user_apikey(self) -> str:
        """
        Get persisted user account from config file
        :return:
        """
        try:
            api_key = os.environ.get("QWAK_API_KEY")
            if api_key:
                Session().set_environment(api_key)
                return api_key
            else:
                self._config.read(self._config_file)
                return self._config.get(
                    section=self._environment, option=self.API_KEY_FIELD
                )

        except FileNotFoundError:
            raise QwakLoginException(
                f"Could not read user configuration from {self._config_file}. "
                f"Please make sure one has been set using `qwak configure` command"
            )

        except configparser.NoSectionError:
            raise QwakLoginException(
                f"Could not read user configuration from {self._config_file}. "
                f"Please make sure one has been set using `qwak configure` command"
            )
