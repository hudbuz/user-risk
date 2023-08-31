import logging
import time
from abc import ABC, abstractmethod
from random import randint
from typing import Optional, Tuple, Callable

import grpc
from qwak.exceptions import QwakException

from .grpc_auth import Auth0Client

logger = logging.getLogger()


def create_grpc_channel(
    url: str,
    enable_ssl: bool = True,
    enable_auth: bool = True,
    auth_metadata_plugin: grpc.AuthMetadataPlugin = None,
    timeout: int = 100,
    options=None,
    backoff_options={},
    max_attempts=4,
    status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
    attempt=0,
) -> grpc.Channel:
    """
    Create a gRPC channel
    Args:
        url: gRPC URL to connect to
        enable_ssl: Enable TLS/SSL, optionally provide a server side certificate
        enable_auth: Enable user auth
        auth_metadata_plugin: Metadata plugin to use to sign requests, only used
            with "enable auth" when SSL/TLS is enabled
        timeout: Connection timeout to server
        options:  An optional list of key-value pairs (channel_arguments in gRPC Core runtime) to configure the channel.
        backoff_options: dictionary - init_backoff_ms: default 50, max_backoff_ms: default 500, multiplier: default 2
        max_attempts: max number of retry attempts
        attempt: current retry attempts
        status_for_retry: grpc statuses to retry upon
    Returns: Returns a grpc.Channel
    """
    if not url:
        raise QwakException("Unable to create gRPC channel. URL has not been defined.")

    if enable_ssl or url.endswith(":443"):
        credentials = grpc.ssl_channel_credentials()
        if enable_auth:
            if auth_metadata_plugin is None:
                auth_metadata_plugin = Auth0Client()
            credentials = grpc.composite_channel_credentials(
                credentials, grpc.metadata_call_credentials(auth_metadata_plugin)
            )

        channel = grpc.secure_channel(url, credentials=credentials, options=options)
    else:
        channel = grpc.insecure_channel(url, options=options)
    try:
        interceptors = (
            RetryOnRpcErrorClientInterceptor(
                max_attempts=max_attempts,
                sleeping_policy=ExponentialBackoff(**backoff_options),
                status_for_retry=status_for_retry,
            ),
        )

        intercept_channel = grpc.intercept_channel(channel, *interceptors)
        try:
            grpc.channel_ready_future(intercept_channel).result(timeout=timeout)
        except (grpc.FutureTimeoutError, grpc.RpcError, Exception) as e:
            logger.debug(
                f"Received error: {repr(e)} attempt #{attempt + 1} of {max_attempts}"
            )
            if attempt < max_attempts:
                return create_grpc_channel(
                    url,
                    enable_ssl,
                    enable_auth,
                    auth_metadata_plugin,
                    timeout,
                    options,
                    backoff_options,
                    max_attempts,
                    status_for_retry,
                    attempt + 1,
                )
            else:
                raise e

        return intercept_channel
    except grpc.FutureTimeoutError as e:
        raise QwakException(
            f"Connection timed out while attempting to connect to {url}, with: {repr(e)}"
        ) from e


def create_grpc_channel_or_none(
    url: str,
    enable_ssl: bool = True,
    enable_auth: bool = True,
    auth_metadata_plugin: grpc.AuthMetadataPlugin = None,
    timeout: int = 30,
    options=None,
    backoff_options={},
    max_attempts=2,
    status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
    attempt=0,
) -> Callable[[Optional[str], Optional[bool]], Optional[grpc.Channel]]:
    def deferred_channel(
        url_overwrite: Optional[str] = None, ssl_overwrite: Optional[bool] = None
    ):
        try:
            return create_grpc_channel(
                url_overwrite if url_overwrite else url,
                ssl_overwrite if ssl_overwrite else enable_ssl,
                enable_auth,
                auth_metadata_plugin,
                timeout,
                options,
                backoff_options,
                max_attempts,
                status_for_retry,
                attempt,
            )
        except Exception as e:
            logger.debug(f"create_grpc_channel error: {repr(e)}")
            return None

    return deferred_channel


class SleepingPolicy(ABC):
    @abstractmethod
    def sleep(self, try_i: int):
        """
        How long to sleep in milliseconds.
        :param try_i: the number of retry (starting from zero)
        """
        if try_i < 0:
            raise ValueError("Number of retries must be non-negative.")


class ExponentialBackoff(SleepingPolicy):
    def __init__(
        self,
        *,
        init_backoff_ms: int = 50,
        max_backoff_ms: int = 5000,
        multiplier: int = 2,
    ):
        self.init_backoff = init_backoff_ms
        self.max_backoff = max_backoff_ms
        self.multiplier = multiplier

    def sleep(self, try_i: int):
        sleep_time = min(self.init_backoff * self.multiplier**try_i, self.max_backoff)
        sleep_ms = sleep_time + randint(0, self.init_backoff)  # nosec B311
        logger.debug("ExponentialBackoff - Sleeping between retries")
        time.sleep(sleep_ms / 1000)


class RetryOnRpcErrorClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(
        self,
        *,
        max_attempts: int,
        sleeping_policy: SleepingPolicy,
        status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        self.max_attempts = max_attempts
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        for try_i in range(self.max_attempts):
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):
                # Return if it was last attempt
                if try_i == (self.max_attempts - 1):
                    return response

                # If status code is not in retryable status codes
                if (
                    self.status_for_retry
                    and response.code() not in self.status_for_retry
                ):
                    return response
                logger.debug(
                    f"Retry GRPC call attempt #{try_i} after status {response.code()}"
                )
                logger.debug(f"Client call details: {client_call_details}")
                self.sleeping_policy.sleep(try_i)
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)
