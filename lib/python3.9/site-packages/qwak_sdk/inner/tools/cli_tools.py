import functools
import time
import warnings
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple, Union

import click
from click import Context, OptionParser
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import UserAccountConfiguration
from qwak.inner.di_configuration.session import Session

from qwak_sdk.inner.tools.tracking import log_event


def get_user_identifiers(account: UserAccountConfiguration):
    try:
        user_id = account.get_user_config().api_key
        identifier_cls = namedtuple("Identifier", ["user_id"])
        return identifier_cls(user_id)

    except QwakException:
        # User might not be registered or have a valid API key
        return None


def profile_setter_wrapper(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        Session().set_environment(ctx.params["environment"])
        return func(*args, **kwargs)

    return wrapper


def usage_statistics_wrapper(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        start_time = time.time()

        account = UserAccountConfiguration()
        user_identifiers = get_user_identifiers(account)

        # If we could not get user identifiers for some reason (for example - user not logged in) - do not track
        if not user_identifiers:
            return func(*args, **kwargs)

        event_properties = {
            "command_group": ctx.command_path,
            "commands": ctx.command.name,
            "duration": time.time() - start_time,
        }
        try:
            return_value = func(*args, **kwargs)

            event_properties["response"] = return_value
            event_properties["status"] = 0
            log_event(
                event_properties,
                user_id=user_identifiers.user_id,
            )

            return return_value

        except BaseException as e:
            command_status = 1
            if type(e) == KeyboardInterrupt:
                command_status = 1

            event_properties["error_type"] = type(e).__name__
            event_properties["error_message"] = str(e)
            event_properties["status"] = command_status

            log_event(
                event_properties,
                user_id=user_identifiers.user_id,
            )
            raise

    return wrapper


class DeprecatedOption(click.Option):
    def __init__(self, *args, **kwargs):
        self.deprecated = kwargs.pop("deprecated", ())
        self.preferred = kwargs.pop("preferred", args[0][-1])
        super(DeprecatedOption, self).__init__(*args, **kwargs)


class QwakCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(
            -1,
            click.core.Option(
                ("--environment",),
                default="default",
                required=False,
                is_eager=True,
                help="Qwak environment",
            ),
        )

    def make_parser(self, ctx: Context) -> OptionParser:
        parser = super(QwakCommand, self).make_parser(ctx)

        # get the parser options
        options = set(parser._short_opt.values())
        options |= set(parser._long_opt.values())

        for option in options:
            if not isinstance(option.obj, DeprecatedOption):
                continue

            def make_process(an_option):
                """Construct a closure to the parser option processor"""

                orig_process = an_option.process
                deprecated = getattr(an_option.obj, "deprecated", None)
                preferred = getattr(an_option.obj, "preferred", None)

                if not deprecated:
                    f"Expected `deprecated` value for `{an_option.obj.name}`"

                def process(value, state):
                    """The function above us on the stack used 'opt' to
                    pick option from a dict, see if it is deprecated"""

                    # reach up the stack and get 'opt'
                    import inspect

                    frame = inspect.currentframe()
                    try:
                        opt = frame.f_back.f_locals.get("opt")
                    finally:
                        del frame

                    if opt in deprecated:
                        warnings.warn(
                            f"{opt} has been deprecated, use {preferred}."
                            f" {opt} will be removed in future releases.",
                            DeprecationWarning,
                        )

                    return orig_process(value, state)

                return process

            option.process = make_process(option)

        return parser

    @profile_setter_wrapper
    @usage_statistics_wrapper
    def invoke(self, ctx: click.Context) -> Any:
        return super().invoke(ctx)


def ask_yesno(
    question: str, force: bool, print_callback: Callable[[str], None] = print
) -> bool:
    """
    Helper to get yes / no answer from user.

    Args:
        question: yes/no question to show the user
        force: automatically returns True if force is True
        print_callback: function to use for printing

    Returns:
        True/False by user input (unless forced and then always True)
    """

    print_callback(f"{question}")
    if force:
        print_callback("Forced yes")
        return True
    return click.confirm("continue?")


def dictify_params(env_vars: Union[Dict[str, str], List[str], Tuple[str]]):
    result = dict()

    if env_vars is None:
        return result

    if isinstance(env_vars, dict):
        return env_vars

    if isinstance(env_vars, (list, tuple)):
        for env_var in env_vars:
            if "=" not in env_var:
                raise QwakException(
                    f'The environment variable definition passed {env_var} is invalid. Format is "KEY=VALUE"'
                )
            split_param = env_var.split("=")
            result[split_param[0]] = split_param[1]

    return result
