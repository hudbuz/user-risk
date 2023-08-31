import logging
import sys
from contextlib import contextmanager
from typing import Callable, Optional

from qwak.tools.logger.logger import get_qwak_logger, get_qwak_logger_verbosity_level
from yaspin import yaspin

logger = get_qwak_logger()


@contextmanager
def qwak_spinner(
    begin_text: Optional[str],
    end_text: Optional[str] = "",
    print_callback: Callable[[str], None] = logger.info,
):
    """Qwak spinner.

    Args:
        begin_text: Text to shown when spinner starts.
        end_text: Text to shown when spinner ends.
        print_callback: Callback used to print the output.
    """
    if (
        logging.getLevelName(get_qwak_logger_verbosity_level()) < logging.WARNING
        if print_callback != print
        else False
    ) or not sys.stdout.isatty():
        print_callback(begin_text)
        yield
        print_callback(end_text)
    else:
        with yaspin(text=begin_text, color="blue", timer=True) as sp:
            try:
                yield sp
            except Exception as e:
                sp.fail("💥")
                raise e
            if end_text:
                sp.text = end_text
            sp.ok("✅")
