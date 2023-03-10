from typing import Final
from logging import INFO, getLogger, FileHandler, StreamHandler, Formatter, Logger


def init_logger(name: str) -> Logger:
    """Initialize the module logger."""

    Logger: Final = getLogger(name)
    Logger.setLevel(INFO)

    # create file handler which logs even debug messages
    fh = FileHandler('run_logs.log')
    fh.setLevel(INFO)
    # create console handler with a higher log level
    ch = StreamHandler()
    ch.setLevel(INFO)
    # create formatter and add it to the handlers
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    Logger.addHandler(ch)
    Logger.addHandler(fh)

    return Logger
    