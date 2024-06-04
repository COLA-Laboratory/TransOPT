import logging

from rich.logging import RichHandler

loggers = {}

LOGGER_NAME = "Transopt"


def get_logger(logger_name: str) -> logging.Logger:
    # https://rich.readthedocs.io/en/latest/reference/logging.html#rich.logging.RichHandler
    # https://rich.readthedocs.io/en/latest/logging.html#handle-exceptions
    if logger_name in loggers:
        return loggers[logger_name]
    
    _logger = logging.getLogger(logger_name) 
    rich_handler = RichHandler(
        show_time=False,
        rich_tracebacks=False,
        show_path=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
    )

    file_handler = logging.FileHandler('application.log') 
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    _logger.handlers.clear()
    _logger.addHandler(rich_handler)
    # _logger.addHandler(file_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    
    loggers[logger_name] = _logger
    return _logger

# logger = logging.getLogger(LOGGER_NAME)
# logger.setLevel(logging.DEBUG)

logger = get_logger(LOGGER_NAME)
