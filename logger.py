import logging
import sys

class CustomFormatter(logging.Formatter):

    blue = "\x1b[36;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    colored_part = "%(asctime)s - %(name)s - %(levelname)s"
    noncolored_part = " - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + colored_part + reset + noncolored_part,
        logging.INFO: green + colored_part + reset + noncolored_part,
        logging.WARNING: yellow + colored_part + reset + noncolored_part,
        logging.ERROR: red + colored_part + reset + noncolored_part,
        logging.CRITICAL: bold_red + colored_part + reset + noncolored_part
    }

    def format(self, record):
        record.name = record.name.split("/")[-1]
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(logger_name: str):
    """
    Usually, loggers get named with __file__.
    You can pass __file__ as the argument when you get the logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.propagate = False
    return logger