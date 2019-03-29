import logging
import os


def build_logger(name: str, log_path: str, log_level: str = "INFO"):
    try:
        os.mkdir(log_path)
    except FileExistsError:
        pass

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    handler = logging.FileHandler("{}/{}.log".format(log_path, name), encoding="utf-8")
    handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger
