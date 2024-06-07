import logging


def create_logger(name: str, file_path: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
