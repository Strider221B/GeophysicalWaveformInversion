import logging

from configs.config import Config

class Logger:

    _logger = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = cls._initialize_and_get_logger()
        return cls._logger

    @staticmethod
    def _initialize_and_get_logger() -> logging.Logger:
        logger = logging.getLogger('analytics_logger')
        logger.setLevel(Config.log_level)
        ch = logging.StreamHandler()
        ch.setLevel(Config.log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
