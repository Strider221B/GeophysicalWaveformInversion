import logging

from configs.config import Config

class Logger:

    _logger = None

    @classmethod
    def get_logger(cls) -> logging:
        if cls._logger is None:
            logging.basicConfig(format="{asctime} - {levelname} - {message}",
                                style="{",
                                datefmt="%Y-%m-%d %H:%M:%S",
                                level=Config.log_level)
            cls._logger = logging
        return cls._logger
