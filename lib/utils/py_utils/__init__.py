# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import logging, os


class QuickLogger:
    def __init__(self, log_dir, log_name='train_logs.txt'):
        # set log
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        formatter = logging.Formatter("%(message)s")

        file_log = logging.FileHandler(log_file, mode='w')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        # console_log = logging.StreamHandler()
        # console_log.setLevel(logging.INFO)
        # console_log.setFormatter(formatter)
        # self.logger.addHandler(console_log)
        self.logger.addHandler(file_log)

    def get_logger(self):
        return self.logger
