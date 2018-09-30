from __future__ import division
from __future__ import print_function

from config import cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.py_utils import QuickLogger


if __name__ == "__main__":
    print(cfg.output_dir)
    log = QuickLogger(log_dir=cfg.output_dir).get_logger()
    log.info(cfg)
    log.info("hello")
    log.info("hello")
    log.info("hello")
    log.info("hello")
    log.info("hello")
    log.info("hello")
    log.info("hello")
