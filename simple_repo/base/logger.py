import logging


def config_log(level=logging.ERROR):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_debug(obj, msg: str):
    logging.debug("{}:{}".format(obj.__class__.__name__, msg))


def log_info(obj, msg: str):
    logging.info("{}:{}".format(obj.__class__.__name__, msg))


def log_bare_info(msg: str):
    logging.info("{}".format(msg))


def log_error(obj, msg: str):
    logging.error("{}:{}".format(obj.__class__.__name__, msg))


def log_info_param(obj, **param):
    logging.info(
        "{}:Setting parameters [{}]".format(
            obj.__class__.__name__,
            ", ".join("{}: {}".format(k, v) for k, v in param.items()),
        )
    )


def log_val(msg: str, value):
    logging.info("{} {}".format(msg, value))
