import logging


def set_logger():
    logFormatter = logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger()

    fileHandler = logging.FileHandler("logs/dsc.log", mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # logger.addHandler(consoleHandler)

    return logger