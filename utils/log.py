import logging
from pathlib import Path


def set_logger(args):
    logFormatter = logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger()

    path = Path("logs")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    path /= Path(args.task + ".log")

    fileHandler = logging.FileHandler(path, mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    return logger