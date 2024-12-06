"""
This file contains utility functions/classes relating to

- UI / Logging
- File IO
- Constants that point to dirs
"""
import logging
import os


# === file io ===

def abspath(path: str) -> str:
    """returns the absolute path (instead of "~", env vars, or relative)"""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

def split_path(path: str) -> list[str]:
    """returns list of path components"""
    norm_path = os.path.normpath(path)
    return norm_path.split(os.sep)


def mkdir_if_not_exists(dirname: str) -> str:
    """makes a directory if it doesn't exist"""
    path_slug = split_path(dirname)
    curr_path = os.sep if dirname[0] == os.sep else ""
    for path_seg in path_slug:
        curr_path = os.path.join(curr_path, path_seg)
        if len(curr_path) > 0 and not os.path.exists(curr_path):
            os.mkdir(curr_path)
    return abspath(curr_path)

def list_dir(dir: str, *, only_dirs: bool = False) -> list[str]:
    children = sorted([os.path.join(dir, child) for child in os.listdir(dir)])
    if only_dirs:
        children = [child for child in children if os.path.isdir(child)]
    return children


# === constants / settings ===

DATA_DIR = abspath(os.environ.get("DATA_DIR", "~/data"))
YCB_DIR = abspath(os.environ.get("YCB_DIR", os.path.join(DATA_DIR, "ycb")))
SHAPENET_DIR = abspath(os.environ.get("SHAPENET_DIR", os.path.join(DATA_DIR, "ShapeNetCore.v2")))



# == ui / logging ===

class CustomFormatter(logging.Formatter):

    green = "\033[92m"
    cyan = "\033[96m"
    yellow = "\033[33m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\033[0m"

    FORMATS = {
        logging.DEBUG: f"%(asctime)s.%(msecs)03d {green}[DEBUG]{reset} %(message)s",
        logging.INFO: f"%(asctime)s.%(msecs)03d {cyan}[INFO]{reset} %(message)s",
        logging.WARNING: f"%(asctime)s.%(msecs)03d {yellow}[WARNING]{reset} %(message)s",
        logging.ERROR: f"%(asctime)s.%(msecs)03d {red}[ERROR]{reset} %(message)s",
        logging.CRITICAL: f"%(asctime)s.%(msecs)03d {bold_red}[CRITICAL]{reset} %(message)s",
    }

    FORMATS_NO_COLOR = {
        logging.DEBUG: f"%(asctime)s.%(msecs)03d [DEBUG] %(message)s",
        logging.INFO: f"%(asctime)s.%(msecs)03d [INFO] %(message)s",
        logging.WARNING: f"%(asctime)s.%(msecs)03d [WARNING] %(message)s",
        logging.ERROR: f"%(asctime)s.%(msecs)03d [ERROR] %(message)s",
        logging.CRITICAL: f"%(asctime)s.%(msecs)03d [CRITICAL] %(message)s",
    }

    def __init__(self, use_color: bool = True):
        self.formats = self.FORMATS if use_color else self.FORMATS_NO_COLOR

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%H:%0M:%S")
        return formatter.format(record)


def setup_logger(level = logging.INFO, file: str | None = None) -> logging.Logger:
    """Sets up a formatted logger that I like with the desired level and output file"""
    logger = logging.getLogger()
    logger.setLevel(level)

    # create console handler with a higher log level
    if file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename=file)

    handler.setLevel(level)

    handler.setFormatter(CustomFormatter(file is None))

    logger.addHandler(handler)
    return logger




