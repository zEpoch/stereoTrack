import torch
import numpy as np
import random
import logging


def seed_all(seed=42):
    """
    Set random seed for reproducibility
    
    Parameters
    ----------
    seed : int
        Random seed value
        
    Examples
    --------
    >>> seed_all(42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def setup_logging(log_file=None):
    """
    Setup logging configuration
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

