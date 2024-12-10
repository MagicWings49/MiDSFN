import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 2,
    "C": 3,
    "B": 4,
    "E": 5,
    "D": 6,
    "G": 7,
    "F": 8,
    "I": 9,
    "H": 10,
    "K": 11,
    "J": 12,
    "M": 13,
    "L": 14,
    "O": 15,
    "N": 16,
    "Q": 17,
    "P": 18,
    "S": 19,
    "R": 20,
    "U": 21,
    "T": 22,
    "W": 23,
    "V": 24,
    "Y": 25,
    "X": 26,
    "Z": 27,
    "a": 2,
    "c": 3,
    "b": 4,
    "e": 5,
    "d": 6,
    "g": 7,
    "f": 8,
    "i": 9,
    "h": 10,
    "k": 11,
    "j": 12,
    "m": 13,
    "l": 14,
    "o": 15,
    "n": 16,
    "q": 17,
    "p": 18,
    "s": 19,
    "r": 20,
    "u": 21,
    "t": 22,
    "w": 23,
    "v": 24,
    "y": 25,
    "x": 26,
    "z": 27,
    " ": 0,
    ".": 0,
    "(": 0,
    ")": 0,
    "-": 1,
    "/": 0,
    ":": 1,
    "0": 0,
    "1": 28,
    "2": 29,
    "3": 30,
    "4": 31,
    "5": 32,
    "6": 33,
    "7": 34,
    "8": 35,
    "9": 36,
    "'": 1,
    "聽": 0,
    "?": 0,
    "  ": 0,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, p, d_g, m_n, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), d_g, m_n, torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_microbe(sequence, max_length=37):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    sequence = sequence[:max_length]
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
