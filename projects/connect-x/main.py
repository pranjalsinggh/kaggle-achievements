"""Connect X agent using its bundled native search library. MIT license.

No NumPy, Numba, compilation, network access, or external data is required.
Upload main.py and libconnectx.so together as a zip archive.
"""
import ctypes
import os
import sys

_LIB = None
_BOARD_TYPE = ctypes.c_int * 42
_IMPORT_FOLDERS = list(sys.path)


def _load(configuration):
    global _LIB
    if _LIB is not None:
        return _LIB
    filename = 'connectx.dll' if os.name == 'nt' else 'libconnectx.so'
    raw = getattr(configuration, '__raw_path__', '/kaggle_simulations/agent/main.py')
    folders = [os.path.dirname(raw), '/kaggle_simulations/agent', os.getcwd()]
    folders.extend(_IMPORT_FOLDERS)
    for folder in folders:
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            _LIB = ctypes.CDLL(os.path.abspath(candidate))
            break
    if _LIB is None:
        raise RuntimeError('Bundled native search library not found')
    _LIB.cx_choose.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_double, ctypes.c_int]
    _LIB.cx_choose.restype = ctypes.c_int
    _LIB.cx_depth.restype = ctypes.c_int
    _LIB.cx_nodes.restype = ctypes.c_uint64
    _LIB.cx_winning.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    _LIB.cx_winning.restype = ctypes.c_uint64
    return _LIB


def choose(board, mark, budget=1.45, maximum_depth=42, configuration=None):
    if configuration is None:
        class Config:
            __raw_path__ = os.path.join(os.getcwd(), 'main.py')
        configuration=Config()
    lib = _load(configuration)
    return int(lib.cx_choose(_BOARD_TYPE(*board), int(mark), float(budget), int(maximum_depth)))


def agent(observation, configuration):
    if configuration.columns != 7 or configuration.rows != 6 or configuration.inarow != 4:
        return next((c for c in range(configuration.columns) if observation.board[c] == 0), 0)
    return choose(observation.board, observation.mark, configuration=configuration)
