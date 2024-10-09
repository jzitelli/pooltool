import ctypes
import os.path as path
from ctypes import POINTER, Structure, c_double, c_int, cast
from logging import getLogger

_logger = getLogger(__name__)
import numpy as np


class c_double_complex(Structure):
    _fields_ = [("real", c_double), ("imag", c_double)]


c_double_p = POINTER(c_double)
c_double_complex_p = POINTER(c_double_complex)


PIx2 = np.pi * 2
CUBE_ROOTS_OF_1 = np.exp(1j * PIx2 / 3 * np.arange(3))


_ZERO_TOLERANCE = 1e-12
_ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
_IMAG_TOLERANCE = 1e-12
_IMAG_TOLERANCE_SQRD = _IMAG_TOLERANCE**2


try:
    _lib = ctypes.cdll.LoadLibrary(
        path.join(path.dirname(path.abspath(__file__)), "_poly_solvers.so")
    )
except:
    _lib = ctypes.cdll.LoadLibrary(
        path.join(path.dirname(path.abspath(__file__)), "poly_solvers.dll")
    )
_lib.quartic_solve.argtypes = (c_double_p, c_double_complex_p)

_lib.find_min_quartic_root_in_real_interval.argtypes = (c_double_p, c_double, c_double)
_lib.find_min_quartic_root_in_real_interval.restype = c_double

_lib.find_collision_time.argtypes = (
    c_double_p,
    c_double_p,
    c_double,
    c_double,
    c_double,
)
_lib.find_collision_time.restype = c_double

_lib.find_corner_collision_time.argtypes = (c_double_p, c_double_p, c_double, c_double)
_lib.find_corner_collision_time.restype = c_double

_lib.sort_complex_conjugate_pairs.argtypes = [c_double_complex_p]
_lib.sort_complex_conjugate_pairs.restype = c_int


def find_collision_time(a_i, a_j, R, t0, t1):
    t = _lib.find_collision_time(
        cast(a_i.ctypes.data, c_double_p), cast(a_j.ctypes.data, c_double_p), R, t0, t1
    )
    if t < t1:
        return t


def find_corner_collision_time(r_c, a, R, tau_min):
    tau = _lib.find_corner_collision_time(
        cast(r_c.ctypes.data, c_double_p), cast(a.ctypes.data, c_double_p), R, tau_min
    )
    if tau < tau_min:
        return tau


def find_min_quartic_root_in_real_interval(p, t0, t1):
    t = _lib.find_min_quartic_root_in_real_interval(
        cast(p.ctypes.data, c_double_p), t0, t1
    )
    if t < t1:
        return t


# from numba import njit
# from numpy.typing import NDArray


# @njit
def quartic_solve(p):
    _lib.quartic_solve(cast(p.ctypes.data, c_double_p), quartic_solve.outp)
    return quartic_solve.out


quartic_solve.out = np.zeros(4, dtype=np.complex128)
quartic_solve.outp = cast(quartic_solve.out.ctypes.data, c_double_complex_p)


def sort_complex_conjugate_pairs(roots):
    return _lib.sort_complex_conjugate_pairs(
        cast(roots.ctypes.data, c_double_complex_p)
    )
