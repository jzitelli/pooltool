from typing import Callable, Dict

import numpy as np
from numba import jit
from sympy import Eq, solve, symbols

import pooltool.constants as const
from pooltool.utils.strenum import StrEnum, auto


@jit(nopython=True, cache=const.numba_cache)
def roots_quadratic(a, b, c):
    """Solve a quadratic equation At^2 + Bt + C = 0 (just-in-time compiled)"""
    if a == 0:
        u = -c / b
        return u, u
    bp = b / 2
    delta = bp * bp - a * c
    u1 = (-bp - delta**0.5) / a
    u2 = -u1 - b / a
    return u1, u2


def quartic_truth(a_val, b_val, c_val, d_val, e_val, digits=40):
    x, a, b, c, d, e = symbols("x a b c d e")
    general_solution = solve(a * x**4 + b * x**3 + c * x**2 + d * x + e, x)
    return [
        sol.evalf(digits, subs={a: a_val, b: b_val, c: c_val, d: d_val, e: e_val})
        for sol in general_solution
    ]


@jit(nopython=True, cache=const.numba_cache)
def quartic_analytic(p):
    """Analytic solutions to the quartic polynomial

    This function was created with the help of sympy.

    To start, I solved the general quartic polynomial roots:

    >>> from sympy import symbols, Eq, solve
    >>> x, a, b, c, d, e = symbols('x a b c d e')
    >>> general_solution = solve(a*x**4 + b*x**3 + c*x**2 + d*x + e, x)

    This yields 4 expressions, one for each root. Each expression is piecewise
    conditional, where if the following equality is true, the first expression is used,
    and otherwise the second expression is used.

    >>> general_solution[0].args[0][1]
    Eq(e/a - b*d/(4*a**2) + c**2/(12*a**2), 0)

    So in total there are 8 expressions, 2 for each root, and the expression used for
    each root is determined based on whether the above equality holds true. These
    expressions are huge, so to better digest them, I used the following common
    subexpression elimination:

    >>> from sympy import cse
    >>> cse(
    >>>     [
    >>>         general_solution[0].args[0][0],
    >>>         general_solution[0].args[1][0],
    >>>         general_solution[1].args[0][0],
    >>>         general_solution[1].args[1][0],
    >>>         general_solution[2].args[0][0],
    >>>         general_solution[2].args[1][0],
    >>>         general_solution[3].args[0][0],
    >>>         general_solution[3].args[1][0],
    >>>     ]
    >>> )

    Then I used a vim macro to convert these subexpressions into the lines of python
    code you see below.

    Note:

    This will never be ready for primetime and here is why. The formula for the roots of
    a quartic polynomial, just like the quadratic formula we learn in school, gives an
    exact answer in terms of the coefficients. However, to compute this formula on a
    computer, we have to perform a sequence of arithmetic operations (additions,
    multiplications, divisions, square roots, etc.). Each of these operations introduces
    a small amount of round-off error due to the finite precision of floating-point
    arithmetic.

    The crux of the issue is that the formulas for the roots of a cubic or quartic
    polynomial involve operations like subtracting two nearly equal numbers (leading to
    loss of significance, aka catastrophic cancellation) or taking the square root of a
    small negative number that should have been zero (resulting in a spurious imaginary
    part).  These errors can get amplified in a way that makes the computed roots very
    inaccurate, especially for "ill-conditioned" polynomials where the roots are very
    sensitive to the coefficients.

    On the other hand, numerical algorithms like the one used in numpy's roots function
    are designed to minimize these round-off errors and make the result as accurate as
    possible, even if the polynomial is ill-conditioned. This can make them more
    accurate than the direct formula in practice, even though they are only
    approximations.

    For example, the companion matrix method used by numpy computes the roots as the
    eigenvalues of a matrix. This avoids the problematic operations in the direct
    formula and makes the result more accurate.

    In other words, the formulas are exact in the realm of pure mathematics where we can
    do arithmetic with infinite precision. But when we bring them into the world of
    floating-point arithmetic with its inherent round-off errors, they can behave badly.
    """

    cp = p.astype(np.complex128)
    a, b, c, d, e = cp.T

    x0 = 1 / a
    x1 = c * x0
    x2 = a ** (-2)
    x3 = b**2
    x4 = x2 * x3
    x5 = x1 - 3 * x4 / 8
    x6 = x5**3
    x7 = d * x0
    x8 = b * x2
    x9 = c * x8
    x10 = a ** (-3)
    x11 = b**3 * x10
    x12 = (x11 / 8 + x7 - x9 / 2) ** 2
    x13 = -d * x8 / 4 + e * x0
    x14 = c * x10 * x3 / 16 + x13 - 3 * b**4 / (256 * a**4)
    x15 = -x12 / 8 + x14 * x5 / 3 - x6 / 108
    x16 = 2 * x15 ** (1 / 3)
    x17 = x11 / 4 + 2 * x7 - x9
    x18 = 2 * x1 / 3 - x2 * x3 / 4
    x19 = np.sqrt(-x16 - x18)
    x20 = x17 / x19
    x21 = 4 * x1 / 3
    x22 = -x21 + x4 / 2
    x23 = np.sqrt(x16 + x20 + x22) / 2
    x24 = x19 / 2
    x25 = b * x0 / 4
    x26 = x24 + x25
    x27 = -(c**2) * x2 / 12 - x13
    x28 = (
        x12 / 16 - x14 * x5 / 6 + x6 / 216 + np.sqrt(x15**2 / 4 + x27**3 / 27)
    ) ** (1 / 3)
    x29 = 2 * x28
    x30 = 2 * x27 / (3 * x28)
    x31 = -x29 + x30
    x32 = np.sqrt(-x18 - x31)
    x32[x32 == 0] = const.EPS
    x33 = x17 / x32
    x34 = np.sqrt(x22 + x31 + x33) / 2
    x35 = x32 / 2
    x36 = x25 + x35
    x37 = -x2 * x3 / 2 + x21
    x38 = np.sqrt(x16 - x20 - x37) / 2
    x39 = np.sqrt(-x29 + x30 - x33 - x37) / 2
    x40 = -x25

    roots = np.zeros((p.shape[0], 4), dtype=np.complex128)
    cond = np.abs(e / a - b * d / (4 * a**2) + c**2 / (12 * a**2)) < const.EPS

    for i in range(len(cond)):
        if cond[i]:
            roots[i, 0] = -x23[i] - x26[i]
            roots[i, 1] = x23[i] - x26[i]
            roots[i, 2] = x24[i] - x25[i] - x38[i]
            roots[i, 3] = x24[i] + x38[i] + x40[i]
        else:
            roots[i, 0] = -x34[i] - x36[i]
            roots[i, 1] = x34[i] - x36[i]
            roots[i, 2] = -x25[i] + x35[i] - x39[i]
            roots[i, 3] = x35[i] + x39[i] + x40[i]

    return roots


def roots_numerical(p):
    """Solve multiple polynomial equations

    This is a vectorized implementation of numpy.roots that can solve multiple
    polynomials in a vectorized fashion. The solution is taken from this wonderful
    stackoverflow answer: https://stackoverflow.com/a/35853977

    Parameters
    ==========
    p : array
        A mxn array of polynomial coefficients, where m is the number of equations and
        n-1 is the order of the polynomial. If n is 5 (4th order polynomial), the
        columns are in the order a, b, c, d, e, where these coefficients make up the
        polynomial equation at^4 + bt^3 + ct^2 + dt + e = 0

    Notes
    =====
    - This function is not amenable to numbaization (0.54.1). There are a couple of
      hurdles to address. p[...,None,0] needs to be refactored since None/np.newaxis
      cause compile error. But even bigger an issue is that np.linalg.eigvals is only
      supported for 2D arrays, but the strategy here is to pass np.lingalg.eigvals a
      vectorized 3D array.
    """
    n = p.shape[-1]
    A = np.zeros(p.shape[:1] + (n - 1, n - 1), np.float64)
    A[..., 1:, :-1] = np.eye(n - 2)
    A[..., 0, :] = -p[..., 1:] / p[..., None, 0]
    return np.linalg.eigvals(A)


class QuarticSolver(StrEnum):
    ANALYTIC = auto()
    NUMERIC = auto()


_routine: Dict[QuarticSolver, Callable] = {
    QuarticSolver.NUMERIC: roots_numerical,
    QuarticSolver.ANALYTIC: quartic_analytic,
}


def min_real_root(p, solver: QuarticSolver = QuarticSolver.NUMERIC, tol=1e-9):
    """Given an array of polynomial coefficients, find the minimum real root

    Parameters
    ==========
    p:
        A mxn array of polynomial coefficients, where m is the number of equations and
        n-1 is the order of the polynomial. If n is 5 (4th order polynomial), the
        columns are in the order a, b, c, d, e, where these coefficients make up the
        polynomial equation at^4 + bt^3 + ct^2 + dt + e = 0
    solver:
        How should the roots be calculated? Choose one of {"numeric",
        "quartic_analytic"}. "numeric" works for n-degree polynomials,
        "quartic_analytic" works for quartics and is fast.
    tol:
        Roots are real if their imaginary components are less than than tol.

    Returns
    =======
    output : (time, index)
        `time` is the minimum real root from the set of polynomials, and `index`
        specifies the index of the responsible polynomial. i.e. the polynomial with the
        root `time` is p[index]
    """
    # Get the roots for the polynomials
    assert QuarticSolver(solver)
    times = _routine[solver](p)

    # If the root has a nonzero imaginary component, set to infinity
    # If the root has a nonpositive real component, set to infinity
    times[(abs(times.imag) > tol) | (times.real <= tol)] = np.inf

    # now find the minimum time and the index of the responsible polynomial
    times = np.min(times.real, axis=1)

    return times.min(), times.argmin()
