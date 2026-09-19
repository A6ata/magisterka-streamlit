from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from model import (
    Params,
    assimilation_linear,
    assimilation_nonlinear,
    diagnostics,
    emissions_from_tau,
    find_all_roots,
    F_nonlinear,
    growth_rate,
    rhs_linear,
    rhs_nonlinear,
    solve_linear,
    solve_nonlinear,
    tau_nonlinear_from_E,
    u_from_tau,
    x_nonlinear_from_E,
)
from dynamics import stable_path_to_E0


BASE_B = 0.45
BASE_DELTA = 0.01
BASE_ETA = 1.5
BASE_PBAR = 1.0
BASE_M_L = 0.5
BASE_M_NL = 2.0

DATA_DIR = Path(__file__).parent / "data"


def base_params():
    common = dict(B=BASE_B, delta=BASE_DELTA, eta=BASE_ETA, Pbar=BASE_PBAR)
    return Params(**common, m=BASE_M_L), Params(**common, m=BASE_M_NL)


def base_solutions():
    p_l, p_nl = base_params()
    lin = solve_linear(p_l)
    nonlin = solve_nonlinear(p_nl, n_grid=5000)
    return lin, nonlin


def result_row(result: dict) -> dict:
    eig = np.asarray(result["eigenvalues"], dtype=complex)
    eig = eig[np.argsort(np.real(eig))]
    return {
        "model": "liniowy" if result["model"] == "linear" else "nieliniowy",
        "gałąź": result["branch"],
        "E*": float(result["E"]),
        "P*": float(result["pollution"]),
        "τ*": float(result["tau"]),
        "u*": float(result["u"]),
        "x*": float(result["x"]),
        "g*": float(result["g"]),
        "Z*": float(result["Z"]),
        "A(E*)": float(result["A"]),
        "stabilność": stability_pl(result["stability"]),
        "max |reszta|": float(result["residual_max"]),
        "dopuszczalny": bool(result["admissible"]),
        "dodatni wzrost": bool(result["positive_growth"]),
        "TVC": bool(result["transversality"]),
        "λ1": eig[0],
        "λ2": eig[1],
        "λ3": eig[2],
    }


def stability_pl(value: str) -> str:
    return {
        "saddle": "siodłowy",
        "unstable": "niestabilny",
        "nonhyperbolic": "niehiperboliczny",
        "other": "inny",
    }.get(value, value)


def base_table() -> pd.DataFrame:
    lin, nl_list = base_solutions()
    rows = []
    if lin is not None:
        rows.append(result_row(lin))
    rows.extend(result_row(r) for r in nl_list)
    return pd.DataFrame(rows)


def assimilation_dataframe(Pbar: float, m_l: float, m_nl: float, n: int = 500):
    E = np.linspace(0.0, Pbar, n)
    p_l = Params(BASE_B, BASE_DELTA, BASE_ETA, Pbar, m_l)
    p_nl = Params(BASE_B, BASE_DELTA, BASE_ETA, Pbar, m_nl)
    return pd.DataFrame(
        {
            "E": E,
            "Liniowa": [assimilation_linear(e, p_l) for e in E],
            "Nieliniowa": [assimilation_nonlinear(e, p_nl) for e in E],
        }
    )


def solve_pair(B: float, delta: float, eta: float, Pbar: float, m_l: float, m_nl: float):
    p_l = Params(B=B, delta=delta, eta=eta, Pbar=Pbar, m=m_l)
    p_nl = Params(B=B, delta=delta, eta=eta, Pbar=Pbar, m=m_nl)
    lin = solve_linear(p_l)
    # solve_nonlinear jest dokładnie procedurą używaną w pracy dla Pbar=1
    nl = solve_nonlinear(p_nl, n_grid=3000)
    return p_l, p_nl, lin, nl


def load_sensitivity(parameter: str) -> pd.DataFrame:
    mapping = {
        "η": "sensitivity_eta.csv",
        "δ": "sensitivity_delta.csv",
        "m": "sensitivity_m.csv",
    }
    return pd.read_csv(DATA_DIR / mapping[parameter])


def load_initial_condition_sensitivity() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "initial_condition_sensitivity.csv")


def stable_paths_for_E0(E0: float):
    p_l, p_nl = base_params()
    lin = solve_linear(p_l)
    nl_list = solve_nonlinear(p_nl, n_grid=5000)
    nl = nl_list[0] if nl_list else None

    path_l = diag_l = None
    path_nl = diag_nl = None

    if lin is not None:
        path_l, diag_l = stable_path_to_E0(
            result=lin,
            params=p_l,
            rhs=rhs_linear,
            model_name="linear",
            E0=float(E0),
            epsilon=1e-7,
            max_backward_time=200.0,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05,
            bound_eps=1e-8,
        )

    if nl is not None:
        path_nl, diag_nl = stable_path_to_E0(
            result=nl,
            params=p_nl,
            rhs=rhs_nonlinear,
            model_name="nonlinear",
            E0=float(E0),
            epsilon=1e-7,
            max_backward_time=200.0,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05,
            bound_eps=1e-8,
        )

    return (path_l, diag_l), (path_nl, diag_nl)


def load_multiplicity_region() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "multiplicity_region.csv")


def load_multiplicity_boundaries() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "multiplicity_boundaries.csv")


def load_off_manifold_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "off_manifold_summary.csv")


def load_off_manifold_trajectory(model: str, case: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"trajectory_{model}_{case}.csv")


# -------------------------------------------------------------------
# Eksploracyjny solver obu gałęzi. Nie zastępuje procedury bazowej z pracy;
# służy tylko do interaktywnej eksploracji szerszych parametrów.
# -------------------------------------------------------------------
def solve_nonlinear_all_branches(p: Params, n_grid: int = 4000, eps: float = 1e-7):
    lower = eps
    upper = p.Pbar - eps
    roots = find_all_roots(lambda E: F_nonlinear(E, p), lower, upper, n_grid=n_grid)
    results = []
    for number, E_star in enumerate(roots, start=1):
        tau_star = tau_nonlinear_from_E(E_star, p)
        x_star = x_nonlinear_from_E(E_star, p)
        state = np.array([x_star, E_star, tau_star], dtype=float)
        diag = diagnostics(state, p, rhs_nonlinear)
        branch = "lower" if E_star < p.Pbar / 2 else "upper" if E_star > p.Pbar / 2 else "middle"
        results.append(
            {
                "model": "nonlinear",
                "root": number,
                "branch": branch,
                "E": E_star,
                "pollution": p.Pbar - E_star,
                "tau": tau_star,
                "u": u_from_tau(tau_star),
                "x": x_star,
                "g": growth_rate(tau_star, p),
                "Z": emissions_from_tau(tau_star, p),
                "A": assimilation_nonlinear(E_star, p),
                **diag,
            }
        )
    return results


# Wielopunktowość: przekształcenie wielomianowe użyte w przesłanym skrypcie many.py.
def polynomial_stationary_points(Pbar: float, m: float, B: float = BASE_B, delta: float = BASE_DELTA, eta: float = BASE_ETA,
                                 imag_tol: float = 1e-7, residual_tol: float = 1e-7, merge_tol: float = 1e-7):
    from numpy.polynomial.polynomial import polymul

    s = m * Pbar
    D = np.array([B, s, -s], dtype=float)
    c = delta + (eta - 1.0) * B
    N = np.array([delta * B, c * s, -c * s], dtype=float)

    one_minus_2z = np.array([1.0, -2.0])
    z_poly = np.array([0.0, 1.0])
    term1 = polymul(polymul(one_minus_2z, D), z_poly)
    term1 *= -m * eta * Pbar * B**2

    D_squared = polymul(D, D)
    bracket = np.zeros(max(2, len(D_squared)))
    bracket[1] = Pbar * B**2
    bracket[: len(D_squared)] -= D_squared
    term2 = polymul(N, bracket)

    length = max(len(term1), len(term2))
    G = np.zeros(length)
    G[: len(term1)] += term1
    G[: len(term2)] += term2

    scale = np.max(np.abs(G))
    if scale == 0:
        return []
    G /= scale
    while len(G) > 1 and abs(G[-1]) < 1e-13:
        G = G[:-1]

    roots = np.roots(G[::-1])
    points = []
    p = Params(B=B, delta=delta, eta=eta, Pbar=Pbar, m=m)

    for root in roots:
        if abs(root.imag) > imag_tol:
            continue
        z = float(root.real)
        if not (merge_tol < z < 1.0 - merge_tol):
            continue
        E = Pbar * z
        A = m * E * (1.0 - E / Pbar)
        tau = (B / (B + A)) ** 2
        x = (delta - (1.0 - eta) * B * A / (B + A)) / eta
        F = -m * (1.0 - 2.0 * E / Pbar) - x / (E * tau) + x
        if abs(F) > residual_tol or x <= 0 or not (0 < tau < 1):
            continue

        state = np.array([x, E, tau], dtype=float)
        diag = diagnostics(state, p, rhs_nonlinear)
        points.append(
            {
                "E*": E,
                "P*": Pbar - E,
                "τ*": tau,
                "u*": np.sqrt(tau),
                "x*": x,
                "g*": growth_rate(tau, p),
                "Z*": emissions_from_tau(tau, p),
                "gałąź": "dolna" if E < Pbar / 2 else "górna",
                "stabilność": stability_pl(diag["stability"]),
                "max |reszta|": diag["residual_max"],
                "dodatni wzrost": diag["positive_growth"],
                "TVC": diag["transversality"],
            }
        )

    points.sort(key=lambda d: d["E*"])
    unique = []
    tolerance = merge_tol * max(1.0, Pbar)
    for point in points:
        if not unique or abs(point["E*"] - unique[-1]["E*"]) > tolerance:
            unique.append(point)
    return unique
