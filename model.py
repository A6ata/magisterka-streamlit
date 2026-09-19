import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq, minimize_scalar


# ============================================================
# PARAMETRY
# ============================================================

@dataclass(frozen=True)
class Params:
    B: float
    delta: float
    eta: float
    Pbar: float
    m: float


# ============================================================
# WSPÓLNE FUNKCJE
# ============================================================

def x_from_tau(tau, p):
    """
    Stacjonarna relacja x(tau), wspólna dla obu modeli.
    """
    return (
        p.delta
        - (1.0 - p.eta) * p.B * (1.0 - np.sqrt(tau))
    ) / p.eta


def growth_rate(tau, p):
    """
    Długookresowe tempo wzrostu g*.
    """
    return (
        p.B * (1.0 - np.sqrt(tau)) - p.delta
    ) / p.eta


def u_from_tau(tau):
    """
    Udział kapitału przeznaczanego na redukcję emisji.
    """
    return np.sqrt(tau)


def emissions_from_tau(tau, p):
    """
    Emisje Z wynikające z optymalnego podziału kapitału.
    """
    return p.B * (tau ** (-0.5) - 1.0)


# ============================================================
# MODEL LINIOWY
# ============================================================

def assimilation_linear(E, p):
    return p.m * (p.Pbar - E)


def tau1_linear(p):
    """
    Dolna granica ekonomicznie dopuszczalnego przedziału tau.
    """
    return (p.B / (p.Pbar * p.m + p.B)) ** 2


def E_linear_from_tau(tau, p):
    return (
        p.Pbar
        - (p.B / p.m) * (tau ** (-0.5) - 1.0)
    )


def theta_linear(tau, p):
    E = E_linear_from_tau(tau, p)
    return E * tau


def gamma_linear(tau, p):
    x = x_from_tau(tau, p)
    return x / (p.m + x)


def D_linear(tau, p):
    """
    Równanie stacjonarności modelu liniowego:
    D(tau) = Theta(tau) - Gamma(tau).
    """
    return theta_linear(tau, p) - gamma_linear(tau, p)


def rhs_linear(state, p):
    """
    Prawa strona układu dynamicznego modelu liniowego.
    Kolejność zmiennych: (x, E, tau).
    """
    x, E, tau = state

    alpha = (1.0 - p.eta) / p.eta

    env = (
        p.m * (p.Pbar - E)
        - p.B * (tau ** (-0.5) - 1.0)
    )

    xdot = x * (
        alpha * env / E
        + alpha * p.B * (1.0 - np.sqrt(tau))
        - p.delta / p.eta
        + x
    )

    Edot = env

    taudot = tau * (
        p.m
        - x / (E * tau)
        + x
    )

    return np.array([xdot, Edot, taudot], dtype=float)


# ============================================================
# MODEL NIELINIOWY
# ============================================================

def assimilation_nonlinear(E, p):
    return p.m * E * (1.0 - E / p.Pbar)


def tau_nonlinear_from_E(E, p):
    A = assimilation_nonlinear(E, p)
    return (p.B / (p.B + A)) ** 2


def x_nonlinear_from_E(E, p):
    A = assimilation_nonlinear(E, p)

    return (
        p.delta
        - (1.0 - p.eta)
        * p.B * A / (p.B + A)
    ) / p.eta


def F_nonlinear(E, p):
    """
    Jednowymiarowe równanie wyznaczające punkty stacjonarne
    modelu nieliniowego.
    """
    tau = tau_nonlinear_from_E(E, p)
    x = x_nonlinear_from_E(E, p)

    return (
        -p.m * (1.0 - 2.0 * E / p.Pbar)
        - x / (E * tau)
        + x
    )


def rhs_nonlinear(state, p):
    """
    Prawa strona układu dynamicznego modelu nieliniowego.
    Kolejność zmiennych: (x, E, tau).
    """
    x, E, tau = state

    alpha = (1.0 - p.eta) / p.eta

    env = (
        p.m * E * (1.0 - E / p.Pbar)
        - p.B * (tau ** (-0.5) - 1.0)
    )

    xdot = x * (
        alpha * env / E
        + alpha * p.B * (1.0 - np.sqrt(tau))
        - p.delta / p.eta
        + x
    )

    Edot = env

    taudot = tau * (
        -p.m * (1.0 - 2.0 * E / p.Pbar)
        - x / (E * tau)
        + x
    )

    return np.array([xdot, Edot, taudot], dtype=float)


# ============================================================
# WYSZUKIWANIE PIERWIASTKÓW MODELU NIELINIOWEGO
# ============================================================

def _deduplicate_roots(roots, fun, tol=1e-7):
    """
    Łączy numerycznie identyczne pierwiastki i pozostawia
    rozwiązanie o najmniejszej wartości |F|.
    """
    roots = sorted(roots)

    groups = []

    for root in roots:
        if not groups or abs(root - groups[-1][-1]) > tol:
            groups.append([root])
        else:
            groups[-1].append(root)

    unique = []

    for group in groups:
        best_root = min(
            group,
            key=lambda z: abs(fun(z))
        )
        unique.append(best_root)

    return unique


def find_all_roots(
    fun,
    lower,
    upper,
    n_grid=5000,
    root_tol=1e-12,
    touch_tol=1e-8
):
    """
    Poszukuje wszystkich numerycznie wykrywalnych miejsc zerowych
    funkcji w zadanym przedziale.

    Etap 1:
    wykrywa zmiany znaku i stosuje metodę Brenta.

    Etap 2:
    analizuje lokalne minima |F|, co ogranicza ryzyko
    pominięcia pierwiastka stycznego bez zmiany znaku.
    """

    grid = np.linspace(lower, upper, n_grid)
    values = np.array([fun(z) for z in grid])

    roots = []

    # --------------------------------------------------------
    # 1. Pierwiastki związane ze zmianą znaku
    # --------------------------------------------------------

    for i in range(len(grid) - 1):

        x1 = grid[i]
        x2 = grid[i + 1]

        f1 = values[i]
        f2 = values[i + 1]

        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue

        if abs(f1) < root_tol:
            roots.append(x1)

        if f1 * f2 < 0:
            root = brentq(
                fun,
                x1,
                x2,
                xtol=root_tol,
                rtol=1e-12
            )
            roots.append(root)

    # --------------------------------------------------------
    # 2. Kontrola potencjalnych pierwiastków stycznych
    # --------------------------------------------------------

    abs_values = np.abs(values)

    for i in range(1, len(grid) - 1):

        if (
            np.isfinite(abs_values[i - 1])
            and np.isfinite(abs_values[i])
            and np.isfinite(abs_values[i + 1])
            and abs_values[i] < abs_values[i - 1]
            and abs_values[i] < abs_values[i + 1]
        ):

            result = minimize_scalar(
                lambda z: abs(fun(z)),
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded"
            )

            if result.success and result.fun < touch_tol:
                roots.append(result.x)

    return _deduplicate_roots(
    roots,
    fun
)


# ============================================================
# JACOBIAN NUMERYCZNY I STABILNOŚĆ
# ============================================================

def numerical_jacobian(fun, state, p, rel_step=1e-6):
    """
    Jacobian wyznaczany centralnym ilorazem różnicowym.
    """
    state = np.asarray(state, dtype=float)

    n = len(state)
    J = np.zeros((n, n))

    for j in range(n):

        h = rel_step * max(1.0, abs(state[j]))

        plus = state.copy()
        minus = state.copy()

        plus[j] += h
        minus[j] -= h

        J[:, j] = (
            fun(plus, p) - fun(minus, p)
        ) / (2.0 * h)

    return J


def classify_stability(eigenvalues, tol=1e-8):
    real_parts = np.real(eigenvalues)

    negative = np.sum(real_parts < -tol)
    positive = np.sum(real_parts > tol)
    zero = len(real_parts) - negative - positive

    if negative == 1 and positive == 2:
        return "saddle"

    if negative == 0 and positive == 3:
        return "unstable"

    if zero > 0:
        return "nonhyperbolic"

    return "other"


# ============================================================
# KONTROLA ROZWIĄZANIA
# ============================================================

def diagnostics(state, p, rhs):
    x, E, tau = state

    residuals = rhs(state, p)
    residual_max = np.max(np.abs(residuals))

    u = u_from_tau(tau)
    g = growth_rate(tau, p)

    admissible = (
        E > 0
        and E < p.Pbar
        and x > 0
        and tau > 0
        and tau < 1
        and u > 0
        and u < 1
    )

    positive_growth = g > 0

    transversality = (
        p.delta > (1.0 - p.eta) * g
    )

    residual_tol = 1e-8
    numerically_valid = residual_max < residual_tol

    J = numerical_jacobian(rhs, state, p)
    eigenvalues = np.linalg.eigvals(J)

    stability = classify_stability(eigenvalues)

    return {
        "residual_max": residual_max,
        "admissible": admissible,
        "positive_growth": positive_growth,
        "transversality": transversality,
        "eigenvalues": eigenvalues,
        "stability": stability,
        "numerically_valid": numerically_valid
    }


# ============================================================
# SOLVER MODELU LINIOWEGO
# ============================================================

def solve_linear(p, eps=1e-10):

    tau_min = tau1_linear(p)

    lower = tau_min + eps
    upper = 1.0 - eps

    f_lower = D_linear(lower, p)
    f_upper = D_linear(upper, p)

    if f_lower * f_upper >= 0:
        return None

    tau_star = brentq(
        lambda tau: D_linear(tau, p),
        lower,
        upper,
        xtol=1e-12,
        rtol=1e-12
    )

    E_star = E_linear_from_tau(tau_star, p)
    x_star = x_from_tau(tau_star, p)
    u_star = u_from_tau(tau_star)
    g_star = growth_rate(tau_star, p)

    state = np.array([x_star, E_star, tau_star])

    diag = diagnostics(state, p, rhs_linear)

    Z_star = emissions_from_tau(tau_star, p)
    A_star = assimilation_linear(E_star, p)

    result = {
        "model": "linear",
        "root": 1,
        "branch": "-",
        "E": E_star,
        "pollution": p.Pbar - E_star,
        "tau": tau_star,
        "u": u_star,
        "x": x_star,
        "g": g_star,
        "Z": Z_star,
        "A": A_star,
        **diag
    }

    return result


# ============================================================
# SOLVER MODELU NIELINIOWEGO
# ============================================================

def solve_nonlinear(p, n_grid=5000, eps=1e-8):
    """
    Przy Pbar = 1 ekonomicznie dopuszczalnych punktów
    poszukujemy na górnej gałęzi:
        E in (Pbar/2, Pbar).

    Funkcja zwraca listę wszystkich znalezionych punktów.
    """

    lower = p.Pbar / 2.0 + eps
    upper = p.Pbar - eps

    roots = find_all_roots(
        lambda E: F_nonlinear(E, p),
        lower,
        upper,
        n_grid=n_grid
    )

    results = []

    for number, E_star in enumerate(roots, start=1):

        tau_star = tau_nonlinear_from_E(E_star, p)
        x_star = x_nonlinear_from_E(E_star, p)
        u_star = u_from_tau(tau_star)
        g_star = growth_rate(tau_star, p)

        state = np.array([x_star, E_star, tau_star])

        diag = diagnostics(state, p, rhs_nonlinear)

        Z_star = emissions_from_tau(tau_star, p)
        A_star = assimilation_nonlinear(E_star, p)

        if E_star < p.Pbar / 2:
            branch = "lower"
        elif E_star > p.Pbar / 2:
            branch = "upper"
        else:
            branch = "middle"

        result = {
            "model": "nonlinear",
            "root": number,
            "branch": branch,
            "E": E_star,
            "pollution": p.Pbar - E_star,
            "tau": tau_star,
            "u": u_star,
            "x": x_star,
            "g": g_star,
            "Z": Z_star,
            "A": A_star,
            **diag
        }

        results.append(result)

    return results