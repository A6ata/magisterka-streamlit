import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp

from model import numerical_jacobian


# ============================================================
# DOMYSLNE USTAWIENIA NUMERYCZNE
# ============================================================

DEFAULT_EPSILON = 1e-7
DEFAULT_MAX_BACKWARD_TIME = 200.0
DEFAULT_FORWARD_TIME = 50.0
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12
DEFAULT_MAX_STEP = 0.05
DEFAULT_BOUND_EPS = 1e-8


# ============================================================
# POMOCNICZE NARZEDZIA
# ============================================================

def state_star_from_result(result):
    """Zwraca wektor stanu stacjonarnego (x*, E*, tau*)."""
    return np.array(
        [
            float(result["x"]),
            float(result["E"]),
            float(result["tau"]),
        ],
        dtype=float,
    )


def stable_eigenpair(rhs, state_star, params, tol=1e-8):
    """
    Wyznacza jedyna stabilna wartosc wlasna Jacobiego oraz
    odpowiadajacy jej znormalizowany wektor wlasny.
    """
    J = numerical_jacobian(rhs, state_star, params)
    eigenvalues, eigenvectors = np.linalg.eig(J)

    stable_indices = np.where(np.real(eigenvalues) < -tol)[0]

    if len(stable_indices) != 1:
        raise RuntimeError(
            "Oczekiwano dokladnie jednej stabilnej wartosci wlasnej, "
            f"znaleziono: {len(stable_indices)}."
        )

    idx = stable_indices[0]
    eigenvalue = eigenvalues[idx]
    eigenvector = eigenvectors[:, idx]

    if abs(np.imag(eigenvalue)) > 1e-8:
        raise RuntimeError(
            "Stabilna wartosc wlasna ma istotna czesc urojona."
        )

    if np.max(np.abs(np.imag(eigenvector))) > 1e-8:
        raise RuntimeError(
            "Stabilny wektor wlasny ma istotna czesc urojona."
        )

    eigenvalue = float(np.real(eigenvalue))
    eigenvector = np.real(eigenvector)
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    return eigenvalue, eigenvector, J


def orient_stable_vector(eigenvector, E_star, E_target):
    """
    Ustawia znak stabilnego wektora wlasnego tak, aby podczas
    integracji wstecz skladowa E poruszala sie od E* w strone E_target.
    """
    v = np.asarray(eigenvector, dtype=float).copy()

    desired_direction = np.sign(E_target - E_star)
    if desired_direction == 0:
        return v

    if abs(v[1]) < 1e-14:
        raise RuntimeError(
            "Skladowa E stabilnego wektora wlasnego jest numerycznie zerowa."
        )

    if np.sign(v[1]) != desired_direction:
        v = -v

    return v


def is_state_admissible(state, params, bound_eps=0.0):
    """Sprawdza ekonomiczna dopuszczalnosc pojedynczego stanu."""
    x, E, tau = map(float, state)
    return bool(
        x > bound_eps
        and bound_eps < E < params.Pbar - bound_eps
        and bound_eps < tau < 1.0 - bound_eps
    )


def _boundary_events(params, bound_eps):
    """Tworzy zdarzenia zatrzymujace integracje na granicach dziedziny."""

    def event_E_lower(t, y):
        return y[1] - bound_eps

    def event_E_upper(t, y):
        return params.Pbar - bound_eps - y[1]

    def event_x_lower(t, y):
        return y[0] - bound_eps

    def event_tau_lower(t, y):
        return y[2] - bound_eps

    def event_tau_upper(t, y):
        return 1.0 - bound_eps - y[2]

    events = [
        event_E_lower,
        event_E_upper,
        event_x_lower,
        event_tau_lower,
        event_tau_upper,
    ]

    names = [
        "E_lower_boundary",
        "E_upper_boundary",
        "x_lower_boundary",
        "tau_lower_boundary",
        "tau_upper_boundary",
    ]

    for event in events:
        event.terminal = True
        event.direction = 0

    return events, names


def _triggered_event_name(solution, event_names):
    for name, event_times in zip(event_names, solution.t_events):
        if len(event_times) > 0:
            return name
    return None


def _path_diagnostics(states, state_star, params, E0, bound_eps=0.0):
    """Wspolne diagnostyki dla trajektorii zapisanej jako macierz 3 x n."""
    x = states[0]
    E = states[1]
    tau = states[2]

    if np.any(tau <= 0):
        return None

    u = np.sqrt(tau)
    distance = np.linalg.norm(states.T - state_star, axis=1)

    admissible_path = bool(
        np.all(x > bound_eps)
        and np.all((E > bound_eps) & (E < params.Pbar - bound_eps))
        and np.all((tau > bound_eps) & (tau < 1.0 - bound_eps))
    )

    distance_monotone = bool(np.all(np.diff(distance) <= 1e-9))

    E_star = float(state_star[1])
    if E0 < E_star:
        E_monotone = bool(np.all(np.diff(E) >= -1e-9))
    elif E0 > E_star:
        E_monotone = bool(np.all(np.diff(E) <= 1e-9))
    else:
        E_monotone = True

    midpoint = params.Pbar / 2.0
    crosses_midpoint = bool(np.min(E) < midpoint < np.max(E))

    return {
        "x": x,
        "E": E,
        "tau": tau,
        "u": u,
        "distance": distance,
        "admissible_path": admissible_path,
        "distance_monotone": distance_monotone,
        "E_monotone": E_monotone,
        "crosses_midpoint": crosses_midpoint,
        "min_E": float(np.min(E)),
        "max_E": float(np.max(E)),
        "min_x": float(np.min(x)),
        "max_x": float(np.max(x)),
        "min_tau": float(np.min(tau)),
        "max_tau": float(np.max(tau)),
        "min_u": float(np.min(u)),
        "max_u": float(np.max(u)),
    }


# ============================================================
# STABILNA SCIEZKA DLA ZADANEGO E0
# ============================================================

def stable_path_to_E0(
    result,
    params,
    rhs,
    model_name,
    E0,
    epsilon=DEFAULT_EPSILON,
    max_backward_time=DEFAULT_MAX_BACKWARD_TIME,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    max_step=DEFAULT_MAX_STEP,
    bound_eps=DEFAULT_BOUND_EPS,
):
    """
    Dla zadanego poczatkowego stanu srodowiska E0 wyznacza x0 i tau0
    lezace na stabilnej rozmaitosci punktu stacjonarnego.

    Metoda:
    1. wyznacza stabilny wektor wlasny Jacobiego w punkcie stacjonarnym,
    2. startuje w odleglosci epsilon od punktu stacjonarnego,
    3. integruje pelny nieliniowy uklad wstecz,
    4. zatrzymuje integracje po osiagnieciu E = E0,
    5. odwraca kolejnosc, otrzymujac sciezke od E0 do rownowagi.
    """
    E0 = float(E0)
    state_star = state_star_from_result(result)
    x_star, E_star, tau_star = state_star

    if not (bound_eps < E0 < params.Pbar - bound_eps):
        return None, {
            "model": model_name,
            "E0_target": E0,
            "success": False,
            "status": "E0_outside_domain",
            "epsilon": epsilon,
        }

    lambda_s, v_s, J = stable_eigenpair(rhs, state_star, params)
    v_s = orient_stable_vector(v_s, E_star, E0)

    if abs(E0 - E_star) < 1e-10:
        u_star = float(np.sqrt(tau_star))
        df = pd.DataFrame(
            {
                "time": [0.0],
                "x": [x_star],
                "E": [E_star],
                "tau": [tau_star],
                "u": [u_star],
                "distance_to_steady_state": [0.0],
            }
        )
        diagnostics = {
            "model": model_name,
            "E0_target": E0,
            "success": True,
            "status": "steady_state",
            "epsilon": epsilon,
            "lambda_stable": lambda_s,
            "E_star": E_star,
            "x_star": x_star,
            "tau_star": tau_star,
            "u_star": u_star,
            "E_0": E_star,
            "x_0": x_star,
            "tau_0": tau_star,
            "u_0": u_star,
            "time_horizon": 0.0,
            "initial_distance": 0.0,
            "terminal_distance": 0.0,
            "admissible_path": True,
            "distance_monotone": True,
            "E_monotone": True,
            "crosses_midpoint": False,
            "min_E": E_star,
            "max_E": E_star,
            "min_x": x_star,
            "max_x": x_star,
            "min_tau": tau_star,
            "max_tau": tau_star,
            "min_u": u_star,
            "max_u": u_star,
        }
        return df, diagnostics

    state_near = state_star + epsilon * v_s

    if not is_state_admissible(state_near, params, bound_eps=0.0):
        return None, {
            "model": model_name,
            "E0_target": E0,
            "success": False,
            "status": "epsilon_start_outside_domain",
            "epsilon": epsilon,
            "lambda_stable": lambda_s,
            "E_star": E_star,
        }

    def ode(t, y):
        return rhs(y, params)

    def event_target_E(t, y):
        return y[1] - E0

    event_target_E.terminal = True
    event_target_E.direction = 0

    boundary_events, boundary_names = _boundary_events(params, bound_eps)
    events = [event_target_E] + boundary_events
    event_names = ["target_reached"] + boundary_names

    try:
        solution = solve_ivp(
            ode,
            t_span=(0.0, -float(max_backward_time)),
            y0=state_near,
            events=events,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
    except Exception as exc:
        return None, {
            "model": model_name,
            "E0_target": E0,
            "success": False,
            "status": "integration_exception: " + str(exc),
            "epsilon": epsilon,
            "lambda_stable": lambda_s,
            "E_star": E_star,
        }

    if not solution.success:
        return None, {
            "model": model_name,
            "E0_target": E0,
            "success": False,
            "status": "solver_failed: " + solution.message,
            "epsilon": epsilon,
            "lambda_stable": lambda_s,
            "E_star": E_star,
        }

    triggered_event = _triggered_event_name(solution, event_names)

    if triggered_event != "target_reached":
        return None, {
            "model": model_name,
            "E0_target": E0,
            "success": False,
            "status": triggered_event or "target_not_reached",
            "epsilon": epsilon,
            "lambda_stable": lambda_s,
            "E_star": E_star,
            "x_star": x_star,
            "tau_star": tau_star,
            "u_star": float(np.sqrt(tau_star)),
        }

    t_reversed = solution.t[::-1]
    states = solution.y[:, ::-1]
    time = t_reversed - t_reversed[0]

    path_info = _path_diagnostics(
        states,
        state_star,
        params,
        E0=E0,
        bound_eps=0.0,
    )

    if path_info is None:
        return None, {
            "model": model_name,
            "E0_target": E0,
            "success": False,
            "status": "tau_nonpositive",
            "epsilon": epsilon,
            "lambda_stable": lambda_s,
            "E_star": E_star,
        }

    df = pd.DataFrame(
        {
            "time": time,
            "x": path_info["x"],
            "E": path_info["E"],
            "tau": path_info["tau"],
            "u": path_info["u"],
            "distance_to_steady_state": path_info["distance"],
        }
    )

    diagnostics = {
        "model": model_name,
        "E0_target": E0,
        "success": True,
        "status": "target_reached",
        "epsilon": epsilon,
        "lambda_stable": lambda_s,
        "E_star": E_star,
        "x_star": x_star,
        "tau_star": tau_star,
        "u_star": float(np.sqrt(tau_star)),
        "E_0": float(path_info["E"][0]),
        "x_0": float(path_info["x"][0]),
        "tau_0": float(path_info["tau"][0]),
        "u_0": float(path_info["u"][0]),
        # Horyzont oznacza czas dojscia do epsilon-otoczenia,
        # nie czas osiagniecia dokladnej rownowagi.
        "time_horizon": float(time[-1]),
        "initial_distance": float(path_info["distance"][0]),
        "terminal_distance": float(path_info["distance"][-1]),
        "admissible_path": path_info["admissible_path"],
        "distance_monotone": path_info["distance_monotone"],
        "E_monotone": path_info["E_monotone"],
        "crosses_midpoint": path_info["crosses_midpoint"],
        "min_E": path_info["min_E"],
        "max_E": path_info["max_E"],
        "min_x": path_info["min_x"],
        "max_x": path_info["max_x"],
        "min_tau": path_info["min_tau"],
        "max_tau": path_info["max_tau"],
        "min_u": path_info["min_u"],
        "max_u": path_info["max_u"],
    }

    return df, diagnostics


# ============================================================
# DOWOLNY PUNKT POCZATKOWY - DO STREAMLITA / DIAGNOSTYKI
# ============================================================

def simulate_from_initial_state(
    params,
    rhs,
    model_name,
    x0,
    E0,
    tau0,
    t_max=DEFAULT_FORWARD_TIME,
    state_star=None,
    t_eval=None,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    max_step=DEFAULT_MAX_STEP,
    bound_eps=DEFAULT_BOUND_EPS,
):
    """
    Integruje pelny uklad do przodu z dowolnego dopuszczalnego punktu
    (x0, E0, tau0). Funkcja nie zaklada, ze punkt lezy na stabilnej
    rozmaitosci. Jest przeznaczona m.in. do eksperymentow off-manifold
    i przyszlego interfejsu Streamlit.
    """
    initial_state = np.array([x0, E0, tau0], dtype=float)

    if not is_state_admissible(initial_state, params, bound_eps=bound_eps):
        return None, {
            "model": model_name,
            "success": False,
            "status": "initial_state_outside_domain",
            "x_0": float(x0),
            "E_0": float(E0),
            "tau_0": float(tau0),
        }

    if state_star is not None:
        state_star = np.asarray(state_star, dtype=float)

    def ode(t, y):
        return rhs(y, params)

    events, event_names = _boundary_events(params, bound_eps)

    try:
        solution = solve_ivp(
            ode,
            t_span=(0.0, float(t_max)),
            y0=initial_state,
            events=events,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=t_eval,
        )
    except Exception as exc:
        return None, {
            "model": model_name,
            "success": False,
            "status": "integration_exception: " + str(exc),
            "x_0": float(x0),
            "E_0": float(E0),
            "tau_0": float(tau0),
        }

    triggered_event = _triggered_event_name(solution, event_names)
    reached_horizon = bool(
        solution.success
        and triggered_event is None
        and solution.t.size > 0
        and abs(solution.t[-1] - float(t_max)) <= max(1e-8, 1e-8 * abs(float(t_max)))
    )

    states = solution.y
    x = states[0]
    E = states[1]
    tau = states[2]

    if np.any(tau <= 0):
        u = np.full_like(tau, np.nan, dtype=float)
    else:
        u = np.sqrt(tau)

    if state_star is None:
        distance = np.full(solution.t.shape, np.nan, dtype=float)
        terminal_distance = np.nan
    else:
        distance = np.linalg.norm(states.T - state_star, axis=1)
        terminal_distance = float(distance[-1]) if len(distance) else np.nan

    df = pd.DataFrame(
        {
            "time": solution.t,
            "x": x,
            "E": E,
            "tau": tau,
            "u": u,
            "distance_to_steady_state": distance,
        }
    )

    admissible_path = bool(
        np.all(x > 0)
        and np.all((E > 0) & (E < params.Pbar))
        and np.all((tau > 0) & (tau < 1))
    )

    diagnostics = {
        "model": model_name,
        "success": bool(solution.success),
        "status": (
            triggered_event
            if triggered_event is not None
            else ("horizon_reached" if reached_horizon else solution.message)
        ),
        "reached_horizon": reached_horizon,
        "exit_event": triggered_event,
        "x_0": float(x0),
        "E_0": float(E0),
        "tau_0": float(tau0),
        "u_0": float(np.sqrt(tau0)) if tau0 > 0 else np.nan,
        "time_end": float(solution.t[-1]) if solution.t.size else 0.0,
        "admissible_path": admissible_path,
        "terminal_distance": terminal_distance,
        "min_E": float(np.min(E)) if len(E) else np.nan,
        "max_E": float(np.max(E)) if len(E) else np.nan,
        "min_x": float(np.min(x)) if len(x) else np.nan,
        "max_x": float(np.max(x)) if len(x) else np.nan,
        "min_tau": float(np.min(tau)) if len(tau) else np.nan,
        "max_tau": float(np.max(tau)) if len(tau) else np.nan,
    }

    return df, diagnostics


# ============================================================
# NIEZALEZNA WALIDACJA FORWARD
# ============================================================

def validate_forward_path(
    path,
    result,
    params,
    rhs,
    model_name,
    forward_tol=1e-6,
    rtol=1e-12,
    atol=1e-14,
    max_step=0.01,
    bound_eps=DEFAULT_BOUND_EPS,
):
    """
    Niezaleznie integruje do przodu od punktu (x0, E0, tau0)
    znalezionego na stabilnej sciezce i sprawdza, czy po takim samym
    horyzoncie trajektoria wraca do epsilon-otoczenia punktu stacjonarnego.

    Ze wzgledu na siodlowy charakter rownowagi test jest diagnostyka
    numeryczna dla skonczonego horyzontu, a nie globalnym dowodem stabilnosci.
    """
    if path is None or len(path) == 0:
        return {
            "model": model_name,
            "forward_success": False,
            "forward_status": "missing_reference_path",
        }

    state_star = state_star_from_result(result)

    if len(path) == 1:
        return {
            "model": model_name,
            "E0_target": float(path.iloc[0]["E"]),
            "forward_success": True,
            "forward_status": "steady_state",
            "forward_reached_horizon": True,
            "forward_terminal_distance": 0.0,
            "forward_max_reference_error": 0.0,
            "forward_pass": True,
            "forward_tol": forward_tol,
        }

    x0 = float(path.iloc[0]["x"])
    E0 = float(path.iloc[0]["E"])
    tau0 = float(path.iloc[0]["tau"])
    time_grid = path["time"].to_numpy(dtype=float)
    t_max = float(time_grid[-1])

    forward_path, forward_diag = simulate_from_initial_state(
        params=params,
        rhs=rhs,
        model_name=model_name,
        x0=x0,
        E0=E0,
        tau0=tau0,
        t_max=t_max,
        state_star=state_star,
        t_eval=time_grid,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        bound_eps=bound_eps,
    )

    if forward_path is None:
        return {
            "model": model_name,
            "E0_target": E0,
            "forward_success": False,
            "forward_status": forward_diag.get("status", "forward_failed"),
            "forward_reached_horizon": False,
            "forward_terminal_distance": np.nan,
            "forward_max_reference_error": np.nan,
            "forward_pass": False,
            "forward_tol": forward_tol,
        }

    same_length = len(forward_path) == len(path)
    if same_length:
        ref_states = path[["x", "E", "tau"]].to_numpy(dtype=float)
        fwd_states = forward_path[["x", "E", "tau"]].to_numpy(dtype=float)
        reference_error = np.linalg.norm(fwd_states - ref_states, axis=1)
        max_reference_error = float(np.max(reference_error))
    else:
        max_reference_error = np.nan

    final_distance = float(forward_diag.get("terminal_distance", np.nan))
    reached_horizon = bool(forward_diag.get("reached_horizon", False))

    forward_pass = bool(
        forward_diag.get("success", False)
        and reached_horizon
        and np.isfinite(final_distance)
        and final_distance <= forward_tol
    )

    return {
        "model": model_name,
        "E0_target": E0,
        "forward_success": bool(forward_diag.get("success", False)),
        "forward_status": forward_diag.get("status"),
        "forward_reached_horizon": reached_horizon,
        "forward_terminal_distance": final_distance,
        "forward_max_reference_error": max_reference_error,
        "forward_pass": forward_pass,
        "forward_tol": float(forward_tol),
    }


# ============================================================
# ODPORNOSC WZGLEDEM EPSILON
# ============================================================

def epsilon_robustness(
    result,
    params,
    rhs,
    model_name,
    E0_values,
    epsilon_values=(1e-6, 1e-7, 1e-8),
    **stable_path_kwargs,
):
    """
    Powtarza konstrukcje stabilnej sciezki dla kilku wartosci epsilon.
    Do oceny odpornosci nalezy porownywac przede wszystkim x0 i tau0.
    Horyzont T_epsilon z definicji zmienia sie wraz z epsilon.
    """
    rows = []

    for E0 in E0_values:
        for epsilon in epsilon_values:
            path, diagnostics = stable_path_to_E0(
                result=result,
                params=params,
                rhs=rhs,
                model_name=model_name,
                E0=float(E0),
                epsilon=float(epsilon),
                **stable_path_kwargs,
            )

            rows.append(
                {
                    "model": model_name,
                    "E0_target": float(E0),
                    "epsilon": float(epsilon),
                    "success": bool(diagnostics.get("success", False)),
                    "status": diagnostics.get("status"),
                    "x_0": diagnostics.get("x_0", np.nan),
                    "tau_0": diagnostics.get("tau_0", np.nan),
                    "u_0": diagnostics.get("u_0", np.nan),
                    "time_horizon": diagnostics.get("time_horizon", np.nan),
                    "admissible_path": diagnostics.get("admissible_path", False),
                }
            )

    detailed = pd.DataFrame(rows)

    summaries = []
    for E0, subset in detailed.groupby("E0_target"):
        ok = subset[(subset["success"] == True) & (subset["admissible_path"] == True)]
        summaries.append(
            {
                "model": model_name,
                "E0_target": float(E0),
                "n_success": int(len(ok)),
                "x0_spread": (
                    float(ok["x_0"].max() - ok["x_0"].min())
                    if len(ok) > 0
                    else np.nan
                ),
                "tau0_spread": (
                    float(ok["tau_0"].max() - ok["tau_0"].min())
                    if len(ok) > 0
                    else np.nan
                ),
                "u0_spread": (
                    float(ok["u_0"].max() - ok["u_0"].min())
                    if len(ok) > 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summaries)
    return detailed, summary
