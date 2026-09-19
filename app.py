from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_tools import (
    BASE_B,
    BASE_DELTA,
    BASE_ETA,
    BASE_M_L,
    BASE_M_NL,
    BASE_PBAR,
    assimilation_dataframe,
    base_solutions,
    base_table,
    load_initial_condition_sensitivity,
    load_multiplicity_boundaries,
    load_multiplicity_region,
    load_off_manifold_summary,
    load_off_manifold_trajectory,
    load_sensitivity,
    polynomial_stationary_points,
    result_row,
    solve_pair,
    stable_paths_for_E0,
)


st.set_page_config(
    page_title="Nieliniowa asymilacja a zrównoważony wzrost",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1450px;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: .8rem 1rem; border-radius: .8rem;}
      .hero {padding: 1.5rem 1.6rem; border: 1px solid rgba(128,128,128,.22); border-radius: 1rem; margin-bottom: 1.2rem;}
      .hero h1 {margin-bottom: .35rem;}
      .soft {opacity: .78;}
      .small-note {font-size: .9rem; opacity: .76;}
      div[data-testid="stExpander"] details {border-radius: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def thesis_header(subtitle: str | None = None):
    st.markdown(
        """
        <div class="hero">
          <h1>Wpływ nieliniowej funkcji asymilacji zanieczyszczeń na dynamikę modelu zrównoważonego wzrostu gospodarczego</h1>
          <div class="soft">Agata Kwiatkowska · Informatyka i ekonometria · Poznań 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.subheader(subtitle)


@st.cache_data(show_spinner=False)
def cached_base_table():
    return base_table()


@st.cache_data(show_spinner=False)
def cached_base_solutions():
    return base_solutions()


@st.cache_data(show_spinner=False)
def cached_sensitivity(parameter):
    return load_sensitivity(parameter)


@st.cache_data(show_spinner=False)
def cached_e0_results():
    return load_initial_condition_sensitivity()


@st.cache_data(show_spinner=False)
def cached_paths(E0):
    return stable_paths_for_E0(float(E0))


@st.cache_data(show_spinner=False)
def cached_multiplicity_region():
    return load_multiplicity_region()


@st.cache_data(show_spinner=False)
def cached_multiplicity_boundaries():
    return load_multiplicity_boundaries()


@st.cache_data(show_spinner=False)
def cached_off_summary():
    return load_off_manifold_summary()


@st.cache_data(show_spinner=False)
def cached_off_path(model, case):
    return load_off_manifold_trajectory(model, case)


# ---------- wspólne wykresy ----------
def assimilation_plot(Pbar=BASE_PBAR, m_l=BASE_M_L, m_nl=BASE_M_NL):
    df = assimilation_dataframe(Pbar, m_l, m_nl)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["E"], y=df["Liniowa"], mode="lines", name="Liniowa"))
    fig.add_trace(go.Scatter(x=df["E"], y=df["Nieliniowa"], mode="lines", name="Nieliniowa"))
    fig.add_vline(x=Pbar / 2, line_dash="dot", annotation_text="P̄/2")
    fig.update_layout(
        xaxis_title="Zasób środowiska E",
        yaxis_title="Strumień asymilacji A(E)",
        legend_title_text="Model",
        margin=dict(l=10, r=10, t=20, b=10),
        height=430,
    )
    return fig


def trajectory_figure(path_l, path_nl, variable, ylabel):
    fig = go.Figure()
    if path_l is not None:
        fig.add_trace(go.Scatter(x=path_l["time"], y=path_l[variable], mode="lines", name="Liniowy"))
    if path_nl is not None:
        fig.add_trace(go.Scatter(x=path_nl["time"], y=path_nl[variable], mode="lines", name="Nieliniowy"))
    fig.update_layout(xaxis_title="Czas", yaxis_title=ylabel, height=350, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def show_base_metrics():
    df = cached_base_table()
    if len(df) < 2:
        st.error("Nie udało się odtworzyć scenariusza bazowego.")
        return
    L = df[df["model"] == "liniowy"].iloc[0]
    N = df[df["model"] == "nieliniowy"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Zasób środowiska E*", fmt(N["E*"]), f"{(N['E*']/L['E*']-1)*100:+.1f}%".replace(".", ","))
    c2.metric("Udział kapitału u*", fmt(N["u*"]), f"{(N['u*']/L['u*']-1)*100:+.1f}%".replace(".", ","))
    c3.metric("Tempo wzrostu g*", fmt(N["g*"]), f"{(N['g*']/L['g*']-1)*100:+.1f}%".replace(".", ","))
    st.caption("Wartość główna: model nieliniowy. Zmiana: względem modelu liniowego.")


# =====================================================================
# TRYB OBRONY
# =====================================================================
PRESENTATION_PAGES = [
    "1. Problem i cel",
    "2. Model bazowy",
    "3. Modyfikacja funkcji asymilacji",
    "4. Wyniki analityczne",
    "5. Scenariusz bazowy",
    "6. Dynamika przejściowa",
    "7. Wielopunktowość",
    "8. Wnioski i ograniczenia",
]


def presentation_page(page):
    if page == PRESENTATION_PAGES[0]:
        thesis_header("Problem badawczy")
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown(
                """
                ### Pytanie
                **Jakie konsekwencje dla właściwości dynamicznych modelu może mieć sposób reprezentacji procesu asymilacji zanieczyszczeń?**

                ### Cel
                Określenie konsekwencji zastąpienia liniowej funkcji asymilacji funkcją nieliniową zależną od stanu środowiska w modelu Cazzavillana i Musu (1998).

                ### Analizowane własności
                - istnienie i możliwość występowania wielu punktów stacjonarnych,
                - lokalna stabilność,
                - długookresowe tempo wzrostu,
                - dynamika przejściowa.
                """
            )
        with c2:
            st.markdown("### Schemat mechanizmu")
            st.markdown(
                """
                **Produkcja** → **emisje** → **stan środowiska**  
                ↑　　　　　　　　　　　　　↓  
                **alokacja kapitału** ← **dobrobyt i decyzje optymalne**
                """
            )
            st.info("Praca ma charakter teoretyczno-numeryczny. Parametryzacja nie odwzorowuje konkretnej gospodarki ani konkretnego ekosystemu.")

    elif page == PRESENTATION_PAGES[1]:
        thesis_header("Model bazowy")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Najważniejsze równania")
            st.latex(r"\dot K = B(1-u)K-C")
            st.latex(r"\dot E = A(E)-Z")
            st.latex(r"Z=B\left(\frac{1}{u}-1\right)")
            st.latex(r"g^*=\frac{1}{\eta}\left[B(1-\sqrt{\tau^*})-\delta\right]")
        with c2:
            st.markdown("### Interpretacja")
            st.markdown(
                """
                - **E** — zasób / jakość środowiska,
                - **u** — udział kapitału przeznaczany na ograniczanie emisji,
                - **Z** — strumień emisji,
                - **A(E)** — naturalny strumień asymilacji,
                - **x=C/K**, **τ** — zmienne zredukowanego układu dynamicznego.
                """
            )
            st.success("W modelu bazowym punkt stacjonarny jest lokalnie siodłowy przy spełnieniu warunków istnienia i jednoznaczności.")
        with st.expander("Szczegóły matematyczne"):
            st.write("Aplikacja korzysta bezpośrednio z tego samego zredukowanego układu (x, E, τ), co obliczenia w pracy.")

    elif page == PRESENTATION_PAGES[2]:
        thesis_header("Modyfikacja funkcji asymilacji")
        c1, c2 = st.columns([1.35, 1])
        with c1:
            st.plotly_chart(assimilation_plot(), use_container_width=True)
        with c2:
            st.latex(r"A_L(E)=m_L(\bar P-E)")
            st.latex(r"A_{NL}(E)=m_{NL}E\left(1-\frac{E}{\bar P}\right)")
            st.markdown(
                """
                W funkcji nieliniowej strumień asymilacji:
                - jest zerowy przy całkowitej degradacji i przy braku zanieczyszczeń,
                - osiąga maksimum dla $E=\bar P/2$,
                - zależy od bieżącego stanu środowiska.
                """
            )
            st.info(r"W porównaniu bazowym przyjęto tę samą maksymalną asymilację: $m_{NL}=4m_L$ dla $\bar P=1$.")

    elif page == PRESENTATION_PAGES[3]:
        thesis_header("Najważniejsze wyniki analityczne")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Punkty stacjonarne")
            st.write("Warunek środowiskowy ma dwie gałęzie po obu stronach maksimum asymilacji. Bez dodatkowych założeń nie ma ogólnej gwarancji jednoznaczności punktu stacjonarnego.")
        with c2:
            st.markdown("### Wzrost")
            st.write("Postać równania długookresowego tempa wzrostu pozostaje taka sama, ale wartość g* zależy od otrzymanego punktu stacjonarnego.")
        with c3:
            st.markdown("### Stabilność")
            st.write("Dla η ≥ 1 każdy ekonomicznie dopuszczalny punkt na górnej gałęzi jest siodłowy; punkt E*=P̄/2 jest siodłowy dla każdego η>0. Pozostałe przypadki zależą od parametrów.")
        st.latex(r"E_-(\tau)<\frac{\bar P}{2}<E_+(\tau)")
        st.warning("Istnienie punktu stacjonarnego nie oznacza automatycznie dodatniego wzrostu ani optymalności — te warunki są sprawdzane oddzielnie.")

    elif page == PRESENTATION_PAGES[4]:
        thesis_header("Porównanie w scenariuszu bazowym")
        show_base_metrics()
        df = cached_base_table()[["model", "E*", "P*", "u*", "x*", "g*", "Z*", "stabilność"]].copy()
        st.dataframe(df.style.format({c: "{:.4f}" for c in ["E*", "P*", "u*", "x*", "g*", "Z*"]}), use_container_width=True, hide_index=True)
        st.info("W modelu nieliniowym niższy stacjonarny zasób zanieczyszczeń występuje mimo wyższego strumienia emisji Z*. Jest to konsekwencja odmiennego kształtu funkcji asymilacji i odpowiadającego jej punktu stacjonarnego.")

    elif page == PRESENTATION_PAGES[5]:
        thesis_header("Dynamika przejściowa")
        E0 = st.slider("Początkowy stan środowiska E₀", 0.05, 0.95, 0.50, 0.01, key="presentation_E0")
        with st.spinner("Wyznaczanie stabilnych ścieżek..."):
            (path_l, diag_l), (path_nl, diag_nl) = cached_paths(E0)
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(trajectory_figure(path_l, path_nl, "E", "E(t)"), use_container_width=True)
        c2.plotly_chart(trajectory_figure(path_l, path_nl, "x", "x(t)"), use_container_width=True)
        c3.plotly_chart(trajectory_figure(path_l, path_nl, "u", "u(t)"), use_container_width=True)
        if path_nl is None:
            status = diag_nl.get("status") if diag_nl else "brak wyniku"
            st.warning(f"Dla modelu nieliniowego nie wyznaczono ekonomicznie dopuszczalnej stabilnej ścieżki. Status procedury: {status}.")
        elif diag_l and diag_nl:
            a, b, c = st.columns(3)
            a.metric("u₀ — liniowy", fmt(diag_l.get("u_0")))
            b.metric("u₀ — nieliniowy", fmt(diag_nl.get("u_0")))
            c.metric("E* — nieliniowy", fmt(diag_nl.get("E_star")))
        st.caption("Dla niskich E₀ procedura może osiągnąć granicę u=1. Nie jest to dowód braku wszystkich możliwych rozwiązań brzegowych.")

    elif page == PRESENTATION_PAGES[6]:
        thesis_header("Możliwość wielu punktów stacjonarnych")
        region = cached_multiplicity_region()
        fig = px.scatter(region, x="mNL", y="Pbar", color="n_points", labels={"mNL": "mₙₗ", "Pbar": "P̄", "n_points": "liczba punktów"})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
        c1, c2 = st.columns([1.35, 1])
        with c1:
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("### Przykład z pracy")
            st.latex(r"\bar P=400,\quad m_{NL}=0.05")
            points = polynomial_stationary_points(400.0, 0.05)
            st.dataframe(pd.DataFrame(points).style.format({"E*": "{:.4f}", "u*": "{:.4f}", "x*": "{:.4f}", "g*": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.warning("Eksperyment ma charakter eksploracyjny i wykracza poza podstawową parametryzację. Nie jest ogólną charakterystyką całej przestrzeni parametrów.")

    elif page == PRESENTATION_PAGES[7]:
        thesis_header("Wnioski i ograniczenia")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                ### Wnioski
                1. Postać funkcji asymilacji wpływa na warunki wyznaczające punkty stacjonarne.
                2. Nieliniowa specyfikacja może dopuścić więcej niż jeden ekonomicznie dopuszczalny punkt stacjonarny.
                3. Dla parametryzacji bazowej model nieliniowy daje wyższe E*, niższe u* i wyższe g*.
                4. Różnice dotyczą również dynamiki przejściowej i wymaganych warunków początkowych.
                """
            )
        with c2:
            st.markdown(
                """
                ### Ograniczenia
                - funkcja nieliniowa nie opisuje histerezy ani gwałtownych przejść między stanami ekosystemu,
                - bez ograniczenia stanu model może formalnie generować E<0,
                - parametryzacja jest stylizowana, nie empiryczna,
                - analiza wrażliwości jest jednoczynnikowa,
                - przyjęcie P̄=1 wpływa na warunki istnienia punktów stacjonarnych,
                - wielopunktowość zidentyfikowano tylko w badanym obszarze parametrów.
                """
            )
        st.success("Główny wniosek: sposób modelowania asymilacji nie wpływa wyłącznie na wartości liczbowe rozwiązania — może zmieniać także strukturę możliwych równowag i dynamikę modelu.")


# =====================================================================
# EKSPLORATOR
# =====================================================================
def explorer():
    thesis_header("Eksplorator modelu")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Funkcje asymilacji",
        "Punkty stacjonarne",
        "Analiza wrażliwości",
        "Dynamika przejściowa",
        "Wielopunktowość",
        "Test siodła",
    ])

    with tab1:
        st.subheader("Porównanie funkcji asymilacji")
        a, b, c = st.columns(3)
        Pbar = a.slider("P̄", 0.5, 5.0, 1.0, 0.1, key="assim_Pbar")
        m_l = b.slider("mL", 0.05, 2.0, 0.5, 0.05, key="assim_mL")
        same_max = c.checkbox("Jednakowa maksymalna asymilacja", value=True)
        if same_max:
            m_nl = 4.0 * m_l  # równość maksymalnej asymilacji obu funkcji
            c.metric("mNL", fmt(m_nl, 3))
        else:
            m_nl = c.number_input("mNL", min_value=0.01, max_value=20.0, value=2.0, step=0.1)
        st.plotly_chart(assimilation_plot(Pbar, m_l, m_nl), use_container_width=True)
        st.caption("W modelu liniowym maksimum A_L=mL·P̄, a w nieliniowym A_NL,max=mNL·P̄/4, stąd warunek równości maksimum daje mNL=4mL.")

    with tab2:
        st.subheader("Punkty stacjonarne dla P̄=1")
        st.caption("Ta zakładka odtwarza sposób porównania użyty w głównej analizie pracy.")
        c1, c2, c3, c4 = st.columns(4)
        B = c1.number_input("B", 0.05, 2.0, BASE_B, 0.01)
        delta = c2.number_input("δ", 0.001, 0.10, BASE_DELTA, 0.001, format="%.3f")
        eta = c3.number_input("η", 0.2, 5.0, BASE_ETA, 0.05)
        m_l = c4.number_input("mL", 0.01, 10.0, BASE_M_L, 0.05)
        m_nl = 4.0 * m_l
        p_l, p_nl, lin, nl = solve_pair(B, delta, eta, 1.0, m_l, m_nl)
        rows = []
        if lin is not None:
            rows.append(result_row(lin))
        rows.extend(result_row(r) for r in nl)
        if rows:
            out = pd.DataFrame(rows)
            st.dataframe(out[["model", "gałąź", "E*", "P*", "u*", "x*", "g*", "Z*", "stabilność", "dopuszczalny", "TVC"]].style.format({c: "{:.5f}" for c in ["E*", "P*", "u*", "x*", "g*", "Z*"]}), use_container_width=True, hide_index=True)
        else:
            st.warning("Dla podanych parametrów procedura nie znalazła ekonomicznie dopuszczalnego punktu stacjonarnego.")
        st.caption(f"mNL = 4·mL = {m_nl:.4f}")

    with tab3:
        st.subheader("Jednoczynnikowa analiza wrażliwości — wyniki z pracy")
        pcol, mcol = st.columns([1, 2])
        parameter = pcol.selectbox("Zmieniany parametr", ["η", "δ", "m"])
        metric_map = {"E*": "E", "u*": "u", "x*": "x", "g*": "g"}
        metric_label = mcol.selectbox("Wielkość na osi Y", list(metric_map))
        metric = metric_map[metric_label]
        df = cached_sensitivity(parameter).copy()
        found = df[(df["found"] == True)].copy()
        model_names = {"linear": "Liniowy", "nonlinear": "Nieliniowy"}
        found["Model"] = found["model"].map(model_names)
        xcol = "parameter_value"
        fig = px.line(found, x=xcol, y=metric, color="Model", labels={xcol: parameter, metric: metric_label})
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if parameter == "m":
            st.caption("Na osi poziomej pokazano mL; w modelu nieliniowym zachowano relację mNL=4mL.")
        no_root = df[df["found"] == False]
        if len(no_root):
            vals = no_root["parameter_value"].unique()
            st.info(f"Dla części siatki procedura nie znalazła punktu stacjonarnego. Liczba wartości bez rozwiązania: {len(vals)}.")

    with tab4:
        st.subheader("Stabilna ścieżka dla wspólnego E₀")
        E0 = st.slider("E₀", 0.05, 0.95, 0.50, 0.01, key="explorer_E0")
        with st.spinner("Obliczanie stabilnych ścieżek..."):
            (path_l, diag_l), (path_nl, diag_nl) = cached_paths(E0)
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(trajectory_figure(path_l, path_nl, "E", "E(t)"), use_container_width=True)
        c2.plotly_chart(trajectory_figure(path_l, path_nl, "x", "x(t)"), use_container_width=True)
        c3.plotly_chart(trajectory_figure(path_l, path_nl, "u", "u(t)"), use_container_width=True)
        diag_rows = []
        for label, d in [("Liniowy", diag_l), ("Nieliniowy", diag_nl)]:
            if d:
                diag_rows.append({
                    "Model": label,
                    "sukces": d.get("success"),
                    "status": d.get("status"),
                    "x0": d.get("x_0"),
                    "u0": d.get("u_0"),
                    "E*": d.get("E_star"),
                    "λ stabilna": d.get("lambda_stable"),
                    "dopuszczalna ścieżka": d.get("admissible_path"),
                })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
        grid = cached_e0_results()
        good_nl = grid[(grid["model"] == "nonlinear") & (grid["success"] == True) & (grid["admissible_path"] == True)]
        if len(good_nl):
            st.caption(f"W bazowej siatce E₀ najniższa wartość, dla której procedura wyznaczyła dopuszczalną ścieżkę nieliniową, wynosiła {good_nl['E0_target'].min():.3f}.")

    with tab5:
        st.subheader("Eksploracja wielopunktowości")
        region = cached_multiplicity_region()
        boundaries = cached_multiplicity_boundaries()
        fig = px.scatter(region, x="mNL", y="Pbar", color="n_points", labels={"mNL": "mNL", "Pbar": "P̄", "n_points": "liczba punktów"})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        Pbar = a.number_input("P̄ do sprawdzenia", 1.0, 500.0, 400.0, 1.0, key="multi_Pbar")
        m = b.number_input("mNL do sprawdzenia", 0.001, 20.0, 0.05, 0.001, format="%.4f", key="multi_m")
        points = polynomial_stationary_points(Pbar, m)
        st.write(f"Liczba znalezionych ekonomicznie dopuszczalnych punktów: **{len(points)}**")
        if points:
            st.dataframe(pd.DataFrame(points).style.format({"E*": "{:.6f}", "P*": "{:.6f}", "τ*": "{:.6f}", "u*": "{:.6f}", "x*": "{:.6f}", "g*": "{:.6f}", "Z*": "{:.6f}"}), use_container_width=True, hide_index=True)
        st.warning("Mapa pochodzi z eksploracyjnej siatki mNL∈[0,04;0,065], P̄∈[200;500] przy B=0,45, δ=0,01, η=1,5. Nie należy interpretować jej jako empirycznej kalibracji.")

    with tab6:
        st.subheader("Odchylenie od stabilnej ścieżki")
        c1, c2 = st.columns([1, 2])
        model_label = c1.radio("Model", ["Liniowy", "Nieliniowy"])
        variable = c1.radio("Wykres", ["Odległość od równowagi", "E(t)", "x(t)", "u(t)"])
        model = "linear" if model_label == "Liniowy" else "nonlinear"
        cases = ["stable", "x_minus", "x_plus", "u_minus", "u_plus"]
        labels = {
            "stable": "stabilna ścieżka",
            "x_minus": "x₀ − 1%",
            "x_plus": "x₀ + 1%",
            "u_minus": "u₀ − 1%",
            "u_plus": "u₀ + 1%",
        }
        ymap = {
            "Odległość od równowagi": ("distance_to_steady_state", "Odległość"),
            "E(t)": ("E", "E(t)"),
            "x(t)": ("x", "x(t)"),
            "u(t)": ("u", "u(t)"),
        }
        col, ytitle = ymap[variable]
        fig = go.Figure()
        for case in cases:
            df = cached_off_path(model, case)
            fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=labels[case]))
        fig.update_layout(xaxis_title="Czas", yaxis_title=ytitle, height=500, margin=dict(l=10, r=10, t=20, b=10))
        c2.plotly_chart(fig, use_container_width=True)
        summary = cached_off_summary()
        st.dataframe(summary[summary["model"] == model], use_container_width=True, hide_index=True)
        st.caption("Eksperyment z pracy: E₀=0,50, osobne zaburzenie x₀ lub u₀ o ±1%. Trajektorie poza stabilną rozmaitością dochodzą do granicy ekonomicznie dopuszczalnego obszaru.")


# =====================================================================
# NAWIGACJA
# =====================================================================
st.sidebar.title("Nawigacja")
mode = st.sidebar.radio("Tryb", ["🎓 Obrona", "🔬 Eksplorator"])

if mode == "🎓 Obrona":
    if "presentation_radio" not in st.session_state:
        st.session_state.presentation_radio = PRESENTATION_PAGES[0]

    selected = st.sidebar.radio(
        "Slajd / ekran",
        PRESENTATION_PAGES,
        key="presentation_radio",
    )
    i = PRESENTATION_PAGES.index(selected)
    presentation_page(selected)

    st.divider()
    prev_col, progress_col, next_col = st.columns([1, 3, 1])
    progress_col.progress((i + 1) / len(PRESENTATION_PAGES), text=f"{i+1}/{len(PRESENTATION_PAGES)}")
    if prev_col.button("← Poprzedni", disabled=i == 0, use_container_width=True):
        st.session_state.presentation_radio = PRESENTATION_PAGES[i - 1]
        st.rerun()
    if next_col.button("Następny →", disabled=i == len(PRESENTATION_PAGES) - 1, use_container_width=True):
        st.session_state.presentation_radio = PRESENTATION_PAGES[i + 1]
        st.rerun()
else:
    explorer()

st.sidebar.divider()
st.sidebar.caption("Obliczenia wykorzystują te same funkcje i procedury numeryczne co skrypty użyte w pracy. Zakładka wielopunktowości ma charakter eksploracyjny.")
