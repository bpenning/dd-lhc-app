# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="DM: Direct ↔ Indirect Mapper", layout="wide")

st.title("Dark Matter: Direct ↔ Indirect Mapper (MVP)")

with st.sidebar:
    st.header("Model")
    model = st.selectbox("Simplified model", ["Dirac χ + spin‑1 vector mediator (universal quark couplings)"])
    mchi = st.number_input("mχ [GeV]", min_value=1.0, value=100.0, step=1.0, format="%.3f")
    Mmed = st.number_input("Mmed [GeV]", min_value=10.0, value=1000.0, step=10.0, format="%.3f")
    gchi = st.number_input("gχ", min_value=0.0, value=1.0, step=0.1, format="%.3f")
    gq   = st.number_input("gq (universal to quarks)", min_value=0.0, value=0.1, step=0.01, format="%.3f")
    channel = st.selectbox("ID final state (for comparison to limits)", ["b b̄", "τ+ τ−"])
    rho0 = st.number_input("Local DM density ρ0 [GeV/cm³] (for rate assumptions)", min_value=0.1, value=0.40, step=0.01)
    st.caption("Vector-mediator mapping; SI scattering & s-wave annihilation. See docs for caveats.")

# Constants
Nc = 3.0
mp = 0.938 # GeV
mn = 0.939 # GeV
mN = 0.939 # per-nucleon approx

# quark masses
mq = {"u":0.0022, "d":0.0047, "s":0.096, "c":1.27, "b":4.18, "t":172.76}  # GeV (pole-ish; for thresholds)

# conversion
# Convert GeV^-2 to cm^2: 1 GeV^-2 ≈ 0.389379e-24 cm^2
GEV2_TO_CM2 = 0.389379e-24
# c in cm/s for converting to cm^3/s
C_CM_S = 3.0e10

# helper functions
def red_mass(m1, m2):
    return m1*m2/(m1+m2)


def beta(M, m):
    if M<=2*m: return 0.0
    x = 1.0 - 4.0*(m**2)/(M**2)
    return np.sqrt(max(x,0.0))


def gamma_fermion_vec(M, g, m):
    if M<=2*m: return 0.0
    b = beta(M,m)
    return (g**2*M/(12.0*np.pi))*(1.0 + 2.0*(m**2)/M**2)*b


def mediator_width(Mmed, gq, gchi, mchi):
    """Approximate mediator width (vector), sum over kinematically open quarks and χχ if open."""
    Gamma = 0.0
    for q,m in mq.items():
        Gamma += Nc*gamma_fermion_vec(Mmed, gq, m)
    # χχ if open
    Gamma += gamma_fermion_vec(Mmed, gchi, mchi)
    return Gamma


def sigma_SI_per_nucleon(mchi_val, Mmed_val, gchi_val, gq_val):
    """Return σ_SI per nucleon in cm^2"""
    muN = red_mass(mchi_val, mN)
    sigma_SI = (muN**2/np.pi) * ((3.0*gq_val*gchi_val)/(Mmed_val**2))**2  # in GeV^-2
    return sigma_SI * GEV2_TO_CM2


def sigma_v_total(mchi_val, Mmed_val, gchi_val, gq_val, Gamma_val):
    """Compute Σ_q σv to open quarks; return in cm^3/s"""
    s_pref = 0.0
    for q, mqv in mq.items():
        if mchi_val<=mqv:
            continue
        bq = np.sqrt(max(1.0 - (mqv**2)/(mchi_val**2), 0.0))
        num = Nc*(gchi_val**2)*(gq_val**2)/(2.0*np.pi) * (mchi_val**2) * bq * (1.0 + 0.5*(mqv**2)/(mchi_val**2))
        den = (Mmed_val**2 - 4.0*(mchi_val**2))**2 + (Mmed_val**2)*(Gamma_val**2)
        s_pref += num/den
    # in GeV^-2 -> convert
    return s_pref * GEV2_TO_CM2 * C_CM_S

# --- UI: plotting mode
mode = st.sidebar.selectbox("Plot mode", ["Single point", "Parameter sweep", "Collect model points"]) 

# session state for collected points
if 'model_points' not in st.session_state:
    st.session_state.model_points = []

if mode == "Parameter sweep":
    st.sidebar.markdown("### Sweep settings")
    mchi_min = st.sidebar.number_input("mχ min [GeV]", min_value=0.1, value=1.0, step=1.0)
    mchi_max = st.sidebar.number_input("mχ max [GeV]", min_value=1.0, value=1000.0, step=1.0)
    npoints = st.sidebar.number_input("n points", min_value=10, value=200, step=10)
    if mchi_max <= mchi_min:
        st.sidebar.error("mχ max must be > mχ min for sweep")

col1, col2, col3 = st.columns(3)

# compute current point metrics
Gamma = mediator_width(Mmed, gq, gchi, mchi)
sigma_SI_cm2 = sigma_SI_per_nucleon(mchi, Mmed, gchi, gq)
sv_cm3s = sigma_v_total(mchi, Mmed, gchi, gq, Gamma)

col1.metric("σ_SI (per nucleon)", f"{sigma_SI_cm2:.2e} cm²")
col2.metric("Γ_med", f"{Gamma:.2e} GeV")
col3.metric("⟨σv⟩ (Σ q q̄)", f"{sv_cm3s:.2e} cm³/s")

st.markdown("### Compare to experimental limits")
tab1, tab2 = st.tabs(["Direct detection overlay", "Indirect detection overlay"])

with tab1:
    st.write("Upload a **DD limit curve** CSV with columns: `mchi_GeV, sigma_SI_cm2, label`.")
    dd_file = st.file_uploader("DD CSV", type=["csv"], key="dd")
    fig = go.Figure()

    # Plot depending on mode
    if mode == "Single point":
        fig.add_trace(go.Scatter(x=[mchi], y=[sigma_SI_cm2], mode="markers", name="Model point", marker=dict(size=10, color="crimson")))
    elif mode == "Collect model points":
        if st.button("Add current model to collection"):
            st.session_state.model_points.append({
                'mchi': float(mchi), 'Mmed': float(Mmed), 'gchi': float(gchi), 'gq': float(gq),
                'sigma_SI_cm2': float(sigma_SI_cm2), 'sv_cm3s': float(sv_cm3s)
            })
        if len(st.session_state.model_points) > 0:
            pts = pd.DataFrame(st.session_state.model_points)
            fig.add_trace(go.Scatter(x=pts['mchi'], y=pts['sigma_SI_cm2'], mode='markers+text', text=[f"p{i}" for i in range(len(pts))], textposition='top center', name='Collected points'))
            st.dataframe(pts)
        if st.button("Clear collection"):
            st.session_state.model_points = []
    elif mode == "Parameter sweep":
        if mchi_max > mchi_min and npoints > 1:
            mchis = np.logspace(np.log10(max(mchi_min, 1e-6)), np.log10(mchi_max), int(npoints))
            sigma_vals = []
            Gamma_val = mediator_width(Mmed, gq, gchi, mchi)  # mediator width only depends weakly on mchi via χ threshold; keep it simple and use current mchi for width or compute per-point if desired
            # compute per-point (more correct)
            sigma_vals = [sigma_SI_per_nucleon(mc, Mmed, gchi, gq) for mc in mchis]
            fig.add_trace(go.Scatter(x=mchis, y=sigma_vals, mode='lines', name='Model sweep (σ_SI)'))

    if dd_file is not None:
        dd = pd.read_csv(dd_file)
        for lab, grp in dd.groupby(dd.get("label", "DD limit")):
            fig.add_trace(go.Scatter(x=grp["mchi_GeV"], y=grp["sigma_SI_cm2"], name=str(lab), mode="lines"))
    fig.update_layout(xaxis_type="log", yaxis_type="log",
                      xaxis_title="mχ [GeV]", yaxis_title="σ_SI [cm²]",
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Upload an **ID limit curve** CSV with columns: `mchi_GeV, sv_cm3s, channel, label` "
             "for e.g. Fermi‑LAT dwarfs (bb̄ or τ+τ−).")
    id_file = st.file_uploader("ID CSV", type=["csv"], key="id")
    fig2 = go.Figure()

    if mode == "Single point":
        fig2.add_trace(go.Scatter(x=[mchi], y=[sv_cm3s], mode="markers", name="Model point (Σqq̄)", marker=dict(size=10, color="seagreen")))
    elif mode == "Collect model points":
        if len(st.session_state.model_points) > 0:
            pts = pd.DataFrame(st.session_state.model_points)
            fig2.add_trace(go.Scatter(x=pts['mchi'], y=pts['sv_cm3s'], mode='markers+text', text=[f"p{i}" for i in range(len(pts))], textposition='top center', name='Collected points'))
    elif mode == "Parameter sweep":
        if mchi_max > mchi_min and npoints > 1:
            mchis = np.logspace(np.log10(max(mchi_min, 1e-6)), np.log10(mchi_max), int(npoints))
            sv_vals = []
            sv_vals = [sigma_v_total(mc, Mmed, gchi, gq, mediator_width(Mmed, gq, gchi, mc)) for mc in mchis]
            fig2.add_trace(go.Scatter(x=mchis, y=sv_vals, mode='lines', name='Model sweep (⟨σv⟩)'))

    if id_file is not None:
        iddf = pd.read_csv(id_file)
        # Filter by selected channel
        sel = 'b' if ('b' in channel.lower()) else 'tau'
        if 'channel' in iddf.columns:
            if sel == 'b':
                mask = iddf['channel'].astype(str).str.contains('b', case=False, na=False)
            else:
                mask = iddf['channel'].astype(str).str.contains('tau', case=False, na=False)
            iddf = iddf[mask]
        for lab, grp in iddf.groupby(iddf.get("label", "ID limit")):
            fig2.add_trace(go.Scatter(x=grp["mchi_GeV"], y=grp["sv_cm3s"], name=str(lab), mode="lines"))
    fig2.update_layout(xaxis_type="log", yaxis_type="log",
                       xaxis_title="mχ [GeV]", yaxis_title="⟨σv⟩ [cm³/s]",
                       height=500)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**Notes**  
- SI mapping and annihilation formulas follow standard vector‑mediator simplified‑model benchmarks used to compare LHC, DD, and ID searches.  
- For ID comparisons we overlay *published* cross‑section **limits vs mass** (per channel) so the astrophysical J‑factors are those used by each experiment.
""")
