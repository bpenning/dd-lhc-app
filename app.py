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
def red_mass(m1, m2): return m1*m2/(m1+m2)

muN = red_mass(mchi, mN)

# --- Direct detection: σ_SI^N
# σ_SI^N = μ_N^2 / π * (3 gq gχ / Mmed^2)^2  (vector, universal quark couplings)
sigma_SI = (muN**2/np.pi) * ((3.0*gq*gchi)/(Mmed**2))**2  # in GeV^-2
# Convert GeV^-2 to cm^2: 1 GeV^-2 ≈ 0.389379e-24 cm^2
GEV2_TO_CM2 = 0.389379e-24
sigma_SI_cm2 = sigma_SI * GEV2_TO_CM2

# --- Mediator width (approx, vector), sum over kinematically open quarks
mq = {"u":0.0022, "d":0.0047, "s":0.096, "c":1.27, "b":4.18, "t":172.76}  # GeV (pole-ish; for thresholds)
def beta(M, m): 
    if M<=2*m: return 0.0
    x = 1.0 - 4.0*(m**2)/(M**2)
    return np.sqrt(max(x,0.0))

def gamma_fermion_vec(M, g, m):
    if M<=2*m: return 0.0
    b = beta(M,m)
    return (g**2*M/(12.0*np.pi))*(1.0 + 2.0*(m**2)/M**2)*b

Gamma = 0.0
# quark decays
for q,m in mq.items():
    Gamma += Nc*gamma_fermion_vec(Mmed, gq, m)
# χχ if open
Gamma += gamma_fermion_vec(Mmed, gchi, mchi)

# --- ID: <σv> to q q̄ (sum over open quarks)
def sigma_v(mchi, M, gchi, gq, Gamma):
    s_pref = 0.0
    for q, mqv in mq.items():
        if mchi<=mqv: 
            continue
        bq = np.sqrt(1.0 - (mqv**2)/(mchi**2))
        num = Nc*(gchi**2)*(gq**2)/(2.0*np.pi) * (mchi**2) * bq * (1.0 + 0.5*(mqv**2)/(mchi**2))
        den = (M**2 - 4.0*(mchi**2))**2 + (M**2)*(Gamma**2)
        s_pref += num/den
    return s_pref  # in GeV^-2

sv_gevm2 = sigma_v(mchi, Mmed, gchi, gq, Gamma)
sv_cm3s = sv_gevm2 * GEV2_TO_CM2 * 3.0e10  # cm^3/s ; multiply by c ~ 3e10 cm/s (in natural units)

col1, col2, col3 = st.columns(3)
col1.metric("σ_SI (per nucleon)", f"{sigma_SI_cm2:.2e} cm²")
col2.metric("Γ_med", f"{Gamma:.2e} GeV")
col3.metric("⟨σv⟩ (Σ q q̄)", f"{sv_cm3s:.2e} cm³/s")

st.markdown("### Compare to experimental limits")
tab1, tab2 = st.tabs(["Direct detection overlay", "Indirect detection overlay"])

with tab1:
    st.write("Upload a **DD limit curve** CSV with columns: `mchi_GeV, sigma_SI_cm2, label`.")
    dd_file = st.file_uploader("DD CSV", type=["csv"], key="dd")
    fig = go.Figure()
    # Scatter our single-point prediction
    fig.add_trace(go.Scatter(x=[mchi], y=[sigma_SI_cm2], mode="markers",
                             name="Model point", marker=dict(size=10, color="crimson")))
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
    # Our predicted <σv> (approx total to open qq̄); for channel-by-channel compare, pick bb̄ in the file
    fig2.add_trace(go.Scatter(x=[mchi], y=[sv_cm3s], mode="markers",
                              name="Model point (Σqq̄)", marker=dict(size=10, color="seagreen")))
    if id_file is not None:
        iddf = pd.read_csv(id_file)
        # Filter by selected channel if present
        if "channel" in iddf.columns:
            iddf = iddf[iddf["channel"].astype(str).str.contains("b", case=False) if ("b" in channel) else
                        iddf["channel"].astype(str).str.contains("tau", case=False)]
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
