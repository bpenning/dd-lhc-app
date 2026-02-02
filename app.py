
import streamlit as st
import pandas as pd
import numpy as np
import math

# ------------------------------
# Constants and nucleon structure inputs
# ------------------------------
# Couplings
DEFAULT_gDM = 1.0
DEFAULT_gSM = 1.0  # scalar universal SM coupling (Higgs-like portal)
DEFAULT_gu = 1.0
DEFAULT_gd = 1.0
DEFAULT_gs = 1.0

global mn, v, fup, fdp, fsp, fTG, Delta_u_p, Delta_d_p, Delta_s_p, Delta_u_n, Delta_d_n, Delta_s_n

# Nucleon mass [GeV] and conversion: cm^2 -> GeV^-2 (as in user's code)
mn = 0.938
conv_units = 2.568 * pow(10.0, 27.0)

# Axial charges (spin-dependent)
Delta_d_p, Delta_d_n = -0.42, -0.42
Delta_u_p, Delta_u_n = 0.85, 0.85
Delta_s_p, Delta_s_n = -0.08, -0.08

# Scalar matrix elements (spin-independent, scalar mediator)
# v: EW vev [GeV]
v = 246.0
fup, fdp = 0.0208, 0.0411  # arXiv:1506.04142
fsp = 0.043                 # arXiv:1301.1114
fTG = 1.0 - fup - fdp - fsp

# ------------------------------
# Core computations (vectorised)
# ------------------------------

def _mu_nDM(mDM: pd.Series) -> pd.Series:
    return mn * mDM / (mn + mDM)


def dd2lhc_SD(df: pd.DataFrame, gDM: float, gu: float, gd: float, gs: float, target: str) -> pd.DataFrame:
    """Direct-detection (sigma) -> mediator mass for axial (SD) interactions.
    target: 'proton' or 'neutron'
    Requires columns: m_DM [GeV], sigma [cm^2]
    Produces: m_med [GeV]
    """
    df = df.copy()
    if target == 'neutron':
        f = abs(gDM * (gu * Delta_u_n + gd * Delta_d_n + gs * Delta_s_n))
    else:
        f = abs(gDM * (gu * Delta_u_p + gd * Delta_d_p + gs * Delta_s_p))

    df['mu_nDM'] = _mu_nDM(df['m_DM'])
    df['sigma_in_GeV'] = df['sigma'] * conv_units
    # m_med = [ f * mu ]^(1/2) / [ (pi * sigma / 3) ]^(1/4)
    df['m_med'] = np.power(f * df['mu_nDM'], 0.5) / np.power(math.pi * df['sigma_in_GeV'] / 3.0, 0.25)
    return df


def dd2lhc_SI(df: pd.DataFrame, gDM: float, gu: float, gd: float, gSM: float, modifier: str) -> pd.DataFrame:
    """Direct-detection (sigma) -> mediator mass for SI interactions.
    modifier: 'scalar' or 'vector'
    Requires columns: m_DM [GeV], sigma [cm^2]
    Produces: m_med [GeV]
    """
    df = df.copy()
    df['mu_nDM'] = _mu_nDM(df['m_DM'])
    df['sigma_in_GeV'] = df['sigma'] * conv_units

    if modifier == 'scalar':
        # fmMed2 corresponds to coupling prefactor / m_med^2 in the Higgs-portal-like case
        fmMed2 = (mn / v) * gSM * gDM * (fup + fdp + fsp + 2.0 / 27.0 * fTG * 3.0)
        df['m_med'] = np.power(fmMed2 * df['mu_nDM'], 0.5) / np.power(math.pi * df['sigma_in_GeV'], 0.25)
    else:  # 'vector'
        # f = (2 gu + gd) * gDM
        df['m_med'] = np.power(((2.0 * gu + gd) * gDM) * df['mu_nDM'], 0.5) / np.power(math.pi * df['sigma_in_GeV'], 0.25)
    return df


def lhc2dd_SD(df: pd.DataFrame, gDM: float, gu: float, gd: float, gs: float, target: str) -> pd.DataFrame:
    """Mediator mass -> direct-detection sigma for axial (SD) interactions.
    target: 'proton' or 'neutron'
    Requires columns: m_DM [GeV], m_med [GeV]
    Produces: sigma [cm^2]
    """
    df = df.copy()
    df['mu_nDM'] = _mu_nDM(df['m_DM'])

    if target == 'neutron':
        f = abs(gDM * (gu * Delta_u_n + gd * Delta_d_n + gs * Delta_s_n))
    else:
        f = abs(gDM * (gu * Delta_u_p + gd * Delta_d_p + gs * Delta_s_p))

    # sigma_in_GeV = 3 * (f * mu)^2 / (pi * m_med^4)
    df['sigma_in_GeV'] = 3.0 * np.power(f * df['mu_nDM'], 2.0) / (math.pi * np.power(df['m_med'], 4.0))
    df['sigma'] = df['sigma_in_GeV'] / conv_units
    return df


def lhc2dd_SI(df: pd.DataFrame, gDM: float, gu: float, gd: float, gSM: float, modifier: str) -> pd.DataFrame:
    """Mediator mass -> direct-detection sigma for SI interactions.
    modifier: 'scalar' or 'vector'
    Requires columns: m_DM [GeV], m_med [GeV]
    Produces: sigma [cm^2]
    """
    df = df.copy()
    df['mu_nDM'] = _mu_nDM(df['m_DM'])

    if modifier == 'vector':
        f = (2.0 * gu + gd) * gDM
        sigma_eq = np.power(f * df['mu_nDM'], 2.0) / (math.pi * np.power(df['m_med'], 4.0))
    else:  # scalar (Higgs-like)
        f = (mn / v) * gSM * gDM * (fup + fdp + fsp + 2.0 / 27.0 * fTG * 3.0) / np.power(df['m_med'], 2.0)
        sigma_eq = np.power(f * df['mu_nDM'], 2.0) / math.pi

    df['sigma_in_GeV'] = sigma_eq
    df['sigma'] = df['sigma_in_GeV'] / conv_units
    return df


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="DD <-> LHC Converter", page_icon="⚙️", layout="wide")

st.title("Direct Detection <-> LHC Converter")

st.markdown(
    """
This app converts between direct-detection cross-sections (sigma, cm^2) and simplified-model mediator masses (m_med, GeV),
for common interaction types (spin-independent scalar/vector; spin-dependent proton/neutron). It implements the vectorised
formulae from the provided reference code.
    """
)

with st.sidebar:
    st.header("Calculation setup")
    direction = st.radio(
        "Direction",
        options=("Direct detection -> LHC (m_med)", "LHC -> Direct detection (sigma)"),
        index=0,
    )

    interaction = st.selectbox(
        "Interaction",
        options=(
            "SI - scalar (Higgs-like)",
            "SI - vector",
            "SD - proton",
            "SD - neutron",
        ),
        index=0,
    )

    st.subheader("Couplings")
    gDM = st.number_input("gDM", min_value=0.0, value=DEFAULT_gDM, step=0.1, format="%f")

    # Show relevant couplings depending on interaction
    if interaction.startswith("SI - scalar"):
        gSM = st.number_input("gSM (scalar universal)", min_value=0.0, value=DEFAULT_gSM, step=0.1, format="%f")
        gu = st.number_input("gu (only used for vector/SD)", min_value=0.0, value=DEFAULT_gu, step=0.1, format="%f")
        gd = st.number_input("gd (only used for vector/SD)", min_value=0.0, value=DEFAULT_gd, step=0.1, format="%f")
        gs = st.number_input("gs (only used for SD)", min_value=0.0, value=DEFAULT_gs, step=0.1, format="%f")
    elif interaction.startswith("SI - vector"):
        gSM = st.number_input("gSM (only used for scalar)", min_value=0.0, value=DEFAULT_gSM, step=0.1, format="%f")
        gu = st.number_input("gu", min_value=0.0, value=DEFAULT_gu, step=0.1, format="%f")
        gd = st.number_input("gd", min_value=0.0, value=DEFAULT_gd, step=0.1, format="%f")
        gs = st.number_input("gs (only used for SD)", min_value=0.0, value=DEFAULT_gs, step=0.1, format="%f")
    else:  # SD
        gSM = st.number_input("gSM (only used for scalar)", min_value=0.0, value=DEFAULT_gSM, step=0.1, format="%f")
        gu = st.number_input("gu", min_value=0.0, value=DEFAULT_gu, step=0.1, format="%f")
        gd = st.number_input("gd", min_value=0.0, value=DEFAULT_gd, step=0.1, format="%f")
        gs = st.number_input("gs", min_value=0.0, value=DEFAULT_gs, step=0.1, format="%f")

    with st.expander("Advanced constants"):
        _mn = st.number_input("Nucleon mass mn [GeV]", value=float(mn), step=0.001, format="%f")
        _v = st.number_input("EW vev v [GeV] (scalar SI)", value=float(v), step=1.0, format="%f")
        _fup = st.number_input("f_u^p (scalar)", value=float(fup), step=0.001, format="%f")
        _fdp = st.number_input("f_d^p (scalar)", value=float(fdp), step=0.001, format="%f")
        _fsp = st.number_input("f_s^p (scalar)", value=float(fsp), step=0.001, format="%f")
        _Delta_up = st.number_input("Delta u_p (SD)", value=float(Delta_u_p), step=0.01, format="%f")
        _Delta_dp = st.number_input("Delta d_p (SD)", value=float(Delta_d_p), step=0.01, format="%f")
        _Delta_sp = st.number_input("Delta s_p (SD)", value=float(Delta_s_p), step=0.01, format="%f")
        _Delta_un = st.number_input("Delta u_n (SD)", value=float(Delta_u_n), step=0.01, format="%f")
        _Delta_dn = st.number_input("Delta d_n (SD)", value=float(Delta_d_n), step=0.01, format="%f")
        _Delta_sn = st.number_input("Delta s_n (SD)", value=float(Delta_s_n), step=0.01, format="%f")

        # Update globals for this session (kept local to computations via closures if needed)
        global mn, v, fup, fdp, fsp, fTG, Delta_u_p, Delta_d_p, Delta_s_p, Delta_u_n, Delta_d_n, Delta_s_n
        mn = _mn
        v = _v
        fup = _fup
        fdp = _fdp
        fsp = _fsp
        fTG = 1.0 - fup - fdp - fsp
        Delta_u_p = _Delta_up
        Delta_d_p = _Delta_dp
        Delta_s_p = _Delta_sp
        Delta_u_n = _Delta_un
        Delta_d_n = _Delta_dn
        Delta_s_n = _Delta_sn

st.markdown("---")

st.subheader("Input data")
mode_cols = (
    ("Direct detection -> LHC (m_med)", ["m_DM", "sigma"]),
    ("LHC -> Direct detection (sigma)", ["m_DM", "m_med"]),
)
required_cols = dict(mode_cols)[direction]

st.caption(f"Required columns: {', '.join(required_cols)}")

input_choice = st.radio("Provide data via:", options=("Upload CSV", "Manual entry"), index=0, horizontal=True)

def _validate_df(df: pd.DataFrame) -> str:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return f"Missing required columns: {missing}"
    if (df[required_cols] <= 0).any().any():
        return "All required columns must be strictly positive."
    return ""

if input_choice == "Upload CSV":
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_in = None
else:
    # Manual single-row entry
    cols = st.columns(len(required_cols))
    values = []
    for i, c in enumerate(required_cols):
        default = 100.0 if c in ("m_DM", "m_med") else 1e-45
        if c == "sigma":
            default = 1e-45
        values.append(cols[i].number_input(c, min_value=1e-300, value=float(default), format="%e"))
    df_in = pd.DataFrame([dict(zip(required_cols, values))])

if 'df_in' in locals() and df_in is not None:
    err = _validate_df(df_in)
    if err:
        st.error(err)
    else:
        st.write("Input preview:")
        st.dataframe(df_in.head(), use_container_width=True)

        # Compute
        if direction.startswith("Direct detection"):
            if interaction.startswith("SD - "):
                target = 'proton' if 'proton' in interaction else 'neutron'
                df_out = dd2lhc_SD(df_in, gDM=gDM, gu=gu, gd=gd, gs=gs, target=target)
            elif interaction.startswith("SI - vector"):
                df_out = dd2lhc_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='vector')
            else:  # SI - scalar
                df_out = dd2lhc_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='scalar')
        else:
            if interaction.startswith("SD - "):
                target = 'proton' if 'proton' in interaction else 'neutron'
                df_out = lhc2dd_SD(df_in, gDM=gDM, gu=gu, gd=gd, gs=gs, target=target)
            elif interaction.startswith("SI - vector"):
                df_out = lhc2dd_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='vector')
            else:  # SI - scalar
                df_out = lhc2dd_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='scalar')

        st.markdown("### Results")
        keep_cols = list(dict.fromkeys(required_cols + [c for c in df_out.columns if c not in required_cols]))
        st.dataframe(df_out[keep_cols], use_container_width=True)

        # Download
        csv_bytes = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )

st.markdown("""
**Notes**
- Units: m_DM and m_med in GeV; sigma in cm^2. Internally sigma is converted using conv_units = 2.568e27.
- Advanced constants (f_u, f_d, f_s, Delta q) can be adjusted in the sidebar if you need different inputs.
- For SI-scalar, the parameter gSM is used; for SI-vector and SD, the quark couplings gu, gd, gs are used (gs unused for SI-vector).
""")
