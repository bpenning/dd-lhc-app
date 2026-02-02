# Streamlit UI for DD <-> LHC converter
import streamlit as st
import pandas as pd
import numpy as np

from conversions import (
    get_default_config,
    Config,
    dd2lhc_SD,
    dd2lhc_SI,
    lhc2dd_SD,
    lhc2dd_SI,
)

st.set_page_config(page_title="DD <-> LHC Converter", page_icon="⚙️", layout="wide")

st.title("Direct Detection <-> LHC Converter")

st.markdown(
    """
This app converts between direct-detection cross-sections (sigma, cm^2) and simplified-model mediator masses (m_med, GeV),
for common interaction types (spin-independent scalar/vector; spin-dependent proton/neutron). It implements vectorised
formulae in a separate conversions module and includes improved input validation and configuration handling.
    """
)

# Default numeric formatting
SIGMA_INPUT_FORMAT = "%e"
GENERIC_FLOAT_FORMAT = "%f"

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
    DEFAULTS = get_default_config()
    gDM = st.number_input("gDM", min_value=0.0, value=float(DEFAULTS.gDM), step=0.1, format=GENERIC_FLOAT_FORMAT)

    # Show relevant couplings depending on interaction
    if interaction.startswith("SI - scalar"):
        gSM = st.number_input("gSM (scalar universal)", min_value=0.0, value=float(DEFAULTS.gSM), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gu = st.number_input("gu (only used for vector/SD)", min_value=0.0, value=float(DEFAULTS.gu), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gd = st.number_input("gd (only used for vector/SD)", min_value=0.0, value=float(DEFAULTS.gd), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gs = st.number_input("gs (only used for SD)", min_value=0.0, value=float(DEFAULTS.gs), step=0.1, format=GENERIC_FLOAT_FORMAT)
    elif interaction.startswith("SI - vector"):
        gSM = st.number_input("gSM (only used for scalar)", min_value=0.0, value=float(DEFAULTS.gSM), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gu = st.number_input("gu", min_value=0.0, value=float(DEFAULTS.gu), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gd = st.number_input("gd", min_value=0.0, value=float(DEFAULTS.gd), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gs = st.number_input("gs (only used for SD)", min_value=0.0, value=float(DEFAULTS.gs), step=0.1, format=GENERIC_FLOAT_FORMAT)
    else:  # SD
        gSM = st.number_input("gSM (only used for scalar)", min_value=0.0, value=float(DEFAULTS.gSM), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gu = st.number_input("gu", min_value=0.0, value=float(DEFAULTS.gu), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gd = st.number_input("gd", min_value=0.0, value=float(DEFAULTS.gd), step=0.1, format=GENERIC_FLOAT_FORMAT)
        gs = st.number_input("gs", min_value=0.0, value=float(DEFAULTS.gs), step=0.1, format=GENERIC_FLOAT_FORMAT)

    with st.expander("Advanced constants"):
        _cfg = get_default_config()
        _mn = st.number_input("Nucleon mass mn [GeV]", value=float(_cfg.mn), step=0.001, format=GENERIC_FLOAT_FORMAT)
        _v = st.number_input("EW vev v [GeV] (scalar SI)", value=float(_cfg.v), step=1.0, format=GENERIC_FLOAT_FORMAT)
        _fup = st.number_input("f_u^p (scalar)", value=float(_cfg.fup), step=0.001, format=GENERIC_FLOAT_FORMAT)
        _fdp = st.number_input("f_d^p (scalar)", value=float(_cfg.fdp), step=0.001, format=GENERIC_FLOAT_FORMAT)
        _fsp = st.number_input("f_s^p (scalar)", value=float(_cfg.fsp), step=0.001, format=GENERIC_FLOAT_FORMAT)
        _Delta_up = st.number_input("Delta u_p (SD)", value=float(_cfg.Delta_u_p), step=0.01, format=GENERIC_FLOAT_FORMAT)
        _Delta_dp = st.number_input("Delta d_p (SD)", value=float(_cfg.Delta_d_p), step=0.01, format=GENERIC_FLOAT_FORMAT)
        _Delta_sp = st.number_input("Delta s_p (SD)", value=float(_cfg.Delta_s_p), step=0.01, format=GENERIC_FLOAT_FORMAT)
        _Delta_un = st.number_input("Delta u_n (SD)", value=float(_cfg.Delta_u_n), step=0.01, format=GENERIC_FLOAT_FORMAT)
        _Delta_dn = st.number_input("Delta d_n (SD)", value=float(_cfg.Delta_d_n), step=0.01, format=GENERIC_FLOAT_FORMAT)
        _Delta_sn = st.number_input("Delta s_n (SD)", value=float(_cfg.Delta_s_n), step=0.01, format=GENERIC_FLOAT_FORMAT)

        # Build a local Config instance (do not mutate module globals)
        local_config = Config(
            mn=_mn,
            v=_v,
            fup=_fup,
            fdp=_fdp,
            fsp=_fsp,
            fTG=1.0 - _fup - _fdp - _fsp,
            Delta_u_p=_Delta_up,
            Delta_d_p=_Delta_dp,
            Delta_s_p=_Delta_sp,
            Delta_u_n=_Delta_un,
            Delta_d_n=_Delta_dn,
            Delta_s_n=_Delta_sn,
            gDM=float(DEFAULTS.gDM),
            gu=float(DEFAULTS.gu),
            gd=float(DEFAULTS.gd),
            gs=float(DEFAULTS.gs),
            gSM=float(DEFAULTS.gSM),
        )

    # If the user didn't expand advanced constants, use defaults
    if 'local_config' not in locals():
        local_config = get_default_config()

st.markdown("---")

st.subheader("Input data")
mode_cols = (
    ("Direct detection -> LHC (m_med)", ["m_DM", "sigma"]),
    ("LHC -> Direct detection (sigma)", ["m_DM", "m_med"]),
)
required_cols = dict(mode_cols)[direction]

st.caption(f"Required columns: {', '.join(required_cols)}")

input_choice = st.radio("Provide data via:", options=("Upload CSV", "Manual entry"), index=0, horizontal=True)


def _validate_df(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Validate and coerce required columns.

    Returns (error_message, df_coerced). If error_message == '' then df_coerced is safe to use.
    """
    df = df.copy()
    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return f"Missing required columns: {missing}", df

    # Coerce required columns to numeric
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    non_numeric = [c for c in required_cols if df[c].isna().any()]
    if non_numeric:
        return f"Required columns contain non-numeric or missing values: {non_numeric}", df

    # Check positivity
    if (df[required_cols] <= 0).any().any():
        return "All required columns must be strictly positive.", df

    # All good
    return "", df

if input_choice == "Upload CSV":
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        try:
            df_tmp = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_in = None
        else:
            err, df_in = _validate_df(df_tmp)
            if err:
                st.error(err)
                df_in = None
else:
    # Manual single-row entry
    cols = st.columns(len(required_cols))
    values = []
    for i, c in enumerate(required_cols):
        default = 100.0 if c in ("m_DM", "m_med") else 1e-45
        if c == "sigma":
            default = 1e-45
        fmt = SIGMA_INPUT_FORMAT if c == "sigma" else GENERIC_FLOAT_FORMAT
        values.append(cols[i].number_input(c, min_value=1e-300, value=float(default), format=fmt))
    df_tmp = pd.DataFrame([dict(zip(required_cols, values))])
    err, df_in = _validate_df(df_tmp)
    if err:
        st.error(err)
        df_in = None

if 'df_in' in locals() and df_in is not None:
    st.write("Input preview:")
    st.dataframe(df_in.head(), use_container_width=True)

    # Additional pre-checks
    if direction.startswith("LHC ->") and (df_in["m_med"] <= 0).any():
        st.error("All m_med values must be strictly positive for LHC -> Direct-detection conversion.")
    else:
        # Compute
        if direction.startswith("Direct detection"):
            if interaction.startswith("SD - "):
                target = 'proton' if 'proton' in interaction else 'neutron'
                df_out = dd2lhc_SD(df_in, gDM=gDM, gu=gu, gd=gd, gs=gs, target=target, config=local_config)
            elif interaction.startswith("SI - vector"):
                df_out = dd2lhc_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='vector', config=local_config)
            else:  # SI - scalar
                df_out = dd2lhc_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='scalar', config=local_config)
        else:
            if interaction.startswith("SD - "):
                target = 'proton' if 'proton' in interaction else 'neutron'
                df_out = lhc2dd_SD(df_in, gDM=gDM, gu=gu, gd=gd, gs=gs, target=target, config=local_config)
            elif interaction.startswith("SI - vector"):
                df_out = lhc2dd_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='vector', config=local_config)
            else:  # SI - scalar
                df_out = lhc2dd_SI(df_in, gDM=gDM, gu=gu, gd=gd, gSM=gSM, modifier='scalar', config=local_config)

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
