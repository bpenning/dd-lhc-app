# DM: Direct ↔ Indirect Mapper (MVP)

Interactive Streamlit app on Hugging Face Spaces to relate direct-detection per-nucleon σ_SI to indirect-detection ⟨σv⟩ within a Dirac-χ + spin‑1 vector mediator model.

## Use
- Left panel: set model parameters (mχ, Mmed, gχ, gq).
- Main area: upload CSV limit curves or use built-in examples.
- Compare your model point to DD (σ_SI vs mχ) and ID (⟨σv⟩ vs mχ) overlays.

## Data format
- **DD CSV**: `mchi_GeV, sigma_SI_cm2, label`
- **ID CSV**: `mchi_GeV, sv_cm3s, channel, label` (e.g. channel `bbbar` or `tautau`)

## Local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
