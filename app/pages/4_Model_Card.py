"""Model card: assumptions, sources, validation and the full parameter table."""

from pathlib import Path

import pandas as pd
import streamlit as st

from app_setup import ensure_package  # noqa: F401
from components import REPO, load_baseline

st.set_page_config(page_title="Model Card", page_icon="📋", layout="wide")
st.title("📋 Model Card")

card = REPO / "docs" / "model_card.md"
if card.exists():
    st.markdown(card.read_text(encoding="utf-8"))
else:
    st.warning("docs/model_card.md not found.")

st.divider()
st.subheader("Full parameter table (baseline)")


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


baseline = load_baseline()
flat = _flatten(baseline.to_dict())
table = pd.DataFrame(
    [{"parameter": k, "value": str(v)} for k, v in flat.items()]
)
st.dataframe(table, use_container_width=True, hide_index=True, height=600)
