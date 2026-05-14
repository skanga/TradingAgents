"""Memory page — render the rolling decision log."""

import streamlit as st

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.default_config import DEFAULT_CONFIG
from gui.log_browser import memory_log_path, read_memory_log
from gui.md_utils import safe_md

st.set_page_config(page_title="Memory · TradingAgents", layout="wide")

st.title("Decision memory")

raw = read_memory_log()
log = TradingMemoryLog(DEFAULT_CONFIG)
parsed_entries = log.load_entries()
if not parsed_entries:
    st.info(
        f"Memory log is empty or not yet created at:\n\n`{memory_log_path()}`\n\n"
        "It populates after your first run and resolves after the holding window closes."
    )
    st.stop()

entries = [log.format_entry(entry) for entry in parsed_entries]
resolved = [
    log.format_entry(entry) for entry in parsed_entries if not entry.get("pending")
]
pending = [
    log.format_entry(entry) for entry in parsed_entries if entry.get("pending")
]

c1, c2 = st.columns(2)
c1.metric("Total entries", len(entries))
c2.metric("Resolved / Pending", f"{len(resolved)} / {len(pending)}")

view = st.radio("Show", ["All", "Resolved", "Pending"], horizontal=True, index=0)
shown = entries if view == "All" else (resolved if view == "Resolved" else pending)

st.markdown("---")
for e in shown:
    st.markdown(safe_md(e))
    st.markdown("---")

with st.expander("Show raw memory log file"):
    st.code(raw, language="markdown")
st.caption(f"Source: `{memory_log_path()}`")
