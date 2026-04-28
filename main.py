"""
Σημείο εισόδου της εφαρμογής.
Εκτέλεση: streamlit run main.py
"""

import streamlit as st
from agent.services.agent_service import create_agent
from agent.services.refinement_service import run_with_refinement
from agent.ui.components import render_sidebar, render_agent_response

# ── Page configuration ────────────────────────────────
st.set_page_config(
    page_title="Autonomous AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Autonomous AI Agent")
st.caption("Tool Use · FAISS Memory · Self-Refinement · Powered by Llama 3")

render_sidebar()


# ── Agent initialization (cached — τρέχει μόνο μία φορά) ──
@st.cache_resource
def get_agent():
    return create_agent()


# ── Chat history ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Εμφάνιση ιστορικού συνομιλίας
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ────────────────────────────────────────
if prompt := st.chat_input("Κάνε μια ερώτηση στον Agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ο Agent σκέφτεται..."):
            response = run_with_refinement(get_agent(), prompt)

        render_agent_response(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.final_answer}
        )
