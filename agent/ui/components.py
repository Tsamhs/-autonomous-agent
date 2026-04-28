"""
Επαναχρησιμοποιήσιμα Streamlit components.
Κρατάμε την UI λογική χωριστά από το business logic.
"""

import streamlit as st
from agent.models.response import AgentResponse


def render_sidebar() -> None:
    """Εμφανίζει το sidebar με πληροφορίες για τον agent."""
    with st.sidebar:
        st.header("⚙️ Πληροφορίες")
        st.success("🟢 Agent Online")

        st.markdown("**Tools διαθέσιμα:**")
        st.markdown("- 🔢 `multiply_numbers`")
        st.markdown("- 📄 `read_file`")
        st.markdown("- 🐍 `execute_python`")

        st.divider()
        st.markdown("**Memory:** FAISS Vector Store")
        st.markdown("**Model:** Llama 3.1 8B (Groq)")
        st.markdown("**Self-Refinement:** ✅ Ενεργό")


def render_agent_response(response: AgentResponse) -> None:
    """
    Εμφανίζει την απάντηση του agent μαζί με τα refinement steps
    σε αναδιπλούμενο expander.
    """
    with st.expander("🔍 Δες τη διαδικασία Self-Refinement", expanded=False):
        for step in response.steps:
            st.markdown(f"**{step.name}:**")
            st.info(step.content)

    # Ένδειξη αν χρειάστηκε βελτίωση
    if response.was_refined:
        st.caption("✏️ Η απάντηση βελτιώθηκε αυτόματα.")

    st.markdown(f"**{response.final_answer}**")
