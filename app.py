import os
import subprocess
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Autonomous AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous AI Agent")
st.caption("Tool Use · FAISS Memory · Self-Refinement · Powered by Llama 3")

# ─────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────

@tool
def multiply_numbers(text: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο ΜΟΝΟ όταν χρειάζεται να πολλαπλασιάσεις δύο αριθμούς. Δώσε input μορφής 'a,b'."""
    a, b = text.split(",")
    return str(float(a) * float(b))

@tool
def read_file(filepath: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο για να διαβάσεις το περιεχόμενο ενός αρχείου."""
    try:
        with open(filepath.strip(), "r", encoding="utf-8") as f:
            return f.read()[:3000]
    except FileNotFoundError:
        return f"Σφάλμα: Το αρχείο '{filepath}' δεν βρέθηκε."
    except Exception as e:
        return f"Σφάλμα: {str(e)}"

@tool
def execute_python(code: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο για να εκτελέσεις Python κώδικα."""
    try:
        with open("temp_code.py", "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            ["python", "temp_code.py"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else f"Σφάλμα: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Σφάλμα: Timeout."
    except Exception as e:
        return f"Σφάλμα: {str(e)}"

# ─────────────────────────────────────────
# INIT AGENT (cached)
# ─────────────────────────────────────────

@st.cache_resource
def init_agent():
    os.environ["GROQ_API_KEY"] = "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"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(["Αρχικοποίηση μνήμης agent."], embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [multiply_numbers, read_file, execute_python]
    agent = initialize_agent(
        tools, llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5
    )
    return agent

# ─────────────────────────────────────────
# SELF-REFINEMENT
# ─────────────────────────────────────────

def run_with_refinement(agent, query: str):
    steps = []
    
    # Round 1
    steps.append(("🔄 Round 1", "Παράγω αρχική απάντηση..."))
    answer = agent.run(query)
    steps.append(("📝 Αρχική απάντηση", answer))
    
    # Round 2 - verification
    steps.append(("🔍 Round 2", "Αυτο-αξιολόγηση..."))
    verdict = agent.run(f'Is this answer correct for "{query}"? Answer: "{answer}". Reply YES or NO only.')
    steps.append(("🔍 Verdict", verdict))
    
    if "NO" in verdict.upper():
        steps.append(("🔧 Refinement", "Βελτιώνω την απάντηση..."))
        answer = agent.run(f"Give a better answer to: {query}")
        steps.append(("✅ Βελτιωμένη απάντηση", answer))
    else:
        steps.append(("✅ Έγκριση", "Η απάντηση είναι σωστή!"))
    
    return answer, steps

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.header("⚙️ Πληροφορίες")
    st.success("🟢 Agent Online")
    st.markdown("**Tools διαθέσιμα:**")
    st.markdown("- 🔢 multiply_numbers")
    st.markdown("- 📄 read_file")
    st.markdown("- 🐍 execute_python")
    st.markdown("**Memory:** FAISS Vector Store")
    st.markdown("**Model:** Llama 3.1 8B (Groq)")
    st.markdown("**Self-Refinement:** ✅ Ενεργό")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Κάνε μια ερώτηση στον Agent..."):
    
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Ο Agent σκέφτεται..."):
            agent = init_agent()
            final_answer, steps = run_with_refinement(agent, prompt)
        
        # Show refinement steps
        with st.expander("🔍 Δες τη διαδικασία Self-Refinement", expanded=False):
            for step_name, step_content in steps:
                st.markdown(f"**{step_name}:** {step_content}")
        
        st.markdown(f"**{final_answer}**")
        st.session_state.messages.append({"role": "assistant", "content": final_answer})