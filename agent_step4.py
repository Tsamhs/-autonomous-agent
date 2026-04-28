import os
import subprocess
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["GROQ_API_KEY"] = "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"

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
# FAISS MEMORY
# ─────────────────────────────────────────

print("Φορτώνω το embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(["Αρχικοποίηση μνήμης agent."], embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# ─────────────────────────────────────────
# LLM + AGENT
# ─────────────────────────────────────────

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
tools = [multiply_numbers, read_file, execute_python]

agent = initialize_agent(
    tools, llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# ─────────────────────────────────────────
# SELF-REFINEMENT LOOP (Βήμα 4)
# ─────────────────────────────────────────

def run_with_refinement(query: str, max_rounds: int = 2) -> str:
    print(f"\n{'='*50}")
    print(f"🔄 Round 1: Παράγω αρχική απάντηση...")
    print(f"{'='*50}")
    
    answer = agent.run(query)
    print(f"\n📝 Αρχική απάντηση: {answer}")

    print(f"\n{'='*50}")
    print(f"🔍 Round 2: Αυτο-αξιολόγηση...")
    print(f"{'='*50}")

    verification_prompt = f"""
The original question was: "{query}"
The answer given was: "{answer}"
Is this answer correct and complete? Reply with only YES or NO.
"""
    verdict = agent.run(verification_prompt)
    print(f"\n🔍 Verdict: {verdict}")

    if "NO" in verdict.upper():
        print(f"\n🔧 Βελτιώνω την απάντηση...")
        answer = agent.run(f"Give a better answer to: {query}")
        print(f"\n✅ Βελτιωμένη απάντηση: {answer}")
    else:
        print(f"\n✅ Η απάντηση εγκρίθηκε!")

    return answer

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n✅ Agent με Self-Refinement έτοιμος!")
    print("Γράψε 'exit' για να τερματίσεις.")
    print("-" * 50)

    while True:
        query = input("\nΕρώτηση: ")
        if query.lower() == "exit":
            break
        final_answer = run_with_refinement(query)
        print(f"\n🎯 ΤΕΛΙΚΗ ΑΠΑΝΤΗΣΗ: {final_answer}")