import os
import subprocess
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. API Key
os.environ["GROQ_API_KEY"] = "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"

# ─────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────

@tool
def multiply_numbers(text: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο ΜΟΝΟ όταν χρειάζεται να πολλαπλασιάσεις δύο αριθμούς. Δώσε input μορφής 'a,b'."""
    a, b = text.split(",")
    result = float(a) * float(b)
    return str(result)

@tool
def read_file(filepath: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο για να διαβάσεις το περιεχόμενο ενός αρχείου. Δώσε το path του αρχείου ως input."""
    try:
        with open(filepath.strip(), "r", encoding="utf-8") as f:
            content = f.read()
        return content[:3000]
    except FileNotFoundError:
        return f"Σφάλμα: Το αρχείο '{filepath}' δεν βρέθηκε."
    except Exception as e:
        return f"Σφάλμα κατά την ανάγνωση: {str(e)}"

@tool
def execute_python(code: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο για να εκτελέσεις Python κώδικα. Δώσε valid Python code ως input."""
    try:
        # Γράφουμε τον κώδικα σε temp αρχείο
        with open("temp_code.py", "w", encoding="utf-8") as f:
            f.write(code)
        
        result = subprocess.run(
            ["python", "temp_code.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout if result.stdout else "Ο κώδικας εκτελέστηκε χωρίς output."
        else:
            return f"Σφάλμα: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Σφάλμα: Timeout."
    except Exception as e:
        return f"Σφάλμα: {str(e)}"

# ─────────────────────────────────────────
# 3. FAISS VECTOR MEMORY
# ─────────────────────────────────────────

print("Φορτώνω το embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_texts(
    ["Αρχικοποίηση μνήμης agent."],
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# ─────────────────────────────────────────
# 4. LLM + AGENT
# ─────────────────────────────────────────

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

tools = [multiply_numbers, read_file, execute_python]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # ← ΠΡΟΣΘΕΣΕ ΑΥΤΟfrom
)
# ─────────────────────────────────────────
# 5. MAIN LOOP
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n✅ Agent έτοιμος! Tools: multiply | read_file | execute_python | FAISS memory")
    print("Γράψε 'exit' για να τερματίσεις.")
    print("-" * 50)

    while True:
        query = input("\nΕρώτηση: ")
        if query.lower() == "exit":
            break
        agent.run(query)