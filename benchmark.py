import os
import time
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
# AGENT INIT
# ─────────────────────────────────────────

print("Φορτώνω το embedding model...")
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

# ─────────────────────────────────────────
# BENCHMARK TASKS
# ─────────────────────────────────────────

tasks = [
    {
        "id": 1,
        "category": "Multiplication",
        "query": "Πόσο κάνει 435 επί 928;",
        "expected": "403680"
    },
    {
        "id": 2,
        "category": "Multiplication",
        "query": "Πόσο κάνει 5000 επί 3900;",
        "expected": "19500000"
    },
    {
        "id": 3,
        "category": "Multiplication",
        "query": "Πολλαπλασίασε 123 και 456",
        "expected": "56088"
    },
    {
        "id": 4,
        "category": "Code Execution",
        "query": "Τρέξε κώδικα: print(2**10)",
        "expected": "1024"
    },
    {
        "id": 5,
        "category": "Code Execution",
        "query": "Τρέξε κώδικα: print(sum(range(1, 101)))",
        "expected": "5050"
    },
    {
        "id": 6,
        "category": "Code Execution",
        "query": "Τρέξε κώδικα: print(len('autonomous agent'))",
        "expected": "16"
    },
    {
        "id": 7,
        "category": "Reasoning",
        "query": "Ποια είναι η πρωτεύουσα της Ελλάδας;",
        "expected": "Αθήνα"
    },
    {
        "id": 8,
        "category": "Reasoning",
        "query": "Πόσες μέρες έχει ο Φεβρουάριος σε δίσεκτο έτος;",
        "expected": "29"
    },
]

# ─────────────────────────────────────────
# RUN BENCHMARK
# ─────────────────────────────────────────

print("\n" + "="*60)
print("🚀 BENCHMARK ΞΕΚΙΝΑΕΙ")
print("="*60)

results = []

for task in tasks:
    print(f"\n[Task {task['id']}] {task['category']}: {task['query']}")
    
    start_time = time.time()
    try:
        answer = agent.run(task["query"])
        elapsed = round(time.time() - start_time, 2)
        success = task["expected"].lower() in answer.lower()
        status = "✅ PASS" if success else "❌ FAIL"
    except Exception as e:
        answer = f"ERROR: {str(e)}"
        elapsed = round(time.time() - start_time, 2)
        success = False
        status = "💥 ERROR"
    
    results.append({
        "id": task["id"],
        "category": task["category"],
        "success": success,
        "elapsed": elapsed,
        "answer": answer
    })
    
    print(f"   Απάντηση: {answer}")
    print(f"   Αναμενόμενο: {task['expected']}")
    print(f"   Αποτέλεσμα: {status} ({elapsed}s)")

# ─────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────

print("\n" + "="*60)
print("📊 ΑΠΟΤΕΛΕΣΜΑΤΑ BENCHMARK")
print("="*60)

total = len(results)
passed = sum(1 for r in results if r["success"])
avg_time = round(sum(r["elapsed"] for r in results) / total, 2)

print(f"\n✅ Επιτυχία:     {passed}/{total} tasks ({round(passed/total*100)}%)")
print(f"⏱️  Μέσος χρόνος: {avg_time}s ανά task")

# Ανά κατηγορία
categories = {}
for r in results:
    cat = next(t["category"] for t in tasks if t["id"] == r["id"])
    if cat not in categories:
        categories[cat] = {"total": 0, "passed": 0}
    categories[cat]["total"] += 1
    if r["success"]:
        categories[cat]["passed"] += 1

print("\n📈 Ανά κατηγορία:")
for cat, data in categories.items():
    pct = round(data["passed"]/data["total"]*100)
    print(f"   {cat}: {data['passed']}/{data['total']} ({pct}%)")

print("\n" + "="*60)
print("✅ BENCHMARK ΟΛΟΚΛΗΡΩΘΗΚΕ")
print("="*60)