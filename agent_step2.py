import os
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# 1. Ενεργοποιούμε το δωρεάν API
os.environ["GROQ_API_KEY"] = "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"

# 2. Φτιάχνουμε το "Εργαλείο" (Tool)
# Το docstring (τα σχόλια με τα 3 αυτάκια) είναι ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ! 
# Από αυτό καταλαβαίνει το LLM πότε πρέπει να χρησιμοποιήσει το εργαλείο.
@tool
def multiply_numbers(a: float, b: float) -> float:
    """Χρησιμοποίησε αυτό το εργαλείο ΜΟΝΟ όταν χρειάζεται να πολλαπλασιάσεις δύο αριθμούς."""
    return a * b

# 3. Αρχικοποίηση του Μοντέλου (Llama 3 δωρεάν!)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 4. Δίνουμε το εργαλείο στον Agent
tools = [multiply_numbers]

# 5. Δημιουργία του Agent
# Το verbose=True θα μας τυπώσει όλη τη "σκέψη" του στο τερματικό!
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True 
)

if __name__ == "__main__":
    print("Ο Agent είναι έτοιμος και τρέχει ΔΩΡΕΑΝ με Llama 3!")
    print("Δοκίμασε να του κάνεις μια δύσκολη ερώτηση πολλαπλασιασμού.")
    print("-" * 50)
    
    query = input("Τι θες να ρωτήσεις τον Agent; (π.χ., Πόσο κάνει 435 επί 928;) : ")
    agent.run(query)
