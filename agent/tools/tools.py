"""
Ορισμός των εργαλείων (tools) που έχει στη διάθεσή του ο agent.
Κάθε tool είναι μια αυτόνομη συνάρτηση με περιγραφή που καθοδηγεί το LLM.
"""

import subprocess
from langchain.tools import tool


@tool
def multiply_numbers(text: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο ΜΟΝΟ όταν χρειάζεται να πολλαπλασιάσεις δύο αριθμούς. Δώσε input μορφής 'a,b'."""
    try:
        a, b = text.split(",")
        return str(float(a) * float(b))
    except ValueError:
        return "Σφάλμα: Δώσε δύο αριθμούς χωρισμένους με κόμμα, π.χ. '12,34'."


@tool
def read_file(filepath: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο για να διαβάσεις το περιεχόμενο ενός αρχείου. Δώσε το path ως input."""
    try:
        with open(filepath.strip(), "r", encoding="utf-8") as f:
            # Περιορισμός στους 3000 χαρακτήρες για να μην κοπεί το context window
            return f.read()[:3000]
    except FileNotFoundError:
        return f"Σφάλμα: Το αρχείο '{filepath}' δεν βρέθηκε."
    except Exception as e:
        return f"Σφάλμα: {e}"


@tool
def execute_python(code: str) -> str:
    """Χρησιμοποίησε αυτό το εργαλείο για να εκτελέσεις Python κώδικα. Δώσε valid Python code ως input."""
    try:
        # Γράφουμε τον κώδικα σε προσωρινό αρχείο και τον εκτελούμε απομονωμένα
        with open("temp_code.py", "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            ["python", "temp_code.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else f"Σφάλμα: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Σφάλμα: Timeout — ο κώδικας πήρε πάνω από 10 δευτερόλεπτα."
    except Exception as e:
        return f"Σφάλμα: {e}"


# Λίστα όλων των διαθέσιμων tools — εδώ προσθέτουμε νέα tools στο μέλλον
ALL_TOOLS = [multiply_numbers, read_file, execute_python]
