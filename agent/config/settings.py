"""
Κεντρική διαχείριση ρυθμίσεων της εφαρμογής.
Οι τιμές φορτώνονται από το .env αρχείο (δεν γράφονται hardcoded στον κώδικα).
"""

import os
from dotenv import load_dotenv

# Φόρτωση μεταβλητών περιβάλλοντος από το .env αρχείο
load_dotenv()


# ── LLM ──────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Μνήμη (FAISS) ────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MEMORY_K: int = int(os.getenv("MEMORY_K", "3"))

# ── Agent ────────────────────────────────────────────
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))


def validate() -> None:
    """Ελέγχει ότι όλες οι υποχρεωτικές ρυθμίσεις έχουν οριστεί."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "Δεν βρέθηκε GROQ_API_KEY. "
            "Αντέγραψε το .env.example σε .env και συμπλήρωσε το κλειδί."
        )
