"""
Μοντέλα δεδομένων για τις απαντήσεις του agent.
Χρησιμοποιούμε dataclasses για ελαφριά, τυπολογημένη αναπαράσταση δεδομένων.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RefinementStep:
    """Αναπαριστά ένα βήμα της διαδικασίας self-refinement."""

    name: str        # Τίτλος του βήματος (π.χ. "Round 1 - Αρχική απάντηση")
    content: str     # Περιεχόμενο / αποτέλεσμα του βήματος


@dataclass
class AgentResponse:
    """Η τελική απάντηση του agent μαζί με τα ενδιάμεσα βήματα."""

    final_answer: str
    steps: List[RefinementStep] = field(default_factory=list)
    was_refined: bool = False   # True αν χρειάστηκε βελτίωση στο Round 2
