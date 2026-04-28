"""
Υλοποίηση του Self-Refinement loop.
Ο agent αξιολογεί μόνος του την απάντησή του και τη βελτιώνει αν χρειαστεί.
"""

from agent.models.response import AgentResponse, RefinementStep

# Prompt για αυτο-αξιολόγηση — απαιτεί μόνο YES/NO για αξιοπιστία
_VERIFICATION_PROMPT = (
    'Is this answer correct for "{query}"? '
    'Answer: "{answer}". Reply YES or NO only.'
)

_IMPROVEMENT_PROMPT = "Give a better and more complete answer to: {query}"


def run_with_refinement(agent, query: str) -> AgentResponse:
    """
    Εκτελεί το ερώτημα σε δύο rounds:
      Round 1 — παράγει αρχική απάντηση
      Round 2 — αξιολογεί αν είναι σωστή· αν όχι, παράγει βελτιωμένη

    Επιστρέφει AgentResponse με την τελική απάντηση και όλα τα ενδιάμεσα βήματα.
    """
    steps: list[RefinementStep] = []

    # ── Round 1: αρχική απάντηση ──────────────────────
    answer = agent.run(query)
    steps.append(RefinementStep("Round 1 — Αρχική απάντηση", answer))

    # ── Round 2: αυτο-αξιολόγηση ──────────────────────
    verdict = agent.run(
        _VERIFICATION_PROMPT.format(query=query, answer=answer)
    )
    steps.append(RefinementStep("Round 2 — Αξιολόγηση", verdict))

    # ── Round 3 (προαιρετικό): βελτίωση ───────────────
    was_refined = False
    if "NO" in verdict.upper():
        answer = agent.run(_IMPROVEMENT_PROMPT.format(query=query))
        steps.append(RefinementStep("Round 3 — Βελτιωμένη απάντηση", answer))
        was_refined = True

    return AgentResponse(
        final_answer=answer,
        steps=steps,
        was_refined=was_refined,
    )
