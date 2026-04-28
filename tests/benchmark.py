"""
Benchmark αξιολόγησης του agent.
Εκτέλεση: python -m tests.benchmark
"""

import time
from dataclasses import dataclass
from typing import List

from agent.services.agent_service import create_agent


# ── Ορισμός benchmark tasks ───────────────────────────

@dataclass
class BenchmarkTask:
    id: int
    category: str
    query: str
    expected: str   # Λέξη-κλειδί που πρέπει να υπάρχει στην απάντηση


TASKS: List[BenchmarkTask] = [
    BenchmarkTask(1, "Multiplication", "Πόσο κάνει 435 επί 928;",           "403680"),
    BenchmarkTask(2, "Multiplication", "Πόσο κάνει 5000 επί 3900;",         "19500000"),
    BenchmarkTask(3, "Multiplication", "Πολλαπλασίασε 123 και 456",          "56088"),
    BenchmarkTask(4, "Code Execution", "Τρέξε κώδικα: print(2**10)",         "1024"),
    BenchmarkTask(5, "Code Execution", "Τρέξε κώδικα: print(sum(range(1,101)))", "5050"),
    BenchmarkTask(6, "Code Execution", "Τρέξε κώδικα: print(len('autonomous agent'))", "16"),
    BenchmarkTask(7, "Reasoning",      "Ποια είναι η πρωτεύουσα της Ελλάδας;", "Αθήνα"),
    BenchmarkTask(8, "Reasoning",      "Πόσες μέρες έχει ο Φεβρουάριος σε δίσεκτο έτος;", "29"),
]


# ── Εκτέλεση ─────────────────────────────────────────

def run_benchmark() -> None:
    print("\n" + "=" * 60)
    print("🚀 BENCHMARK ΞΕΚΙΝΑΕΙ")
    print("=" * 60)

    agent = create_agent()
    results = []

    for task in TASKS:
        print(f"\n[Task {task.id}] {task.category}: {task.query}")
        start = time.time()
        try:
            answer = agent.run(task.query)
            elapsed = round(time.time() - start, 2)
            success = task.expected.lower() in answer.lower()
            status = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            answer = f"ERROR: {e}"
            elapsed = round(time.time() - start, 2)
            success = False
            status = "💥 ERROR"

        results.append({"task": task, "success": success, "elapsed": elapsed})
        print(f"   Απάντηση:    {answer}")
        print(f"   Αναμενόμενο: {task.expected}")
        print(f"   Αποτέλεσμα:  {status} ({elapsed}s)")

    _print_summary(results)


def _print_summary(results: list) -> None:
    """Εκτυπώνει συνολική αναφορά αποτελεσμάτων."""
    print("\n" + "=" * 60)
    print("📊 ΑΠΟΤΕΛΕΣΜΑΤΑ BENCHMARK")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    avg_time = round(sum(r["elapsed"] for r in results) / total, 2)

    print(f"\n✅ Επιτυχία:     {passed}/{total} ({round(passed / total * 100)}%)")
    print(f"⏱️  Μέσος χρόνος: {avg_time}s ανά task")

    # Ανάλυση ανά κατηγορία
    categories: dict = {}
    for r in results:
        cat = r["task"].category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["success"]:
            categories[cat]["passed"] += 1

    print("\n📈 Ανά κατηγορία:")
    for cat, data in categories.items():
        pct = round(data["passed"] / data["total"] * 100)
        print(f"   {cat}: {data['passed']}/{data['total']} ({pct}%)")

    print("\n" + "=" * 60)
    print("✅ BENCHMARK ΟΛΟΚΛΗΡΩΘΗΚΕ")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
