# Autonomous AI Agent

Αυτόνομος AI Agent χτισμένος με **LangChain**, **Groq (Llama 3.1 8B)**, **FAISS memory** και **Self-Refinement**.

## Χαρακτηριστικά

| Χαρακτηριστικό | Περιγραφή |
|---|---|
| 🔧 Tool Use | Πολλαπλασιασμός, ανάγνωση αρχείων, εκτέλεση Python |
| 🧠 FAISS Memory | Vector-based μνήμη για συνεχή συνομιλία |
| 🔄 Self-Refinement | Αυτόματη αξιολόγηση και βελτίωση απαντήσεων |
| 🖥️ Streamlit UI | Web interface για αλληλεπίδραση |

## Εγκατάσταση

```bash
# 1. Κλωνοποίηση
git clone https://github.com/<your-username>/autonomous-agent.git
cd autonomous-agent

# 2. Virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Εξαρτήσεις
pip install -r requirements.txt

# 4. Ρυθμίσεις
cp .env.example .env
# Άνοιξε το .env και βάλε το GROQ_API_KEY σου
```

## Εκτέλεση

```bash
# Streamlit app
streamlit run main.py

# Benchmark
python -m tests.benchmark
```

## Δομή

```
.
├── agent/
│   ├── config/         # Ρυθμίσεις (settings.py)
│   ├── models/         # Data models (response.py)
│   ├── services/       # Business logic (agent, memory, refinement)
│   ├── tools/          # LangChain tools
│   └── ui/             # Streamlit components
├── tests/
│   └── benchmark.py    # Αξιολόγηση agent
├── main.py             # Entry point
├── .env.example        # Template ρυθμίσεων
└── requirements.txt
```

## API Key

Δωρεάν Groq API key: [console.groq.com](https://console.groq.com)
