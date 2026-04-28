"""
Διαχείριση της μνήμης του agent μέσω FAISS vector store.
Η μνήμη επιτρέπει στον agent να θυμάται προηγούμενες αλληλεπιδράσεις.
"""

from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from agent.config.settings import EMBEDDING_MODEL, MEMORY_K


def create_memory() -> VectorStoreRetrieverMemory:
    """
    Δημιουργεί και επιστρέφει ένα FAISS-based memory object.
    Χρησιμοποιεί sentence-transformers για τη μετατροπή κειμένου σε vectors.
    """
    # Φόρτωση του embedding model (τρέχει τοπικά, δωρεάν)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Αρχικοποίηση του vector store με ένα placeholder κείμενο
    vectorstore = FAISS.from_texts(
        ["Αρχικοποίηση μνήμης agent."],
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": MEMORY_K})
    return VectorStoreRetrieverMemory(retriever=retriever)
