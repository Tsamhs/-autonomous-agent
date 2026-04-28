"""
Αρχικοποίηση και διαχείριση του LangChain agent.
Αυτό το service συνδυάζει LLM + tools + memory σε έναν έτοιμο agent.
"""

import os
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType

from agent.config import settings
from agent.tools.tools import ALL_TOOLS
from agent.services.memory_service import create_memory


def create_agent():
    """
    Δημιουργεί και επιστρέφει έναν πλήρως διαμορφωμένο LangChain agent.
    Εκτελεί validation ρυθμίσεων πριν την εκκίνηση.
    """
    # Έλεγχος ότι υπάρχει το API key πριν ξεκινήσουμε
    settings.validate()
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

    memory = create_memory()

    llm = ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )

    return initialize_agent(
        ALL_TOOLS,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=settings.MAX_ITERATIONS,
    )
