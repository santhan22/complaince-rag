# rag_query.py
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Dict

# -------------------------
# LangChain imports
# -------------------------
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# -------------------------
# Environment + Config
# -------------------------
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment!")


# -------------------------
# LLM (Groq via LangChain)
# -------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.0
)


# -------------------------
# Prompt templates
# -------------------------
REWRITE_SYSTEM = """
You are a question rewriter. Based on the conversation history and the latest user query,
rewrite the user query into a standalone question suitable for document retrieval.
"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM),
    ("human", "{input}")
])

QA_SYSTEM = """
You are a compliance assistant.
Use ONLY the retrieved document chunks to answer.
If the answer is not present, reply: "I don't know; please consult the policy team."
Keep answers concise and factual.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# -------------------------
# Utility: format documents
# -------------------------
def format_docs(docs):
    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        out.append(f"[CHUNK {i} | {src} | page {page}]\n{d.page_content}")
    return "\n\n".join(out)


# -------------------------
# Memory Store (per-session)
# -------------------------
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


# -------------------------
# Build RAG chain (retriever injected)
# -------------------------
def build_rag_chain(retriever):
    """
    Builds a history-aware RAG chain using the provided retriever.
    """

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    def retrieval_fn(inputs):
        query = inputs["query"]
        docs = retriever.invoke(query)
        return {
            "context": format_docs(docs),
            "question": query,
        }

    qa_chain = (
        RunnableMap(
            {
                "query": lambda x: x["query"],
                "retrieval": RunnableLambda(retrieval_fn),
            }
        )
        | RunnableLambda(
            lambda x: {
                "context": x["retrieval"]["context"],
                "question": x["retrieval"]["question"],
            }
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    def rewrite(input_dict):
        rewritten = rewrite_chain.invoke({"input": input_dict["query"]})
        return {"query": rewritten}

    base_chain = RunnableLambda(rewrite) | qa_chain

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history=get_history,
        input_messages_key="query",
        history_messages_key="history",
    )


# -------------------------
# Public API (USED BY app.py)
# -------------------------
def ask_with_history(query: str, retriever, session_id: str = "default") -> str:
    rag_chain = build_rag_chain(retriever)
    return rag_chain.invoke(
        {"query": query},
        config={"configurable": {"session_id": session_id}},
    )


# -------------------------
# CLI (optional)
# -------------------------
if __name__ == "__main__":
    print("Multi-turn RAG CLI (session-only) — Ctrl+C to exit")
    sess = "cli"

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            print("⚠️ CLI mode requires retriever from app.py")
        except KeyboardInterrupt:
            break

