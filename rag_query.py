
import os
import json
import textwrap
import requests
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


CHROMA_DIR = "chroma_db"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same as ingestion
TOP_K = 5

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GROQ_MODEL = "llama-3.1-8b-instant"


if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment. Set it in your .env")


embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)


db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": TOP_K})


def build_context(docs):
    """
    docs: list of Document objects with .page_content and .metadata (langchain docs)
    Returns concatenated context string and a list of sources
    """
    pieces = []
    sources = []
    for i, d in enumerate(docs, start=1):
        content = d.page_content.strip()
        meta = d.metadata or {}
        # include a short metadata string if available
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
        piece = f"--- CHUNK {i} | {meta_str}\n{content}\n"
        pieces.append(piece)
        # sources: try to include filename / page info from metadata
        src = meta.get("source") or meta.get("filename") or meta.get("source_id") or f"chunk_{i}"
        sources.append(src)
    context = "\n\n".join(pieces)
    return context, sources


SYSTEM_PROMPT = textwrap.dedent("""
You are a compliance assistant. Given a user's query and supporting document chunks, produce a JSON object with a 'suggestions' list.
Each suggestion must include:
 - id (integer),
 - text (one-line suggestion),
 - type (Compliance action / Recommendation / Explanation / Escalation),
 - confidence (0-1 numeric),
 - why (one-line justification referencing the chunk id / short quote),
 - sources (list of source identifiers),
 - actionability (High/Medium/Low),
 - next_step (one-line instruction).

Output only valid JSON. If you cannot answer, return {"suggestions": []}.
""").strip()

USER_PROMPT_TEMPLATE = textwrap.dedent("""
User question:
{query}

Supporting document chunks (only use these to justify your suggestions). If you need to refer to specific text, quote at most one short excerpt per suggestion and include the chunk number.

Chunks:
{context}

Produce up to 5 suggestions in the JSON schema described by system prompt. Keep text concise.
""").strip()

def call_groq_chat(system_prompt: str, user_prompt: str, model=GROQ_MODEL, temperature=0.0, max_tokens=800):
    """
    Calls the Groq OpenAI-compatible chat completions endpoint.
    Returns raw assistant text.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()


    choices = data.get("choices") or []
    if len(choices) == 0:
        raise ValueError("No choices returned from Groq")

   
    first = choices[0]
    text = ""
    if "message" in first and isinstance(first["message"], dict):
        text = first["message"].get("content", "")
    else:
   
        text = first.get("text") or first.get("delta", {}).get("content", "")

    return text

def ask(query: str):

    docs = retriever.invoke(query)

    print(f"[retrieval] got {len(docs)} chunks")

    
    context, sources = build_context(docs)

    
    user_prompt = USER_PROMPT_TEMPLATE.format(query=query, context=context)

   
    raw = call_groq_chat(SYSTEM_PROMPT, user_prompt)
    print("[raw response]\n", raw)

  
    parsed = None
    try:
      
        start = raw.find("{")
        if start != -1:
            candidate = raw[start:]
            parsed = json.loads(candidate)
        else:
            parsed = json.loads(raw)
    except Exception as e:
        print("Warning: failed to parse JSON directly:", str(e))
      
        return {"raw": raw, "suggestions": []}

    return parsed

if __name__ == "__main__":
    print("Interactive RAG query. Type your question and press enter. Ctrl+C to exit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
            if not q:
                continue
            out = ask(q)
           
            print("\n=== Parsed Suggestions ===")
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as ex:
            print("Error:", ex)
            break
