
import streamlit as st
import json


from rag_query import ask

st.set_page_config(page_title="Compliance RAG Assistant", layout="wide")
st.title("Compliance RAG Assistant — Basic UI")

st.markdown(
    """
Enter a question below and click **Search**.  
You'll get structured suggestions + sources from your RAG system.
"""
)


with st.sidebar:
    st.header("Query")
    query = st.text_area("Enter your question", height=140)
    search_btn = st.button("Search")

if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""


if search_btn and query.strip():
    with st.spinner("Running RAG pipeline..."):
        try:
            resp = ask(query)
            st.session_state.last_response = resp
            st.session_state.last_query = query
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

resp = st.session_state.last_response


if resp:
    st.subheader(f"Raw JSON for query: **{st.session_state.last_query}**")
    st.json(resp)

    suggestions = resp.get("suggestions", [])

    if not suggestions:
        st.warning("No suggestions found.")
    else:
        st.subheader("Suggestions (Summary Table)")

        
        table = [
            {
                "id": s.get("id"),
                "text": s.get("text"),
                "type": s.get("type"),
                "confidence": s.get("confidence"),
                "actionability": s.get("actionability"),
            }
            for s in suggestions
        ]
        st.table(table)

       
        for s in suggestions:
            sid = s.get("id", "no-id")
            with st.expander(f"Suggestion {sid}: {s.get('text', '')}"):
                st.markdown(f"**Type:** {s.get('type')}")
                st.markdown(f"**Confidence:** {s.get('confidence')}")
                st.markdown(f"**Actionability:** {s.get('actionability')}")
                st.markdown(f"**Next Step:** {s.get('next_step')}")
                st.markdown(f"**Why:** {s.get('why')}")
                st.markdown("**Sources:**")
                for src in s.get("sources", []):
                    st.write(f"- {src}")

else:
    st.info("Enter a question in the sidebar and click **Search** to begin.")
