# --- Gemini + Google Search Chatbot Integration (Final Optimized) ---
import google.generativeai as genai
from googleapiclient.discovery import build
from dotenv import load_dotenv
import re
from urllib.parse import urlparse

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def google_search_items(query, num=6):
    """Search Google Custom Search and return structured results."""
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_SEARCH_CX"), num=num).execute()
        return res.get("items", [])
    except Exception as e:
        return [{"title": "search_error", "snippet": f"Search error: {e}", "link": ""}]

def extract_fact(items):
    """Try to find clear factual statements (like captain names, stats, etc.)."""
    for it in items:
        snippet = it.get("snippet", "")
        if any(k in snippet.lower() for k in ["captain", "coach", "won", "title", "ipl", "record"]):
            return snippet.strip(), it.get("link", "")
    return None, None

def concise_sources(items, n=3):
    """Return simplified formatted source list."""
    s = []
    for i, it in enumerate(items[:n], 1):
        domain = urlparse(it.get("link", "")).netloc
        s.append(f"{i}. {it.get('title','')} — {domain}")
    return "\n".join(s)

st.markdown("---")
st.header("💬 Ask the IPL Chatbot")

user_query = st.text_input("Ask any IPL question (e.g., 'List all RCB captains till date')")

if user_query:
    with st.spinner("Fetching real-time IPL data..."):
        try:
            search_queries = [
                f"{user_query} site:espncricinfo.com",
                f"{user_query} site:timesofindia.indiatimes.com",
                f"{user_query} site:cricbuzz.com",
                f"{user_query} site:royalchallengers.com",
                f"{user_query} IPL"
            ]

            all_results = []
            for q in search_queries:
                items = google_search_items(q)
                if items:
                    all_results.extend(items)
                if len(all_results) >= 5:
                    break

            if not all_results:
                st.error("No search results found. Verify API keys in your .env file.")
            else:
                fact, source = extract_fact(all_results)
                context = "\n".join([f"{it['title']} — {it['snippet']}" for it in all_results[:5]])

                prompt = (
                    f"Answer the question in one short, factual sentence using ONLY the data below.\n"
                    f"Provide concise, ChatGPT-style answer — no extra text.\n\n"
                    f"CONTEXT:\n{context}\n\nQUESTION: {user_query}"
                )

                response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)
                final_answer = response.text.strip() if response and response.text else "No clear answer found."

                st.success(final_answer)
                st.caption("Sources:")
                st.text(concise_sources(all_results))
        except Exception as e:
            st.error(f"Error: {e}")
