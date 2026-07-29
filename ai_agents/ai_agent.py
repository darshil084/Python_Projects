import re
import urllib.parse
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import time

SEARCH_RESULTS = 6
PASSAGES_PER_PAGE = 4   # fixed spelling
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_PASSAGES = 5
SUMMARY_SENTENCES = 3
TIMEOUT = 8


def unwrap_ddg(url):
    """If DuckDuckGo returns a redirect wrapper, extract the real URL."""

    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uudg = qs.get("uudg")
            if uudg:
                return urllib.parse.unquote(uudg[0])
    except Exception:
        pass
    return url


def web_search(query, max_results=SEARCH_RESULTS):
    """Search the web and return a list of URLs."""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url")
            if not url:
                continue
            url = unwrap_ddg(url)
            urls.append(url)
    return urls


def search_web(query, max_results=SEARCH_RESULTS):
    """Search the web and return a list of URLs."""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url")
            if not url:
                continue
            url = unwrap_ddg(url)
            urls.append(url)
    return urls


def fetch_text(url, timeout=TIMEOUT):
    """Fetch and clean text content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if r.status_code != 200:
            return ""
        ct = r.headers.get("content-type", "")
        if "html" not in ct.lower():
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
            tag.extract()

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])

        if text.strip():
            return re.sub(r"\s+", " ", text).strip()

        meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if meta and meta.get("content"):
            return meta["content"].strip()
        if soup.title and soup.title.string:
            return soup.title.string.strip()

    except Exception:
        return ""
    return ""


def chunk_passages(text, max_words=120):   # fixed param name
    """Split text into passages of approximately max_words words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks


def split_sentences(text):
    """A simple sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


class ShortResearchAgent:   # fixed class name
    def __init__(self, embed_model=EMBEDDING_MODEL):
        print(f"Loading embedder: {embed_model}...")
        self.embedder = SentenceTransformer(embed_model)

    def run(self, query):
        start = time.time()

        # 1. Search
        urls = search_web(query)
        print(f"Found {len(urls)} urls.")

        # 2. Fetch & Chunk
        docs = []
        for u in urls:
            txt = fetch_text(u)
            if not txt:
                continue
            chunks = chunk_passages(txt, max_words=120)
            for c in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url": u, "passage": c})

        if not docs:
            print("No documents fetched.")
            return {"query": query, "passages": [], "summary": ""}

        # 3. Embed
        texts = [d["passage"] for d in docs]
        emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]

        # 4. Rank
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        sims = [cosine(e, q_emb) for e in emb_texts]
        top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]

        top_passages = [
            {
                "url": docs[i]["url"],
                "passage": docs[i]["passage"],
                "score": float(sims[i])
            }
            for i in top_idx
        ]

        # 5. Summarize
        sentences = []
        for tp in top_passages:
            for s in split_sentences(tp["passage"]):
                sentences.append({"sent": s, "url": tp["url"]})

        if not sentences:
            summary = "No summary could be generated."
        else:
            sent_texts = [s["sent"] for s in sentences]
            sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True, show_progress_bar=False)
            sent_sims = [cosine(e, q_emb) for e in sent_embs]

            top_sent_idx = np.argsort(sent_sims)[::-1][:SUMMARY_SENTENCES]
            chosen = [sentences[idx] for idx in top_sent_idx]

            seen = set()
            lines = []
            for s in chosen:
                key = s["sent"].lower()[:80]
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{s['sent']} (Source: {s['url']})")

            summary = " ".join(lines)

        elapsed = time.time() - start
        return {
            "query": query,
            "passages": top_passages,
            "summary": summary,
            "time": elapsed
        }


if __name__ == "__main__":
    agent = ShortResearchAgent()
    q = "benifits of masters in germany over india"

    print(f"Running query: {q}\n")
    out = agent.run(q)

    print("\nTop passages:")
    for p in out["passages"]:
        print(f"- score {p['score']:.3f} src {p['url']}\n  {p['passage'][:200]}...\n")

    print("--- Extractive summary ---")
    print(out["summary"])
    print("--------------------------")
    print(f"\nDone in {out['time']:.1f}s")
