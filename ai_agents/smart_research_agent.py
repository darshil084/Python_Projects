import re
import time
import urllib.parse
import requests
import numpy as np

from ddgs import DDGS
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


# ================= CONFIG =================

SEARCH_RESULTS = 8
PASSAGES_PER_PAGE = 3
TOP_PASSAGES = 4
SUMMARY_SENTENCES = 7

TIMEOUT = 8
MAX_WORDS = 120
MAX_WORKERS = 6

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =============== UTILITIES ===============

def unwrap_ddg(url):

    try:
        p = urllib.parse.urlparse(url)

        if "duckduckgo.com" in p.netloc:

            q = urllib.parse.parse_qs(p.query)

            if "uudg" in q:
                return urllib.parse.unquote(q["uudg"][0])
    except:
        pass

    return url


def improve_query(query):

    extra = " Germany masters university comparison India benefits jobs fees visa"

    return query + extra


def search_web(query):

    urls = []

    query = improve_query(query)

    with DDGS() as ddgs:

        for r in ddgs.text(query, max_results=SEARCH_RESULTS):

            url = r.get("href") or r.get("url")

            if url:
                urls.append(unwrap_ddg(url))

    return urls


def remove_duplicate_domains(urls):

    seen = set()
    clean = []

    for u in urls:

        domain = urllib.parse.urlparse(u).netloc

        if domain not in seen:
            seen.add(domain)
            clean.append(u)

    return clean


def fetch_text(url):

    headers = {"User-Agent": "Mozilla/5.0"}

    try:

        r = requests.get(url, timeout=TIMEOUT, headers=headers)

        if r.status_code != 200:
            return ""

        if "html" not in r.headers.get("content-type", ""):
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "iframe", "svg", "aside"
        ]):
            tag.extract()

        paragraphs = soup.find_all("p")

        text = " ".join(p.get_text() for p in paragraphs)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    except:
        return ""


def chunk_text(text):

    words = text.split()

    chunks = []

    for i in range(0, len(words), MAX_WORDS):

        part = words[i:i + MAX_WORDS]

        chunks.append(" ".join(part))

    return chunks


def split_sentences(text):

    return re.split(r'(?<=[.!?])\s+', text)


def is_relevant(text, query):

    keywords = query.lower().split()

    score = 0

    t = text.lower()

    for k in keywords:

        if k in t:
            score += 1

    return score >= 2


def cosine(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# ============== MAIN AGENT ==============

class SmartResearchAgent:


    def __init__(self):

        print("Loading embedding model...")

        self.embedder = SentenceTransformer(MODEL_NAME)

        self.cache = {}


    def fetch_all(self, urls):

        docs = []

        with ThreadPoolExecutor(MAX_WORKERS) as pool:

            results = pool.map(fetch_text, urls)

            for url, text in zip(urls, results):

                if not text:
                    continue

                chunks = chunk_text(text)

                for c in chunks[:PASSAGES_PER_PAGE]:

                    if is_relevant(c, self.current_query):

                        docs.append({
                            "url": url,
                            "text": c
                        })

        return docs


    def run(self, query):

        start = time.time()

        self.current_query = query


        # -------- CACHE --------

        if query in self.cache:

            print("Using cached result ⚡")

            return self.cache[query]


        # -------- SEARCH --------

        print("Searching...")

        urls = search_web(query)

        urls = remove_duplicate_domains(urls)

        print("Sources found:", len(urls))


        # -------- FETCH --------

        print("Fetching pages...")

        docs = self.fetch_all(urls)

        if not docs:

            print("No useful data found.")
            return None


        texts = [d["text"] for d in docs]


        # -------- EMBEDDING --------

        print("Embedding text...")

        text_emb = self.embedder.encode(texts)

        q_emb = self.embedder.encode([query])[0]


        # -------- RANKING --------

        scores = [cosine(e, q_emb) for e in text_emb]

        idx = np.argsort(scores)[::-1][:TOP_PASSAGES]


        top = []

        for i in idx:

            top.append({
                "url": docs[i]["url"],
                "text": docs[i]["text"],
                "score": float(scores[i])
            })


        # -------- SUMMARY --------

        sentences = []

        for t in top:

            for s in split_sentences(t["text"]):

                if len(s) > 40:
                    sentences.append(s)


        sent_emb = self.embedder.encode(sentences)

        sent_scores = [cosine(e, q_emb) for e in sent_emb]

        best = np.argsort(sent_scores)[::-1][:SUMMARY_SENTENCES]


        summary = " ".join(sentences[i] for i in best)


        # -------- RESULT --------

        result = {
            "query": query,
            "top_passages": top,
            "summary": summary,
            "time": round(time.time() - start, 2)
        }


        self.cache[query] = result


        return result


# ================= RUN =================

if __name__ == "__main__":

    agent = SmartResearchAgent()


    q = "benefits of masters in germany over india"


    print("\nQuery:", q)


    out = agent.run(q)


    if out:

        print("\n----- SUMMARY -----\n")

        print(out["summary"])


        print("\nTime:", out["time"], "sec")
