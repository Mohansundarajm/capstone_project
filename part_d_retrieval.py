"""Part D tasks 1-4: policy chunking, TF-IDF retrieval, grounded answers via the mock LLM, evaluation."""
import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = os.path.join("data", "policies")
FALLBACK_PREFIX = "I could not find"
POLICY_SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for Instamart support. Answer ONLY from the CONTEXT supplied in the user "
    "message. If the context does not contain the answer, reply exactly: "
    "I could not find that information in the policy documents."
)


def chunk_words(text, size=40, overlap=10):
    """Word windows: first covers words 0..size-1, next starts size-overlap later; stop once a chunk reaches the end."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_chunks(policy_dir=POLICY_DIR, size=40, overlap=10):
    out = []
    for path in sorted(glob.glob(os.path.join(policy_dir, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for i, text in enumerate(chunk_words(f.read(), size, overlap)):
                out.append({"source": name, "chunk_id": f"{name}#{i}", "text": text})
    return out


class PolicyIndex:
    """TF-IDF index over policy chunks; retrieve() ranks by cosine similarity."""

    def __init__(self, policy_dir=POLICY_DIR, size=40, overlap=10):
        self.chunks = load_chunks(policy_dir, size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]}
                for i in order]


_default = None


def get_index():
    global _default
    if _default is None:
        _default = PolicyIndex()
    return _default


def retrieve(query, k=3):
    return get_index().retrieve(query, k)


def answer(question, index=None):
    """Retrieve 3 chunks, ask the mock POLICY ASSISTANT, return (text, sources)."""
    hits = (index or get_index()).retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    text = mock_llm.chat(messages).choices[0].message.content
    return text, [h["source"] for h in hits]


def evaluate(overlap=10, eval_path="data/raw/policy_eval.csv", verbose=False):
    index = PolicyIndex(overlap=overlap)
    df = pd.read_csv(eval_path, keep_default_na=False)
    doc_hit = fact_hit = fallback = n_ans = n_unans = 0
    failures = []
    for r in df.itertuples():
        text, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            n_unans += 1
            if text.startswith(FALLBACK_PREFIX):
                fallback += 1
            else:
                failures.append((r.question, "expected fallback", text))
            continue
        n_ans += 1
        d, f = r.expected_doc in sources, r.expected_fact in text
        doc_hit += d
        fact_hit += f
        if not (d and f):
            failures.append((r.question, f"doc_ok={d} fact_ok={f}", text))
    return {"chunks": len(index.chunks), "answerable": n_ans, "doc_in_top3": doc_hit, "fact_in_answer": fact_hit,
            "unanswerable": n_unans, "fallback": fallback, "failures": failures}


def main():
    idx = PolicyIndex()
    per_doc = pd.Series([c["source"] for c in idx.chunks]).value_counts().sort_index().to_dict()
    print(f"chunks (size 40, overlap 10): {len(idx.chunks)}  per document: {per_doc}")
    top = retrieve("refund policy late delivery")
    print("query 'refund policy late delivery' top-3 sources:", [h["source"] for h in top])
    for ov in (10, 0):
        r = evaluate(overlap=ov)
        print(f"\noverlap={ov}: chunks={r['chunks']} | doc in top-3 {r['doc_in_top3']}/{r['answerable']} | "
              f"fact in answer {r['fact_in_answer']}/{r['answerable']} | fallback {r['fallback']}/{r['unanswerable']}")
        for q, why, text in r["failures"]:
            print(f"  FAIL: {q} -> {why} | answer: {text}")


if __name__ == "__main__":
    main()
