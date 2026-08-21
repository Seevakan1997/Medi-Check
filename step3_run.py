"""
"""

import argparse, os, json, re, sys, torch, requests
import torch.nn as nn, requests, chromadb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

MODEL_NAME   = "distilbert-base-uncased"
MODEL_PATH   = "model.pt"
DB_PATH      = "chroma_db"
COLLECTION   = "healthcare_misinfo"
EMBED_MODEL  = "all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
MAX_LEN      = 256
TOP_K        = 4
MIN_SIM      = 0.40
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

class MisinfoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)
        hidden          = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

def load_dl_model():
    """Load the trained BioBERT classifier."""
    if not __import__("os").path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run step1_train.py first.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = MisinfoClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def dl_score(text, model, tokenizer):
    """
    Returns:
        score_misinfo  : float (0-1) probability of misinformation
        score_credible : float (0-1) probability of credible
        label          : "misinformation" or "credible"
    """
    enc = tokenizer(
        text, truncation=True, max_length=MAX_LEN,
        return_tensors="pt", padding=True
    )
    enc    = {k: v.to(DEVICE) for k, v in enc.items()}
    probs  = torch.softmax(model(**enc), dim=-1)[0].cpu().tolist()
    label  = "misinformation" if probs[1] > 0.5 else "credible"
    return {
        "score_misinfo":  round(probs[1], 4),
        "score_credible": round(probs[0], 4),
        "label":          label,
    }


def load_rag():
    """Load the ChromaDB collection."""
    if not __import__("os").path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run step2_build_rag.py first.")
        sys.exit(1)
    client     = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION)
    encoder    = SentenceTransformer(EMBED_MODEL)
    return collection, encoder


# def rag_retrieve(text, collection, encoder):
    """
    Searches the knowledge base and returns top-K similar entries.
    Each hit contains: id, claim, verdict, source, url, rationale, similarity
    """
    query_emb = encoder.encode(text).tolist()
    results   = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        similarity = round(1.0 - results["distances"][0][i], 4)
        if similarity < MIN_SIM:
            continue
        meta = results["metadatas"][0][i]
        hits.append({
            "id":         doc_id,
            "claim":      results["documents"][0][i],
            "verdict":    meta.get("verdict",   ""),
            "source":     meta.get("source",    ""),
            "url":        meta.get("url",       ""),
            "rationale":  meta.get("rationale", ""),
            "similarity": similarity,
        })
    return hits
def rag_retrieve(text, collection, encoder):
    try:
        print("      Encoding query...")
        query_emb = encoder.encode(text).tolist()
        print("      Querying collection...")
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        print("      Query done.")
        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            similarity = round(1.0 - results["distances"][0][i], 4)
            if similarity < MIN_SIM:
                continue
            meta = results["metadatas"][0][i]
            hits.append({
                "id":        doc_id,
                "claim":     results["documents"][0][i],
                "verdict":   meta.get("verdict",   ""),
                "source":    meta.get("source",    ""),
                "url":       meta.get("url",       ""),
                "rationale": meta.get("rationale", ""),
                "similarity": similarity,
            })
        return hits
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

def format_rag_context(hits):
    """Formats RAG hits into a readable block for the LLM prompt."""
    if not hits:
        return "No similar known misinformation patterns found in the knowledge base."
    lines = ["Known misinformation reference entries from WHO/NHS/CDC:"]
    for i, h in enumerate(hits, 1):
        claim = h["claim"][:300]
        source = h["source"][:120]
        rationale = h["rationale"][:400]
        lines.append(
            f"\n[{i}] Claim    : {claim}\n"
            f"    Verdict  : {h['verdict']}\n"
            f"    Source   : {source}\n"
            f"    Rationale: {rationale}\n"
            f"    Similarity to input: {h['similarity']:.0%}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert healthcare misinformation analyst with knowledge of WHO, NHS, and CDC guidelines up to 2024.

Return ONLY a JSON object. No markdown, no preamble.

{
  "verdict": "misinformation" | "credible" | "uncertain",
  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence explanation",
  "recommendation": "what should be done"
}

SCORE-BASED DECISION RULES:
Score 0-35%   → credible,       high confidence
Score 36-55%  → credible,       medium confidence
Score 56-70%  → misinformation, medium confidence
Score 71-100% → misinformation, high confidence

HARD OVERRIDE LIST — these ALWAYS override the DL score:

ALWAYS MISINFORMATION high:
- Vaccines cause autism
- Bleach or disinfectant cures COVID
- 5G causes or spreads COVID-19
- Antibiotics treat viruses, flu, or COVID
- COVID vaccines alter DNA
- COVID vaccines contain microchips
- Homeopathy cures cancer
- HIV spreads through casual contact or toilet seats
- Ivermectin is proven to cure COVID

ALWAYS CREDIBLE high:
- COVID-19 is airborne or spreads through aerosols (WHO confirmed 2021)
- Washing hands prevents infection
- Exercise reduces heart disease risk
- Smoking causes cancer
- Vaccines are safe and effective
- Sleep is important for health and immunity
- Fruits and vegetables are good for health
- Social distancing reduces virus spread
- Masks reduce respiratory virus transmission

DOCTOR STATEMENT RULE:
- Evaluate the claim itself, not who said it
- "Dr X said COVID is airborne" → evaluate "COVID is airborne" → CREDIBLE

UNCERTAINTY RULE:
- Only use uncertain if genuinely impossible to classify
- Target: uncertain in less than 5% of cases
- Never use uncertain for well-known facts or well-known myths"""


def fallback_decision(dl_result, reason):
    """Use the classifier score when the optional local LLM is unavailable."""
    score = dl_result["score_misinfo"]
    verdict = "misinformation" if score >= 0.56 else "credible"
    confidence = "medium" if score >= 0.71 or score <= 0.35 else "low"
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": f"{reason} The result is based on the DL classifier score.",
        "recommendation": "Review this result with current guidance from a qualified health professional.",
    }


def llm_decide(text, dl_result, rag_context):
    """
    Sends claim + DL score + RAG context to Ollama LLaMA.
    Returns structured verdict dict.
    """
    prompt = f"""Claim: \"{text}\"

DL Score: {dl_result['score_misinfo']:.0%} misinformation probability

{rag_context}

Check the hard override list first. If the claim matches, use that verdict.
Otherwise apply the score-based rules.
Return JSON only."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "system":  SYSTEM_PROMPT,
                "stream":  False,
                "format":  "json",
                "options": {"temperature": 0.1, "num_predict": 128},
            },
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        print("\n" + "="*50)
        print("ERROR: Cannot connect to Ollama.")
        print("Fix: Open a new terminal and run:")
        print("     ollama serve")
        print("     ollama pull llama3")
        print("="*50)
        return fallback_decision(dl_result, "Ollama is unavailable.")
    except requests.exceptions.Timeout:
        print(f"LLM timed out after {OLLAMA_TIMEOUT:g}s; using the DL score fallback.")
        return fallback_decision(dl_result, "The local language model did not respond in time.")

    
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

    return {
        "verdict":        "uncertain",
        "confidence":     "low",
        "reasoning":      "LLM response could not be parsed. Manual review required.",
        "recommendation": "Flag for expert review.",
        "raw":            raw,
    }

def build_provenance(text, dl_result, rag_hits, llm_result):
    """
    Builds a full audit trail for every analysis.
    This satisfies UK GDPR Article 5(2) and ISO/IEC 27001 audit requirements.
    """
    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "input_text":         text,
        "dl_label":           dl_result["label"],
        "dl_score_misinfo":   dl_result["score_misinfo"],
        "dl_score_credible":  dl_result["score_credible"],
        "rag_hits":           rag_hits,
        "rag_hits_count":     len(rag_hits),
        "llm_verdict":        llm_result.get("verdict",        "uncertain"),
        "llm_confidence":     llm_result.get("confidence",     "low"),
        "llm_reasoning":      llm_result.get("reasoning",      ""),
        "llm_recommendation": llm_result.get("recommendation", ""),
        "final_verdict":      llm_result.get("verdict",        "uncertain"),
        "cited_sources": [
            {
                "source":     h["source"],
                "url":        h["url"],
                "claim":      h["claim"],
                "similarity": h["similarity"],
            }
            for h in rag_hits if h["similarity"] > 0.4
        ],
    }


ICONS = {"misinformation": "✗", "credible": "✓", "uncertain": "?"}
COLORS = {"misinformation": "\033[91m", "credible": "\033[92m",
          "uncertain": "\033[93m", "reset": "\033[0m"}

def print_result(prov):
    v     = prov["final_verdict"]
    icon  = ICONS.get(v, "?")
    color = COLORS.get(v, "")
    reset = COLORS["reset"]

    print("\n" + "═"*60)
    print(f"  {color}{icon} VERDICT      : {v.upper()}{reset}")
    print(f"    Confidence   : {prov['llm_confidence'].upper()}")
    print(f"    DL Score     : {prov['dl_score_misinfo']:.0%} misinformation probability")
    print(f"    Timestamp    : {prov['timestamp']}")
    print(f"\n  REASONING:")
    print(f"    {prov['llm_reasoning']}")
    print(f"\n  RECOMMENDATION:")
    print(f"    {prov['llm_recommendation']}")

    if prov["cited_sources"]:
        print(f"\n  CITED SOURCES ({len(prov['cited_sources'])}):")
        for i, s in enumerate(prov["cited_sources"], 1):
            print(f"    [{i}] {s['source']}")
            print(f"         {s['url']}")
            print(f"         Match: \"{s['claim'][:65]}...\"")
            print(f"         Similarity: {s['similarity']:.0%}")
    else:
        print(f"\n  CITED SOURCES: No close matches found in knowledge base")

    print(f"\n  RAG HITS      : {prov['rag_hits_count']} entries retrieved")
    print("═"*60 + "\n")


def run_pipeline(text, model, tokenizer, collection, encoder):
    """
    Runs the full 3-layer pipeline on a single text input.
    Returns the full provenance record.
    """
    print(f"\n  Analysing: \"{text[:70]}{'...' if len(text)>70 else ''}\"")

    print("  [1/3] Running DL classifier (BioBERT)...")
    dl_result = dl_score(text, model, tokenizer)
    print(f"        Score: {dl_result['score_misinfo']:.0%} misinfo | {dl_result['score_credible']:.0%} credible")

    print("  [2/3] Searching RAG knowledge base...")
    rag_hits    = rag_retrieve(text, collection, encoder)
    rag_context = format_rag_context(rag_hits)
    print(f"        Found {len(rag_hits)} similar entries")

    print("  [3/3] LLM reasoning (Ollama LLaMA)...")
    llm_result = llm_decide(text, dl_result, rag_context)

    prov = build_provenance(text, dl_result, rag_hits, llm_result)
    return prov


def main():
    parser = argparse.ArgumentParser(description="Healthcare Misinformation Detector")
    parser.add_argument("--text", default=None, help="Claim to analyse")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    print("Loading DL model...")
    model, tokenizer = load_dl_model()
    print("Loading RAG index...")
    collection, encoder = load_rag()
    print("Ready.\n")

    if args.text:
        prov = run_pipeline(args.text, model, tokenizer, collection, encoder)
        if args.json:
            print(json.dumps(prov, indent=2))
        else:
            print_result(prov)
    else:
        print("Healthcare Misinformation Detector — Interactive Mode")
        print("Type a health claim and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                text = input("Enter claim:\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            prov = run_pipeline(text, model, tokenizer, collection, encoder)
            print_result(prov)


if __name__ == "__main__":
    main()
