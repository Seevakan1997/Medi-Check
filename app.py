"""
app.py - MedCheck Flask API
Run: python app.py
Open: http://127.0.0.1:5000

Flow:
    Input
        → LLM pre-filter (is this a real health claim?)
        → DL classifier (risk score)
        → RAG retrieval (WHO/NHS/CDC evidence)
        → LLM verdict (final decision)
        → Provenance record
"""

import sys, os, re, json
from matplotlib import text
import requests as req
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from step3_run import load_dl_model, dl_score, llm_decide, build_provenance, load_rag,rag_retrieve

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Load DL model once at startup ──────────────────────────────────────────
print("Starting MedCheck API server...")

print("Loading DL model...")
MODEL, TOKENIZER = load_dl_model()

print("Loading RAG database...")
RAG_COLLECTION, RAG_ENCODER = load_rag()

print("RAG database loaded successfully.")
print(f"RAG entries: {RAG_COLLECTION.count()}")

print("Ready.\n")

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# ══════════════════════════════════════════════════════════════════════════
# LLM PRE-FILTER
# Checks if the input is a valid, meaningful health claim before
# running the expensive DL + RAG + LLM pipeline.
# ══════════════════════════════════════════════════════════════════════════

PRE_FILTER_SYSTEM = """You are a healthcare claim validator.

Decide whether the input is related to health, medicine, food, diet, disease, treatment, or wellbeing.

Return ONLY a JSON object.

{
  "is_health_claim": true or false,
  "reason": "one sentence explanation"
}

Rules:
- true  → anything about health, medicine, food, diet, disease, body, treatment, vitamins, symptoms, doctors, viruses, vaccines, remedies, wellbeing
          Examples: "an apple a day keeps the doctor away", "apple cider vinegar is good for stomach",
          "Dr John said coronavirus is airborne", "vaccines cause autism", "bleach cures COVID"
- false → completely unrelated to health: random words, greetings, gibberish, maths, geography
          Examples: "ji", "hello", "2+2=4", "the capital of France"
- When in doubt about health relevance → return TRUE
- Food and diet claims are ALWAYS health claims → return TRUE
- Doctor or medical professional statements are ALWAYS health claims → return TRUE
- Random letter combinations, keyboard mashing, or strings with no real words → always FALSE
- If you cannot identify a clear health topic in the input → return FALSE
- Strings like "sdnlnsdldfs", "asdfgh", "xyzabc" → always FALSE
"""

# PRE_FILTER_SYSTEM = """You are a healthcare claim validator.

# Decide whether the input is related to health, medicine, food, diet, disease, treatment, or wellbeing.

# Return ONLY a JSON object.

# {
#   "is_health_claim": true or false,
#   "reason": "one sentence explanation"
# }

# Rules:
# - true  → anything about health, medicine, food, diet, disease, body, treatment, vitamins, symptoms, doctors, viruses, vaccines, remedies, wellbeing
# - false → completely unrelated to health: random words, greetings, gibberish, maths, geography
# - When in doubt → return TRUE
# - Food and diet claims → always TRUE
# - Doctor or medical professional statements → always TRUE
# - Common health proverbs like "an apple a day" → always TRUE"""


def llm_pre_filter(text):
    """
    Returns (is_valid, reason)
    is_valid: True if text is a genuine health claim worth analysing
    """
    try:
        resp = req.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  f'Input to validate:\n"{text}"',
                "system":  PRE_FILTER_SYSTEM,
                "stream":  False,
                "options": {"temperature": 0.0, "num_predict": 100},
            },
            timeout=30
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group()) if match else {}

        is_valid = result.get("is_health_claim", True)
        reason   = result.get("reason", "")
        return is_valid, reason

    except Exception as e:
        print(f"Pre-filter error (skipping): {e}")
        # If Ollama fails, let the claim through
        return True, "Pre-filter unavailable"


# ══════════════════════════════════════════════════════════════════════════
# RAG (returns empty on Windows due to threading — safe fallback)
# ══════════════════════════════════════════════════════════════════════════

def get_rag_hits(text):
    try:
        print("      Encoding query...")
        
        hits = rag_retrieve(
            text,
            RAG_COLLECTION,
            RAG_ENCODER
        )

        print(f"      RAG returned {len(hits)} hits")

        for i, hit in enumerate(hits, 1):
            print(
                f"      [{i}] "
                f"{hit['similarity']:.1%} - "
                f"{hit['source']}"
            )

        return hits

    except Exception as e:
        import traceback
        print("RAG ERROR:")
        traceback.print_exc()
        return []


def format_context(hits):
    if not hits:
        return "No similar known misinformation patterns found."
    lines = ["Known misinformation reference entries from WHO/NHS/CDC:"]
    for i, h in enumerate(hits, 1):
        lines.append(
            f"\n[{i}] Claim: {h['claim']}\n"
            f"    Source: {h['source']}\n"
            f"    Rationale: {h['rationale']}\n"
            f"    Similarity: {h['similarity']:.0%}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/health", methods=["GET"])
def health():
    ollama_ok = False
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return jsonify({
        "status":    "ok",
        "dl_model":  os.path.exists("model.pt"),
        "rag_index": os.path.exists("chroma_db"),
        "ollama":    ollama_ok,
    })


@app.route("/api/analyse", methods=["POST"])
def analyse():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    if len(text) < 5:
        return jsonify({
            "rejected": True,
            "reason":   "Input too short to be a health claim."
        }), 200

    words = text.split()
    real_words = [w for w in words if re.match(r'^[a-zA-Z]{3,}$', w)]
    real_ratio = len(real_words) / max(len(words), 1)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

    if (real_ratio < 0.4 or avg_word_len > 12) and len(text) < 150:
        return jsonify({
            "rejected": True,
            "reason":   "Input does not appear to be a meaningful health claim."
        }), 200
    
    try:
        print(f"\n{'='*55}")
        print(f"Input: {text[:80]}")

        # ── PRE-FILTER ──────────────────────────────────────────────
        print("[0/3] LLM pre-filter...")
        is_valid, reason = llm_pre_filter(text)

        if not is_valid:
            print(f"      REJECTED — {reason}")
            return jsonify({
                "rejected": True,
                "reason":   reason or "This does not appear to be a healthcare claim.",
            }), 200

        print(f"      ACCEPTED — {reason}")

        # ── DL CLASSIFIER ───────────────────────────────────────────
        print("[1/3] DL classifier...")
        dl_result = dl_score(text, MODEL, TOKENIZER)
        print(f"      {dl_result['score_misinfo']:.0%} misinfo")

        # ── RAG ─────────────────────────────────────────────────────
        print("[2/3] RAG retrieval...")
        rag_hits    = get_rag_hits(text)
        rag_context = format_context(rag_hits)
        print(f"      {len(rag_hits)} hits found")

        # ── LLM VERDICT ─────────────────────────────────────────────
        print("[3/3] LLM decision...")
        llm_result = llm_decide(text, dl_result, rag_context)
        print(f"      Verdict: {llm_result.get('verdict')}")

        prov = build_provenance(text, dl_result, rag_hits, llm_result)
        print("Done.")
        return jsonify(prov)

    except SystemExit:
        return jsonify({"error": "Ollama not running. Run: ollama serve"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:5000")
    print("API: POST http://127.0.0.1:5000/api/analyse\n")
    app.run(debug=False, host="0.0.0.0", port=5000,
            use_reloader=False, threaded=False)