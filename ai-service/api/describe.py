from flask import Flask, request, jsonify
from services.groq_client import GroqClient
from services.chroma_service import ChromaService
import json
import time
import hashlib

app = Flask(__name__)

# 🔹 Initialize services
groq = GroqClient()
chroma = ChromaService()

# 🔥 Day 7 — Tracking variables
APP_START_TIME = time.time()
RESPONSE_TIMES = []

# 🔥 Day 8 — Cache
CACHE = {}
CACHE_TTL = 900

CACHE_HITS = 0
CACHE_MISSES = 0


# 🔥 Initial data
chroma.add_text("Unauthorized transaction detected", "1")
chroma.add_text("Payment failed due to network error", "2")
chroma.add_text("Payment stuck but money deducted", "3")
chroma.add_text("App crashes during login due to server timeout", "4")
chroma.add_text("Account blocked due to suspicious activity", "5")


@app.route('/')
def home():
    return "API is running"


# 🔹 Utility function
def track_response_time(start_time):
    global RESPONSE_TIMES
    duration = (time.time() - start_time) * 1000
    RESPONSE_TIMES.append(duration)

    if len(RESPONSE_TIMES) > 10:
        RESPONSE_TIMES.pop(0)

    return duration


# 🔹 Categorise
@app.route('/categorise', methods=['POST'])
def categorise():
    start_time = time.time()

    data = request.get_json()

    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input'"}), 400

    user_input = data.get("input")

    try:
        prompt = f"""
Classify the following text into one of these categories:
fraud, finance, technical, general, complaint

Return ONLY JSON:
{{
  "category": "...",
  "confidence": 0.0,
  "reasoning": "..."
}}

Text: {user_input}
"""

        response = groq.generate_response(prompt)

        try:
            parsed = json.loads(response)
        except:
            parsed = {"raw_response": response}

        track_response_time(start_time)

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔥 FINAL QUERY
@app.route('/query', methods=['POST'])
def query():
    global CACHE_HITS, CACHE_MISSES

    start_time = time.time()

    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question'"}), 400

        question = data.get("question")
        fresh = data.get("fresh", False)

        cache_key = hashlib.sha256(question.encode()).hexdigest()

        # 🔹 CACHE HIT
        if not fresh and cache_key in CACHE:
            cached = CACHE[cache_key]

            if time.time() - cached["timestamp"] < CACHE_TTL:
                CACHE_HITS += 1
                duration = track_response_time(start_time)

                return jsonify({
                    "answer": cached["answer"],
                    "sources": cached["sources"],
                    "meta": {
                        "confidence": 0.95,
                        "model_used": groq.model,
                        "tokens_used": 0,
                        "response_time_ms": round(duration, 2),
                        "cached": True
                    }
                })

        # 🔹 CACHE MISS
        CACHE_MISSES += 1

        results = chroma.query_with_docs(question)

        documents = []
        if results and "documents" in results:
            documents = results["documents"]

        # 🔥 Relevance filter
        question_words = set(question.lower().split())

        relevant_docs = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            if question_words.intersection(doc_words):
                relevant_docs.append(doc)

        documents = relevant_docs

        # 🔹 NO DATA
        if not documents:
            duration = track_response_time(start_time)

            return jsonify({
                "answer": "No relevant data found",
                "sources": [],
                "meta": {
                    "confidence": 0.0,
                    "model_used": groq.model,
                    "tokens_used": 0,
                    "response_time_ms": round(duration, 2),
                    "cached": False
                }
            })

        context = "\n".join(documents)

        # 🔥 STRICT PROMPT
        prompt = f"""
You MUST answer strictly using ONLY the exact information from the context.

Rules:
- Return ONLY ONE short line
- Select the MOST relevant sentence
- DO NOT combine multiple lines
- DO NOT add any new information
- DO NOT infer anything
- If not found, return: "No relevant data found"

Context:
{context}

Question:
{question}
"""

        response = groq.generate_response(prompt)

        # 🔥 Force single line
        answer = response.strip().split("\n")[0]

        # 🔹 STORE CACHE (ONLY BEST SOURCE)
        CACHE[cache_key] = {
            "answer": answer,
            "sources": [documents[0]] if documents else [],
            "timestamp": time.time()
        }

        duration = track_response_time(start_time)

        return jsonify({
            "answer": answer,
            "sources": [documents[0]] if answer != "No relevant data found" and documents else [],
            "meta": {
                "confidence": 0.9,
                "model_used": groq.model,
                "tokens_used": 0,
                "response_time_ms": round(duration, 2),
                "cached": False
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔥 HEALTH
@app.route('/health', methods=['GET'])
def health():
    try:
        uptime = int(time.time() - APP_START_TIME)

        avg_response = (
            sum(RESPONSE_TIMES) / len(RESPONSE_TIMES)
            if RESPONSE_TIMES else 0
        )

        doc_count = chroma.get_count()

        return jsonify({
            "status": "healthy",
            "model": groq.model,
            "avg_response_time_ms": round(avg_response, 2),
            "last_10_responses": [round(t, 2) for t in RESPONSE_TIMES],
            "chroma_doc_count": doc_count,
            "uptime_seconds": uptime,
            "cache": {
                "hits": CACHE_HITS,
                "misses": CACHE_MISSES,
                "size": len(CACHE)
            }
        })

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)