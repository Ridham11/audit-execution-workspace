from flask import Flask, request, jsonify
from services.groq_client import GroqClient
from services.chroma_service import ChromaService
import time
import hashlib
import uuid
import threading
import requests

app = Flask(__name__)

# 🔹 Services
groq = GroqClient()
chroma = ChromaService()

# 🔹 Metrics
APP_START_TIME = time.time()
RESPONSE_TIMES = []

# 🔹 Cache
CACHE = {}
CACHE_TTL = 900
CACHE_HITS = 0
CACHE_MISSES = 0

# 🔹 Jobs
JOBS = {}

#  Fallback (Day 13)
FALLBACK_RESPONSE = "We are facing temporary issues. Please try again later."

# 🔹 Seed Data
chroma.add_text("Unauthorized transaction detected", "1")
chroma.add_text("Payment failed due to network error", "2")
chroma.add_text("Payment stuck but money deducted", "3")
chroma.add_text("App crashes during login due to server timeout", "4")
chroma.add_text("Account blocked due to suspicious activity", "5")


@app.route('/')
def home():
    return "API is running"


# 🔹 Utility
def track_response_time(start_time):
    global RESPONSE_TIMES
    duration = (time.time() - start_time) * 1000
    RESPONSE_TIMES.append(duration)
    if len(RESPONSE_TIMES) > 10:
        RESPONSE_TIMES.pop(0)
    return duration


# 🔹 BACKGROUND JOB
def process_report(job_id, text):
    try:
        prompt = f"""
Analyze the following text and return:
- summary
- risks
- recommendations

Text:
{text}
"""
        response = groq.generate_response(prompt)
        answer = response.strip()

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = answer

        # 🔹 Webhook
        webhook_url = JOBS[job_id].get("webhook_url")
        if webhook_url:
            try:
                requests.post(webhook_url, json={
                    "job_id": job_id,
                    "status": "completed",
                    "result": answer
                })
            except Exception as e:
                print(f"[WEBHOOK FAILED] {e}")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


# 🔹 QUERY (WITH FALLBACK)
@app.route('/query', methods=['POST'])
def query():
    global CACHE_HITS, CACHE_MISSES

    start_time = time.time()
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question'"}), 400

    question = data["question"]
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
                    "cached": True,
                    "is_fallback": False
                }
            })

    CACHE_MISSES += 1

    results = chroma.query_with_docs(question)
    documents = results.get("documents", []) if results else []

    # 🔹 RELEVANCE FILTER
    question_words = set(question.lower().split())
    documents = [
        doc for doc in documents
        if question_words.intersection(set(doc.lower().split()))
    ]

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
                "cached": False,
                "is_fallback": False
            }
        })

    context = "\n".join(documents)

    prompt = f"""
You MUST answer strictly using ONLY the context.

Rules:
- ONE short line only
- No extra info

Context:
{context}

Question:
{question}
"""

    #  FALLBACK LOGIC
    is_fallback = False

    try:
        response = groq.generate_response(prompt)
        answer = response.strip().split("\n")[0]

    except Exception as e:
        print("Groq error:", str(e))
        answer = FALLBACK_RESPONSE
        is_fallback = True

    CACHE[cache_key] = {
        "answer": answer,
        "sources": [documents[0]],
        "timestamp": time.time()
    }

    duration = track_response_time(start_time)

    return jsonify({
        "answer": answer,
        "sources": [documents[0]],
        "meta": {
            "confidence": 0.9,
            "model_used": groq.model,
            "tokens_used": 0,
            "response_time_ms": round(duration, 2),
            "cached": False,
            "is_fallback": is_fallback
        }
    })


# 🔹 GENERATE REPORT
@app.route('/generate-report', methods=['POST'])
def generate_report():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text'"}), 400

    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "status": "processing",
        "result": None,
        "webhook_url": data.get("webhook_url")
    }

    threading.Thread(target=process_report, args=(job_id, data["text"])).start()

    return jsonify({
        "job_id": job_id,
        "status": "processing"
    })


# 🔹 JOB STATUS
@app.route('/job-status/<job_id>', methods=['GET'])
def job_status(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Invalid job_id"}), 404
    return jsonify(JOBS[job_id])


# 🔹 HEALTH
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "uptime": int(time.time() - APP_START_TIME),
        "cache_size": len(CACHE),
        "jobs": len(JOBS)
    })


if __name__ == '__main__':
    app.run(debug=True)