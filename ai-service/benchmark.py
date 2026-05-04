import requests
import time
import statistics

BASE_URL = "http://127.0.0.1:5000"


def print_results(name, times):
    times.sort()

    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times))]
    p99 = times[int(0.99 * len(times))]

    print(f"\n{name} Results:")
    print(f"p50: {round(p50, 2)} ms")
    print(f"p95: {round(p95, 2)} ms")
    print(f"p99: {round(p99, 2)} ms")


def test_health():
    times = []
    print("\n🔹 Testing Health...")

    for _ in range(50):
        start = time.time()
        requests.get(f"{BASE_URL}/health")
        end = time.time()

        times.append((end - start) * 1000)

    print_results("Health", times)


def test_query():
    times = []
    print("\n🔹 Testing Query (NO CACHE)...")

    for i in range(50):
        start = time.time()

        payload = {
            "question": f"payment issue {i}",  # 🔥 unique each time
            "fresh": True                      # 🔥 bypass cache
        }

        requests.post(f"{BASE_URL}/query", json=payload)

        end = time.time()
        times.append((end - start) * 1000)

    print_results("Query", times)


def test_generate_report():
    times = []
    print("\n🔹 Testing Generate Report (ASYNC)...")

    for i in range(50):
        start = time.time()

        payload = {
            "text": f"fraud case {i}"
        }

        requests.post(f"{BASE_URL}/generate-report", json=payload)

        end = time.time()
        times.append((end - start) * 1000)

    print_results("Generate Report", times)


if __name__ == "__main__":
    print("\n🚀 STARTING PERFORMANCE BENCHMARK...\n")

    test_health()
    test_query()
    test_generate_report()

    print("\n✅ Benchmark Completed\n")