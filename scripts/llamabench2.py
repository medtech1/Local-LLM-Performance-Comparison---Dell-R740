import requests
import time
import random
import string

def get_llama_info():
    model = "Unknown Model"
    n_ctx = "Unknown"
    base_url = "http://localhost:8080"
    try:
        m_resp = requests.get(f"{base_url}/v1/models", timeout=2).json()
        if "data" in m_resp and len(m_resp["data"]) > 0:
            full_path = m_resp["data"][0].get("id", "Unknown")
            model = full_path.split("/")[-1]
    except:
        pass
    try:
        p_resp = requests.get(f"{base_url}/props", timeout=2).json()
        n_ctx = p_resp.get("default_generation_settings", {}).get("n_ctx", "Unknown")
        if n_ctx == "Unknown":
            n_ctx = p_resp.get("n_ctx", "Unknown")
    except:
        pass
    return model, n_ctx

def benchmark_llama(num_iterations=5):
    url = "http://localhost:8080/completion"
    model_name, n_ctx = get_llama_info()
    base_pp_prompt = "word " * 390
    tg_prompt = "Explain quantum computing in exactly one sentence."
    print(f"Benchmarking Model: {model_name}")
    print(f"Context Size (n_ctx): {n_ctx}")
    print(f"Mode: pp512 & tg128 | Iterations: {num_iterations}")
    print("Warming up GPU kernels...", end="", flush=True)
    warmup_payload = {"prompt": "warmup", "n_predict": 16, "stream": False, "cache_prompt": False}
    try:
        requests.post(url, json=warmup_payload)
        print(" Done.")
    except:
        print(" Failed (Server not responding).")
        return
    print("-" * 60)
    results = {"pp512": [], "tg128": [], "ttft": []}
    for i in range(num_iterations):
        salt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        unique_pp_prompt = f"{base_pp_prompt} [ID: {salt}]"
        pp_payload = {"prompt": unique_pp_prompt, "n_predict": 1,
                      "temperature": 0.0, "cache_prompt": False, "stream": False}
        resp_pp = requests.post(url, json=pp_payload).json()
        t_pp = resp_pp.get("timings", {})
        tg_payload = {"prompt": tg_prompt, "n_predict": 128,
                      "temperature": 0.0, "stream": False}
        resp_tg = requests.post(url, json=tg_payload).json()
        t_tg = resp_tg.get("timings", {})
        pp_speed = t_pp.get("prompt_per_second", 0.0)
        tg_speed = t_tg.get("predicted_per_second", 0.0)
        ttft_ms = t_pp.get("prompt_ms", 0.0)
        results["pp512"].append(pp_speed)
        results["tg128"].append(tg_speed)
        results["ttft"].append(ttft_ms)
        status = " (Ignoring for avg)" if i == 0 else ""
        print(f"Run {i+1:02d}: pp512 = {pp_speed:>8.2f} t/s | "
              f"tg128 = {tg_speed:>6.2f} t/s | TTFT = {ttft_ms:>7.2f} ms{status}")
        time.sleep(1)
    def calc_avg(k):
        valid_data = results[k][1:]
        return sum(valid_data) / len(valid_data) if valid_data else 0
    print("\n" + "=" * 60)
    print(f"FINAL AVERAGES (Excluding Iteration 1) - {model_name}")
    print("-" * 60)
    print(f"pp512 (Prompt Processing):      {calc_avg('pp512'):.2f} tokens/s")
    print(f"tg128 (Token Generation):        {calc_avg('tg128'):.2f} tokens/s")
    print(f"Average Latency (TTFT):          {calc_avg('ttft'):.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_llama(num_iterations=5)
