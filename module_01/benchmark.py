
import json
import time
from urllib.request import Request, urlopen

MODEL = "lfm3-vl:2b"
URL = "http://localhost:11434/api/generate"

payload = {
    "model": MODEL,
    "prompt": "Explain why the sky is blue in two sentences.",
    "stream": False,
    "keep_alive": "5m",
    "options": {"temperature": 0, "num_predict": 60},
}

for run in range(1, 3):
    request = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urlopen(request, timeout=180) as response:
        result = json.load(response)

    wall_time = time.perf_counter() - start

    print(f"\nRun {run}")
    print(f"Client wall time: {wall_time:.3f} s")

    for metric in (
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ):
        print(f"{metric}: {result[metric] / 1e9:.3f} s")

    print(f"Input tokens: {result.get('prompt_eval_count', 0)}")
    print(f"Output tokens: {result.get('eval_count', 0)}")
