
import json
import uuid
from urllib.request import Request, urlopen

MODEL = "qwen3.5:2b"
URL = "http://localhost:11434/api/generate"

sentence = "Solar panels convert sunlight into electricity. "

import random

sizes = [384, 512, 512, 384]  # ABBA

for n in sizes:
    # Existing benchmark code
    prompt = f"{uuid.uuid4().hex}\n" + sentence * n
    print(prompt[:60] + "...")

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "raw": True,
        "think": False,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_predict": 1,
            "temperature": 0,
            "num_ctx": 8192,
        },
    }

    request = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(request, timeout=180) as response:
        result = json.load(response)

    tokens = result["prompt_eval_count"]
    seconds = result["prompt_eval_duration"] / 1e9
    cached = result.get("prompt_eval_cached_count", 0)

    print(
        f"Tokens={tokens}, "
        f"Cached={cached}, "
        f"Prefill={seconds:.3f}s, "
        f"Throughput={tokens / seconds:.2f} tok/s"
    )
