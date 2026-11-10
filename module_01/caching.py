
import json
from urllib.request import Request, urlopen

MODEL = "qwen3.5:2b"
# MODEL = "lfm3-vl:2b"
prefix = "Solar panels convert sunlight to electricity. " * 40

prompts = [
    ("Original", "A: " + prefix + "Reply OK."),
    ("Changed end", "A: " + prefix + "Reply YES."),
    ("Changed start", "B: " + prefix + "Reply OK."),
]

# prompts = [
#     ("Original", "A: " + prefix + "Reply OK."),
#     ("Changed start", "B: " + prefix + "Reply OK."),
#     ("Original again", "A: " + prefix + "Reply OK."),
# ]

# prompt = "A: " + prefix + "Reply OK."

# prompts = [
#     ("First request", prompt),
#     ("Identical request 2", prompt),
#     ("Identical request 3", prompt),
# ]

for name, prompt in prompts:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "raw": True,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 1},
    }

    req = Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req, timeout=180) as response:
        result = json.load(response)

    print(f"\n{name}")
    print("Input tokens:", result.get("prompt_eval_count"))
    print("Cached tokens:", result.get(
        "prompt_eval_cached_count", "not reported"))
    print("Prefill time:",
          round(result["prompt_eval_duration"] / 1e9, 3), "s")
