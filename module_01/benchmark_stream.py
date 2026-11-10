
import json
import time
from urllib.request import Request, urlopen

MODEL = "lfm3-vl:2b"
URL = "http://localhost:11434/api/generate"

payload = {
    "model": MODEL,
    "prompt": "Explain how solar panels generate electricity.",
    "stream": True,
    "think": False,
    "keep_alive": "5m",
    "options": {"temperature": 0, "num_predict": 100},
}

request = Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

arrival_times = []
start = time.perf_counter()

with urlopen(request, timeout=180) as response:
    for line in response:
        event = json.loads(line)
        if event.get("response"):
            arrival_times.append(time.perf_counter() - start)

        if event.get("done"):
            final = event
            break

end_to_end = time.perf_counter() - start

print(f"First text: {arrival_times[0]:.3f} s")
if len(arrival_times) >= 20:
    print(f"20th text chunk: {arrival_times[19]:.3f} s")

print(f"End-to-end: {end_to_end:.3f} s")
print(f"Output tokens: {final['eval_count']}")
print(f"Load: {final['load_duration'] / 1e9:.3f} s")
print(f"Generation: {final['eval_duration'] / 1e9:.3f} s")