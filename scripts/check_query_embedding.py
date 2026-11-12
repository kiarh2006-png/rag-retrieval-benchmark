import json
import urllib.request
import numpy as np

url = "http://localhost:11434/api/embed"

payload = {
    "model": "bge-m3",
    "input": "طبق قانون مالیات‌های مستقیم، اشخاص حقیقی ایرانی مقیم ایران نسبت به چه درآمدهایی مشمول مالیات هستند؟"
}

data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req, timeout=120) as response:
    result = json.loads(response.read().decode("utf-8"))

vector = np.asarray(result["embeddings"][0], dtype=np.float32)

print("MODEL:", result["model"])
print("QUERY DIMENSION:", vector.shape)
print("VECTOR NORM:", np.linalg.norm(vector))
print("FINITE VALUES:", np.isfinite(vector).all())
print("QUERY:", payload["input"])

assert vector.shape == (1024,)
assert np.isfinite(vector).all()

print("\nQUERY EMBEDDING TEST PASSED")
