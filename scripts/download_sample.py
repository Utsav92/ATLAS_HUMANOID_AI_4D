import urllib.request
import os

url = "https://upload.wikimedia.org/wikipedia/commons/e/ed/Salsa_Dancing_Fundamentals.webm"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
req = urllib.request.Request(url, headers=headers)

out_file = r"videos\Salsa_Dancing_Fundamentals.webm"
os.makedirs("videos", exist_ok=True)

print("Downloading", url)
with urllib.request.urlopen(req) as resp, open(out_file, 'wb') as f:
    f.write(resp.read())

print("Downloaded successfully to", out_file, "Size:", os.path.getsize(out_file))
