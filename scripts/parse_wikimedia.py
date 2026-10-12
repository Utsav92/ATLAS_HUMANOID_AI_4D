import re

content_path = r"C:\Users\Utsav\.gemini\antigravity\brain\bf9fcd1b-d708-48b1-8610-01f91dc277ee\.system_generated\steps\1222\content.md"
with open(content_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

urls = set(re.findall(r'https://upload\.wikimedia\.org/wikipedia/commons/[^\"\'\s>]+\.webm', text))
for u in sorted(urls):
    print("Found URL:", u)
