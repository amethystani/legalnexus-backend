"""Quick test of DeepSeek citation extraction"""
from langchain_community.llms import Ollama

# Test DeepSeek
llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1)

sample_text = """
The petitioner relied on AIR 2019 SC 123 and (2018) 10 SCC 456.
The case of State v. Kumar (2020) was also cited. 
Reference was made to Punjab HC 2015 judgment.
"""

prompt = f"""Extract ALL citations from this text. List them one per line.

TEXT: {sample_text}

CITATIONS:"""

print("Testing DeepSeek R1:1.5b for citation extraction...")
response = llm.invoke(prompt)
print(f"\nDeepSeek Response:\n{response}\n")

# Parse
lines = str(response).split('\n')
citations = []
for line in lines:
    line = line.strip()
    if line and len(line) > 3:
        citations.append(line)

print(f"Found {len(citations)} citations:")
for c in citations:
    print(f"  - {c}")
