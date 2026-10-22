import os
import csv
import json
import time
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
from ollama import Client
from pydantic import BaseModel

# Config
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OUT_DIR = os.getenv("OUT_DIR", ".")
CACHE_FILE = os.path.join(OUT_DIR, "papers_cache.json")
OUT_CSV = os.path.join(OUT_DIR, "questions_dataset.csv")

QUESTIONS_PER_PAPER = 100
BATCH_SIZE = 20 # small models are much more reliable at ~20 per call
MAX_ROUNDS = 12 # safety cap on calls per paper

arXiv_Ids = {
    'Scattering Amplitudes in Quantum Field Theory': ("2306.05976", 'Physics'),
    'String Field Theory: A Review': ("2405.19421", 'Physics'),
    'Field Theoretic Aspects of Condensed Matter Physics: An Overview': ("2301.13234", 'Physics'),
    'Higher-Order Topological Phases in Crystalline and Non-Crystalline Systems: A Review': ("2309.03688", 'Physics'),
    'A Panorama of Physical Mathematics c. 2022': ("2211.04467", 'Physics/Math'),
    'Effective Field Theories for Condensed Matter Systems': ("2203.10110", 'Physics'),
    'On the Infinity-Topos Semantics of Homotopy Type Theory': ("2212.06937", 'Math'),
    'Langlands Program and Ramanujan Conjecture: A Survey': ("1812.05203", 'Math'),
    'The SAGEX Review on Scattering Amplitudes': ("2203.13011", 'Physics/Math'),
}


# Structured output schema: Ollama constrains the model to this JSON shape
class QA(BaseModel):
    question: str
    type: str


class QAList(BaseModel):
    questions: list[QA]


# 1. get paper metadata (cached it, so internet is only needed once)
def fetch_arxiv_content(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch arXiv ID {arxiv_id}: {e}")
        return None

    ns = {'atom': 'http://www.w3.org/2005/Atom',
          'arxiv': 'http://arxiv.org/schemas/atom'}
    entry = ET.fromstring(xml_data).find('atom:entry', ns)
    if entry is None:
        print(f"  No entry found for arXiv ID {arxiv_id}")
        return None

    authors = [a.findtext('atom:name', default='', namespaces=ns) for a in entry.findall('atom:author', ns)]
    cats = [c.attrib.get('term', '') for c in entry.findall('arxiv:primary_category', ns)]
    cats += [c.attrib.get('term', '') for c in entry.findall('atom:category', ns)]

    return {
        'id': arxiv_id,
        'title': " ".join(entry.findtext('atom:title', default='', namespaces=ns).split()),
        'abstract': " ".join(entry.findtext('atom:summary', default='', namespaces=ns).split()),
        'authors': ', '.join(authors),
        'categories': ', '.join(sorted(set(cats))),
    }


def load_cache(): # run first to cache papers
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_paper_info(arxiv_id, cache):
    if arxiv_id in cache:
        return cache[arxiv_id]
    info = fetch_arxiv_content(arxiv_id)
    if info:
        cache[arxiv_id] = info
        save_cache(cache)
        time.sleep(5) # be polite to arXiv
    return info

# 2. gen questions with a local Ollama model (llama3b)
def generate_batch(paper_info, broad_field, client, n, already_have):
    avoid = ""
    if already_have:
        sample = already_have[-40:] # keep the prompt short
        avoid = ("\nDo NOT repeat or paraphrase any of these existing questions:\n- "
                 + "\n- ".join(sample) + "\n")

    prompt = f"""Below is the title, abstract, and metadata of an academic paper.

Title: {paper_info['title']}
Authors: {paper_info['authors']}
arXiv Categories: {paper_info['categories']}
Broad Field: {broad_field}

Abstract:
{paper_info['abstract']}

Your task:
Generate exactly {n} high-quality research and critical-thinking questions that a scientist or graduate student might ask after reading this paper. Make them diverse: cover methods, assumptions, limitations, connections to other fields, open problems, and possible extensions.

For each question, also assign a 'type' label that reflects the specific scientific subfield or topic the question belongs to (e.g., "Particle Physics", "Quantum Gravity", "Condensed Matter", "Algebraic Topology", "Number Theory", "String Theory", "Cosmology"). Be as specific as the content warrants.
{avoid}
Return a JSON object with a single key "questions" holding a list of objects with keys "question" and "type"."""

    for attempt in range(3):
        try:
            resp = client.chat(
                model=MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a scientific expert that generates rigorous research questions from academic papers."},
                    {"role": "user", "content": prompt},
                ],
                format=QAList.model_json_schema(),
                options={"temperature": 0.8, "num_ctx": 8192},
            )
            parsed = QAList.model_validate_json(resp["message"]["content"])
            return [q.model_dump() for q in parsed.questions]
        except Exception as e:
            print(f"  attempt {attempt + 1}/3 failed: {e}")
            time.sleep(5)
    return []


def generate_questions(paper_info, broad_field, client, n_questions=QUESTIONS_PER_PAPER):
    results, seen = [], set()
    for round_ in range(MAX_ROUNDS):
        if len(results) >= n_questions:
            break
        need = min(BATCH_SIZE, n_questions - len(results))
        batch = generate_batch(paper_info, broad_field, client, need,
                               [q["question"] for q in results])
        for q in batch:
            key = q["question"].strip().lower()
            if key and key not in seen:
                seen.add(key)
                results.append(q)
        print(f"  round {round_ + 1}: {len(results)}/{n_questions} unique questions")
    return results[:n_questions]

# 3. save
def save_to_csv(all_rows, filepath=OUT_CSV):
    df = pd.DataFrame(all_rows, columns=["paper_title", "arxiv_id", "broad_field", "question", "type"])
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df)} questions to '{filepath}'")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = Client(host=OLLAMA_HOST)
    cache = load_cache()
    all_rows = []

    for paper_title, (arxiv_id, broad_field) in arXiv_Ids.items():
        print(f"\nProcessing: {paper_title}  [{arxiv_id}]")

        paper_info = get_paper_info(arxiv_id, cache)
        if paper_info is None:
            print("Skipping due to fetch error.")
            continue
        print(f"Fetched: '{paper_info['title']}'")

        questions = generate_questions(paper_info, broad_field, client)
        print(f"Generated {len(questions)} questions.")

        for q in questions:
            all_rows.append({
                "paper_title": paper_title,
                "arxiv_id": arxiv_id,
                "broad_field": broad_field,
                "question": q["question"],
                "type": q["type"],
            })

        # save after every paper so a killed job doesn't lose everything
        save_to_csv(all_rows)

    df = save_to_csv(all_rows)

    # 2. verify the questions myself
    print(df.head(10).to_string(index=False))
    print(f"Total questions generated: {len(df)}")
    print(f"Question types found:\n{df['type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
