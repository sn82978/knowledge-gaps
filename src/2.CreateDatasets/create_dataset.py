import openai
import csv
import pandas as pd
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
import json
import time
from dotenv import load_dotenv

load_dotenv()


# 1. generate a questions dataset automatically from these papers via OpenAI's API

# papers list

arXiv_Ids = {
    'Scattering Amplitudes in Quantum Field Theory': (2306.05976, 'Physics'),
    'String Field Theory: A Review': (2405.19421, 'Physics'),
    'Field Theoretic Aspects of Condensed Matter Physics: An Overview': (2301.13234, 'Physics'),
    'Higher-Order Topological Phases in Crystalline and Non-Crystalline Systems: A Review': (2309.03688, 'Physics'),
    'A Panorama of Physical Mathematics c. 2022': (2211.04467, 'Physics/Math'),
    'Effective Field Theories for Condensed Matter Systems': (2203.10110, 'Physics'),
    'On the Infinity-Topos Semantics of Homotopy Type Theory': (2212.06937, 'Math'),
    'Langlands Program and Ramanujan Conjecture: A Survey': (1812.05203, 'Math'),
    'The SAGEX Review on Scattering Amplitudes': (2203.13011, 'Physics/Math')
}


# fetch the paper's content via the arXiv API (abstract + metadata)
def fetch_arxiv_content(arxiv_id):
    """
    Fetches full paper metadata and abstract from the arXiv API.
    arXiv API returns Atom XML; we parse title, authors, abstract, and categories.
    """
    # arXiv IDs may be floats in the dict — convert to zero-padded string
    id_str = str(arxiv_id)
    # Remove any trailing zeros that would corrupt the ID
    url = f"http://export.arxiv.org/api/query?id_list={id_str}&max_results=1"

    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch arXiv ID {id_str}: {e}")
        return None

    # Parse the Atom XML response
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    root = ET.fromstring(xml_data)
    entry = root.find('atom:entry', ns)

    if entry is None:
        print(f"No entry found for arXiv ID {id_str}")
        return None

    title = entry.findtext('atom:title', default='', namespaces=ns).strip()
    abstract = entry.findtext('atom:summary', default='', namespaces=ns).strip()
    authors = [a.findtext('atom:name', default='', namespaces=ns) for a in entry.findall('atom:author', ns)]
    categories = [c.attrib.get('term', '') for c in entry.findall('arxiv:primary_category', ns)]
    categories += [c.attrib.get('term', '') for c in entry.findall('atom:category', ns)]

    return {
        'id': id_str,
        'title': title,
        'abstract':abstract,
        'authors': ', '.join(authors),
        'categories': ', '.join(set(categories)),
    }


# send content to openai model
def generate_questions(paper_info, broad_field, client, n_questions=20):
    """
    Sends paper metadata + abstract to OpenAI and requests research/critical-thinking
    questions with a science-topic type label for each.
    Returns a list of dicts: [{'question': ..., 'type': ...}, ...]
    """
    prompt = f"""You are an expert scientific reviewer. Below is the title, abstract, and metadata of an academic paper.

    Title: {paper_info['title']}
    Authors: {paper_info['authors']}
    arXiv Categories: {paper_info['categories']}
    Broad Field: {broad_field}

    Abstract:
    {paper_info['abstract']}

    Your task:
    Generate exactly {n_questions} high-quality research and critical-thinking questions that a scientist or graduate student might ask after reading this paper.

    For each question, also assign a 'type' label that reflects the specific scientific subfield or topic the question belongs to (e.g., "Particle Physics", "Quantum Gravity", "Condensed Matter", "Algebraic Topology", "Number Theory", "String Theory", "Cosmology", etc.). Be as specific as the content warrants.

    Respond ONLY with a valid JSON array. No explanation, no markdown, no extra text. Format:
    [
    {{"question": "...", "type": "..."}},
    ...
    ]
    """

    # send content to openai model and get back the questions & classification
    # retry up to 3 times
    response = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a scientific expert that generates rigorous research questions from academic papers."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.7,
            )
            break  # success — exit retry loop
        except openai.RateLimitError:
            wait = 20 * (attempt + 1)  # 20s, 40s, 60s
            print(f"rate limit: attempt {attempt + 1}/3, waiting {wait}s before retrying")
            time.sleep(wait)
            if attempt == 2:
                print(f"All retries exhausted for '{paper_info['title']}'")
                return []

    if response is None:
        return []

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed for '{paper_info['title']}': {e}")
        print(f"Raw response snippet: {raw[:670]}")
        questions = []

    return questions


# save the questions & classifications in a .csv file
def save_to_csv(all_rows, filepath="questions_dataset.csv"):
    """
    Saves the full dataset to a CSV with columns: paper_title, arxiv_id, broad_field, question, type
    """
    df = pd.DataFrame(all_rows, columns=["paper_title", "arxiv_id", "broad_field", "question", "type"])
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df)} questions to '{filepath}'")
    return df


def main():
    # Initialise the OpenAI client (reads OPENAI_API_KEY from environment)
    client = openai.OpenAI()

    all_rows = []

    for paper_title, (arxiv_id, broad_field) in arXiv_Ids.items():
        print(f"\nProcessing: {paper_title}  [{arxiv_id}]")

        # fetch the paper's content
        paper_info = fetch_arxiv_content(arxiv_id)
        if paper_info is None:
            print("Skipping due to fetch error.")
            continue

        print(f"Fetched: '{paper_info['title']}'")
        print(f"Categories: {paper_info['categories']}")

        # send content to openai model and get back questions & classification
        questions = generate_questions(paper_info, broad_field, client, n_questions=100)
        print(f"Generated {len(questions)} questions.")

        for q in questions:
            all_rows.append({
                "paper_title": paper_title,
                "arxiv_id": str(arxiv_id),
                "broad_field": broad_field,
                "question": q.get("question", ""),
                "type": q.get("type", ""),
            })

        # be nice to the arXiv API and OpenAI to avoid rate limiting
        time.sleep(15)

    # save the questions & classifications in a .csv file
    df = save_to_csv(all_rows, filepath="questions_dataset.csv")

    # 2. verify the questions myself
    print(df.head(10).to_string(index=False))
    print(f"Total questions generated: {len(df)}")
    print(f"Question types found:\n{df['type'].value_counts().to_string()}")

if __name__ == "__main__":
    main()