from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import umap

client = genai.Client()

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?",
    "Solve for x: 2x + 5 = 13",
    "What is a linear equation?",
    "Explain the distributive property",
    "What is a quadratic equation?",
    "What is a prime number?",
    "Explain the fundamental theorem of arithmetic",
    "What is modular arithmetic?",
    "Define the greatest common divisor"]

# Define groups for each text
groups = [
    "Philosophy",      # What is the meaning of life?
    "Philosophy",      # What is the purpose of existence?
    "Cooking",         # How do I bake a cake?
    "Algebra",         # Solve for x: 2x + 5 = 13
    "Algebra",         # What is a linear equation?
    "Algebra",         # Explain the distributive property
    "Algebra",         # What is a quadratic equation?
    "Number Theory",   # What is a prime number?
    "Number Theory",   # Explain the fundamental theorem of arithmetic
    "Number Theory",   # What is modular arithmetic?
    "Number Theory"    # Define the greatest common divisor
]

result = [
    np.array(e.values) for e in client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="CLUSTERING")).embeddings
]

embeddings_matrix = np.array(result)
similarity_matrix = cosine_similarity(embeddings_matrix)

for i, text1 in enumerate(texts):
    for j in range(i + 1, len(texts)):
        text2 = texts[j]
        similarity = similarity_matrix[i, j]
        print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=2, min_dist=0.1, init='random')
embeddings_2d = reducer.fit_transform(embeddings_matrix)

# Create color mapping
unique_groups = list(set(groups))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
group_to_color = {group: colors[i] for i, group in enumerate(unique_groups)}

plt.figure(figsize=(10, 8))

# Plot each group with its own color
for i, (txt, group) in enumerate(zip(texts, groups)):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                color=group_to_color[group], s=100, alpha=0.6, label=group)
    plt.annotate(txt[:1], (embeddings_2d[i, 0], embeddings_2d[i, 1]))

# Remove duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of Text Embeddings')
plt.savefig("./umap_gemini_noodle.png")