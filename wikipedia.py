import wikipediaapi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import time
import random

# Wikipedia-API Setup
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyClusteringApp/1.0 (https://github.com/Zprit/DataScienceClustering; ole-kropp@t-online.de)'
)

# Themenbereiche als Startseiten
start_pages = {
    "AI": "Artificial intelligence",
    "Politics": "Politics",
    "History": "World War II",
    "Biology": "Biology",
    "Animals": "Animal",
    "Economics": "Economics",
    "Physics": "Physics",
    "Popculture": "Film",
    "Sport": "Football",
}

def get_linked_articles_from_page(title, max_links=120):
    page = wiki.page(title)
    links = list(page.links.keys())
    return random.sample(links, min(len(links), max_links))

# Artikel aus allen Startseiten kombinieren
print("📥 Sammle Artikel aus verschiedenen Themen ...")
article_titles = []
for topic, page in start_pages.items():
    print(f"🔹 {topic}: {page}")
    article_titles += get_linked_articles_from_page(page, max_links=120)

# Deduplizieren
article_titles = list(set(article_titles))
print(f"📄 Insgesamt {len(article_titles)} eindeutige Artikel gesammelt.")

# Artikelinhalte abrufen
documents = []
valid_titles = []
for i, title in enumerate(article_titles, 1):
    print(f"[{i}/{len(article_titles)}] Lade Artikel: {title} ...")
    page = wiki.page(title)
    text = page.text if page.exists() else ""
    if text.strip():
        documents.append(text)
        valid_titles.append(title)
    else:
        print(f"⚠️ Artikel '{title}' nicht gefunden oder leer.")
    time.sleep(0.2)
    if len(valid_titles) >= 1000:
        print("✅ Limit von 1000 Artikeln erreicht.")
        break

print(f"✅ Insgesamt {len(valid_titles)} Artikel erfolgreich geladen.")

# TF-IDF Vektorisierung
print("🔎 Vektorisiere Texte ...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(documents)

# Beste Clusteranzahl bestimmen
print("📐 Berechne beste Clusteranzahl ...")
sse = []
silhouette_scores = []
cluster_range = range(5, 21)

for k in cluster_range:
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(X)
    sse.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, model.labels_))

# Elbow + Silhouette Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, sse, marker='o')
plt.title("Elbow-Methode")
plt.xlabel("Cluster")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='x', color='green')
plt.title("Silhouette Score")
plt.xlabel("Cluster")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# Clustering mit bester Clusteranzahl
best_k = cluster_range[np.argmax(silhouette_scores)]
print(f"✅ Beste Clusteranzahl: {best_k}")
model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
labels = model.fit_predict(X)

# PCA Visualisierung
print("📊 Visualisiere Cluster ...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

colors = plt.cm.get_cmap('tab20', best_k)
plt.figure(figsize=(12, 8))
for i in range(best_k):
    idx = np.where(labels == i)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {i+1}", alpha=0.7, color=colors(i))

for i, title in enumerate(valid_titles[:300]):  # beschränke Labels für Lesbarkeit
    plt.annotate(title, (X_pca[i, 0], X_pca[i, 1]), fontsize=6, alpha=0.6)

plt.title("Wikipedia-Artikel Clustering (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()