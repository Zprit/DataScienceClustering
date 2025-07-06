import wikipediaapi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import time

# 1. Wikipedia-API vorbereiten
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyClusteringApp/1.0 (https://github.com/Zprit/DataScienceClustering; ole-kropp@t-online.de)'
)

# 2. Artikel automatisch laden aus Links einer Wikipedia-Seite
def get_linked_articles_from_page(start_title, max_links=30):
    print(f"🔗 Lade bis zu {max_links} Links von '{start_title}' ...")
    page = wiki.page(start_title)
    links = list(page.links.keys())
    return links[:max_links]

# 3. Artikelquellen kombinieren
manual_articles = [
    "Machine learning", "Economics", "Dog"
]
linked_articles = get_linked_articles_from_page("Artificial intelligence", max_links=30)
article_titles = list(set(manual_articles + linked_articles))

print(f"📄 Insgesamt {len(article_titles)} Artikel ausgewählt.")

# 4. Inhalte abrufen mit Fortschrittsanzeige
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

print(f"✅ {len(valid_titles)} Artikel erfolgreich geladen.")

# 5. TF-IDF Vektorisierung
print("🔎 Vektorisiere Texte mit TF-IDF ...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)

# 6. Elbow-Methode und Silhouette Score
print("📐 Bestimme optimale Clusteranzahl ...")
sse = []
silhouette_scores = []
cluster_range = range(2, 10)

for k in cluster_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    sse.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, model.labels_))

# 7. Visualisierung der Elbow-Kurve
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, sse, marker='o')
plt.title("Elbow-Methode (SSE)")
plt.xlabel("Anzahl Cluster")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='x', color='green')
plt.title("Silhouette Score")
plt.xlabel("Anzahl Cluster")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# 8. Beste Clusteranzahl wählen
best_k = cluster_range[np.argmax(silhouette_scores)]
print(f"✅ Beste Clusteranzahl laut Silhouette Score: {best_k}")

# 9. Clustering mit bester Clusteranzahl
model = KMeans(n_clusters=best_k, random_state=42)
labels = model.fit_predict(X)

# 10. Visualisierung mit PCA
print("📊 Reduziere Dimensionen mit PCA für Visualisierung ...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Farben generieren
colors = plt.cm.get_cmap('tab10', best_k)

plt.figure(figsize=(8, 6))
for i in range(best_k):
    idx = np.where(labels == i)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {i+1}", alpha=0.7, color=colors(i))

# Titel anzeigen
for i, title in enumerate(valid_titles):
    plt.annotate(title, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.8)

plt.title("Wikipedia-Artikel Cluster")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
