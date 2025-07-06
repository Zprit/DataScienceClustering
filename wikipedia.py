import wikipediaapi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time

# 1. Artikel-Liste definieren
article_titles = [
    "Machine learning", "Artificial intelligence", "Deep learning",
    "Neural network", "Support vector machine", "Data mining",
    "Economics", "Macroeconomics", "Microeconomics",
    "Finance", "Investment", "Bank",
    "Dog", "Cat", "Pet", "Animal"
]

# 2. Wikipedia-API vorbereiten
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyClusteringApp/1.0 (https://github.com/Zprit/DataScienceClustering; ole-kropp@t-online.de)'
)

# 3. Inhalte abrufen mit Fortschrittsanzeige
documents = []
for i, title in enumerate(article_titles, 1):
    print(f"[{i}/{len(article_titles)}] Lade Artikel: {title} ...")
    page = wiki.page(title)
    text = page.text if page.exists() else ""
    documents.append(text)
    if not text:
        print(f"⚠️ Artikel '{title}' nicht gefunden.")
    time.sleep(0.2)  # höflich bleiben gegenüber der API

print("✅ Alle Artikel geladen.")

# 4. TF-IDF Vektorisierung
print("🔎 Vektorisiere Texte mit TF-IDF ...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)

# 5. Clustering (z.B. KMeans)
k = 4
print(f"🔗 Führe KMeans-Clustering mit {k} Clustern durch ...")
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)

# 6. Visualisierung mit PCA
print("📊 Reduziere Dimensionen mit PCA für Visualisierung ...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

colors = ['red', 'green', 'blue', 'orange']
for i in range(k):
    idx = np.where(labels == i)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {i+1}", alpha=0.7)

# 7. Titel anzeigen
for i, title in enumerate(article_titles):
    plt.annotate(title, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.8)

plt.title("Wikipedia-Artikel Cluster")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
