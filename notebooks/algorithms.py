import numpy as np
from collections import Counter


class myKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)  # Евклидово расстояние
        k_indices = np.argsort(distances)[:self.k]  # Индексы k ближайших соседей
        k_nearest_labels = self.y_train[k_indices]  # Метки этих соседей
        most_common = Counter(k_nearest_labels).most_common(1)  # Самый частый класс
        return most_common[0][0]

class MyKMeans:
    def __init__(self, n_clusters=2, max_iter=1000, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.tol:
                break
            self.centroids = new_centroids

        self.inertia_ = np.sum([
            np.sum((X[self.labels == j] - self.centroids[j]) ** 2)
            for j in range(self.n_clusters)
        ])

        self.cluster_centers_ = self.centroids
        self.labels_ = self.labels

        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

class MyPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Центрирование данных
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Вычисление ковариационной матрицы
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Вычисление собственных значений и собственных векторов
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Сортировка собственных векторов по убыванию собственных значений
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        
        # Сохранение первых n_components собственных векторов
        self.components = eigenvectors[:, :self.n_components]
        
        # Вычисление объясненной доли дисперсии
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        # Центрирование данных
        X_centered = X - self.mean
        
        # Проекция данных на главные компоненты
        X_transformed = np.dot(X_centered, self.components)
        
        return X_transformed