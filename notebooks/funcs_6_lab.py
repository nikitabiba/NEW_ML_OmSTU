<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
import umap
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os
import warnings

warnings.filterwarnings("ignore")


def extract_feature_importances(estimator, X):
    try:
        fs = estimator.named_steps['feature_selection']
        if hasattr(fs, 'get_support'):
            mask = fs.get_support()
            feature_importances = np.zeros(X.shape[1])
            feature_importances[mask] = 1
        elif hasattr(fs, 'feature_importances_'):
            feature_importances = fs.feature_importances_
        elif hasattr(fs, 'components_'):
            feature_importances = fs.components_
        else:
            feature_importances = None
    except Exception as e:
        print(f"Не удалось извлечь важности признаков: {e}")
        feature_importances = None
    return feature_importances


def optimize_for_classification(feature_selector, param_grid, name, class_X, class_y):
    model_path = f'C:\\Users\\My Computer\\Desktop\\Work\\Learn\\ML(OMSTU)\\ML-OmSTU-\\notebooks\\models_6_lab\\{name}_best_model_classification.joblib'
    
    if os.path.exists(model_path):
        print(f'Загрузка модели классификации из {model_path}')
        best_model = load(model_path)
        feature_importances = extract_feature_importances(best_model, class_X)
        best_params = best_model.get_params()
        score = accuracy_score(class_y, best_model.predict(class_X))
        print(f"Лучшие параметры для {name}: {best_params}")
        print(f"Лучший score: {score:.4f}")
    
        return best_model, best_params, feature_importances, score

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('classifier', GaussianNB())
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    n_jobs = -1

    if name == KernelPCA:
        n_jobs = 1
    else:
        pass

    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(class_X, class_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    feature_importances = extract_feature_importances(best_model, class_X)
    
    print(f"Лучшие параметры для {name}: {best_params}")
    print(f"Лучший score: {grid_search.best_score_:.4f}")
    
    dump(best_model, model_path)
    
    return best_model, best_params, feature_importances, grid_search.best_score_


def optimize_for_regression(feature_selector, param_grid, name, reg_X, reg_y):
    model_path = f'C:\\Users\\My Computer\\Desktop\\Work\\Learn\\ML(OMSTU)\\ML-OmSTU-\\notebooks\\models_6_lab\\{name}_best_model_regression.joblib'
    
    if os.path.exists(model_path):
        print(f'Загрузка модели регрессии из {model_path}')
        best_model = load(model_path)
        feature_importances = extract_feature_importances(best_model, reg_X)
        best_params = best_model.get_params()
        score = mean_squared_error(reg_y, best_model.predict(reg_X))
        print(f"Лучшие параметры для {name}: {best_params}")
        print(f"Лучший MSE: {score:.4f}")

        return best_model, best_params, feature_importances, score

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('regressor', BaggingRegressor(max_features=1.0, max_samples=0.75, n_estimators=100))
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(reg_X, reg_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    feature_importances = extract_feature_importances(best_model, reg_X)
    
    print(f"Лучшие параметры для {name}: {best_params}")
    print(f"Лучший MSE: {grid_search.best_score_:.4f}")
    
    dump(best_model, model_path)
    
    return best_model, best_params, feature_importances, grid_search.best_score_

def visualize_results(feature_importances, method_name, is_classification=True):
    if feature_importances is None:
        return
    
    plt.figure(figsize=(12, 6))
    
    if len(feature_importances.shape) == 1:
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.xlabel('Признаки')
        plt.ylabel('Важность/Маска')
        plt.title(f'Важность признаков для {method_name} {"(Классификация)" if is_classification else "(Регрессия)"}')
    else:
        plt.imshow(feature_importances, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.xlabel('Признаки')
        plt.ylabel('Компоненты')
        plt.title(f'Компоненты для {method_name} {"(Классификация)" if is_classification else "(Регрессия)"}')
    
    plt.tight_layout()
    plt.show()

def visualize_kernel_pca_transformed_space(kernel_pca, model, X):
    X_kpca = kernel_pca.transform(X)

    if X_kpca.shape[1] > 2:
        X_kpca = X_kpca[:, :2]

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], 
                          c=y_pred, cmap='viridis', alpha=0.7)
    
    plt.colorbar(scatter, label='Предсказание модели')
    plt.xlabel('1-я компонентa')
    plt.ylabel('2-я компонентa')
    plt.title('KernelPCA — визуализация преобразованного пространства')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_tsne_classification(X_tsne, y_true, y_pred=None):
    plt.figure(figsize=(10, 6))
    
    if y_pred is None:
        c = y_true
        title = 't-SNE пространство (классы — истинные)'
    else:
        c = y_pred
        title = 't-SNE пространство (классы — предсказанные)'

    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Класс')
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_tsne_regression(X_tsne, y_true, y_pred=None):
    plt.figure(figsize=(10, 6))
    
    if y_pred is None:
        c = y_true
        title = 't-SNE пространство (значения — истинные)'
    else:
        c = y_pred
        title = 't-SNE пространство (значения — предсказанные)'

    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Целевая переменная')
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def visualize_2d_projection(X_2d, y, title, is_classification=True, y_pred=None):
    plt.figure(figsize=(10, 6))

    if is_classification:
        labels = y_pred if y_pred is not None else y
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Класс')
    else:
        values = y_pred if y_pred is not None else y
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=values, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Целевая переменная')

    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_scree_pca(pca_model, title="Scree Plot (PCA)"):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(pca_model.explained_variance_ratio_) + 1),
             pca_model.explained_variance_ratio_, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Номер компоненты')
    plt.ylabel('Доля объяснённой дисперсии')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def metrics(model, data, data_y):
    labels = model.fit_predict(data)

    silhouette = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    ar = adjusted_rand_score(data_y, labels)
    nmi = normalized_mutual_info_score(labels, labels)

    silhouette = "{:.2f}".format(silhouette)
    db = "{:.2f}".format(db)
    ar = "{:.2f}".format(ar)
    nmi = "{:.2f}".format(nmi)

    print(f"Silhouette: {silhouette}")
    print(f"Davies-Bouldin: {db}")
    print(f"Adjusted Rand Index: {ar}")
    print(f"Normalized Mutual Information: {nmi}")

    return [silhouette, db, ar, nmi]

=======
<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
import umap
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os
import warnings

warnings.filterwarnings("ignore")


def extract_feature_importances(estimator, X):
    try:
        fs = estimator.named_steps['feature_selection']
        if hasattr(fs, 'get_support'):
            mask = fs.get_support()
            feature_importances = np.zeros(X.shape[1])
            feature_importances[mask] = 1
        elif hasattr(fs, 'feature_importances_'):
            feature_importances = fs.feature_importances_
        elif hasattr(fs, 'components_'):
            feature_importances = fs.components_
        else:
            feature_importances = None
    except Exception as e:
        print(f"Не удалось извлечь важности признаков: {e}")
        feature_importances = None
    return feature_importances


def optimize_for_classification(feature_selector, param_grid, name, class_X, class_y):
    model_path = f'C:\\Users\\My Computer\\Desktop\\Work\\Learn\\ML(OMSTU)\\ML-OmSTU-\\notebooks\\models_6_lab\\{name}_best_model_classification.joblib'
    
    if os.path.exists(model_path):
        print(f'Загрузка модели классификации из {model_path}')
        best_model = load(model_path)
        feature_importances = extract_feature_importances(best_model, class_X)
        best_params = best_model.get_params()
        score = accuracy_score(class_y, best_model.predict(class_X))
        return best_model, best_params, feature_importances, score

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('classifier', GaussianNB())
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    n_jobs = -1

    if name == KernelPCA:
        n_jobs = 1
    else:
        pass

    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(class_X, class_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    feature_importances = extract_feature_importances(best_model, class_X)
    
    print(f"Лучшие параметры для {name}: {best_params}")
    print(f"Лучший score: {grid_search.best_score_:.4f}")
    
    dump(best_model, model_path)
    
    return best_model, best_params, feature_importances, grid_search.best_score_


def optimize_for_regression(feature_selector, param_grid, name, reg_X, reg_y):
    model_path = f'C:\\Users\\My Computer\\Desktop\\Work\\Learn\\ML(OMSTU)\\ML-OmSTU-\\notebooks\\models_6_lab\\{name}_best_model_regression.joblib'
    
    if os.path.exists(model_path):
        print(f'Загрузка модели регрессии из {model_path}')
        best_model = load(model_path)
        feature_importances = extract_feature_importances(best_model, reg_X)
        best_params = best_model.get_params()
        score = mean_squared_error(reg_y, best_model.predict(reg_X))
        return best_model, best_params, feature_importances, score

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('regressor', BaggingRegressor(max_features=1.0, max_samples=0.75, n_estimators=100))
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(reg_X, reg_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    feature_importances = extract_feature_importances(best_model, reg_X)
    
    print(f"Лучшие параметры для {name}: {best_params}")
    print(f"Лучший MSE: {-grid_search.best_score_:.4f}")
    
    dump(best_model, model_path)
    
    return best_model, best_params, feature_importances, grid_search.best_score_

def visualize_results(feature_importances, method_name, is_classification=True):
    if feature_importances is None:
        print(f"Нет доступных данных для визуализации метода {method_name}")
        return
    
    plt.figure(figsize=(12, 6))
    
    if len(feature_importances.shape) == 1:
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.xlabel('Признаки')
        plt.ylabel('Важность/Маска')
        plt.title(f'Важность признаков для {method_name} {"(Классификация)" if is_classification else "(Регрессия)"}')
    else:
        plt.imshow(feature_importances, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.xlabel('Признаки')
        plt.ylabel('Компоненты')
        plt.title(f'Компоненты для {method_name} {"(Классификация)" if is_classification else "(Регрессия)"}')
    
    plt.tight_layout()
    plt.show()

=======
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
import umap
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os
import warnings

warnings.filterwarnings("ignore")


def extract_feature_importances(estimator, X):
    try:
        fs = estimator.named_steps['feature_selection']
        if hasattr(fs, 'get_support'):
            mask = fs.get_support()
            feature_importances = np.zeros(X.shape[1])
            feature_importances[mask] = 1
        elif hasattr(fs, 'feature_importances_'):
            feature_importances = fs.feature_importances_
        elif hasattr(fs, 'components_'):
            feature_importances = fs.components_
        else:
            feature_importances = None
    except Exception as e:
        print(f"Не удалось извлечь важности признаков: {e}")
        feature_importances = None
    return feature_importances


def optimize_for_classification(feature_selector, param_grid, name, class_X, class_y):
    model_path = f'C:\\Users\\My Computer\\Desktop\\Work\\Learn\\ML(OMSTU)\\ML-OmSTU-\\notebooks\\models_6_lab\\{name}_best_model_classification.joblib'
    
    if os.path.exists(model_path):
        print(f'Загрузка модели классификации из {model_path}')
        best_model = load(model_path)
        feature_importances = extract_feature_importances(best_model, class_X)
        best_params = best_model.get_params()
        score = accuracy_score(class_y, best_model.predict(class_X))
        return best_model, best_params, feature_importances, score

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('classifier', GaussianNB())
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    n_jobs = -1

    if name == KernelPCA:
        n_jobs = 1
    else:
        pass

    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(class_X, class_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    feature_importances = extract_feature_importances(best_model, class_X)
    
    print(f"Лучшие параметры для {name}: {best_params}")
    print(f"Лучший score: {grid_search.best_score_:.4f}")
    
    dump(best_model, model_path)
    
    return best_model, best_params, feature_importances, grid_search.best_score_


def optimize_for_regression(feature_selector, param_grid, name, reg_X, reg_y):
    model_path = f'C:\\Users\\My Computer\\Desktop\\Work\\Learn\\ML(OMSTU)\\ML-OmSTU-\\notebooks\\models_6_lab\\{name}_best_model_regression.joblib'
    
    if os.path.exists(model_path):
        print(f'Загрузка модели регрессии из {model_path}')
        best_model = load(model_path)
        feature_importances = extract_feature_importances(best_model, reg_X)
        best_params = best_model.get_params()
        score = mean_squared_error(reg_y, best_model.predict(reg_X))
        return best_model, best_params, feature_importances, score

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('regressor', BaggingRegressor(max_features=1.0, max_samples=0.75, n_estimators=100))
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(reg_X, reg_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    feature_importances = extract_feature_importances(best_model, reg_X)
    
    print(f"Лучшие параметры для {name}: {best_params}")
    print(f"Лучший MSE: {-grid_search.best_score_:.4f}")
    
    dump(best_model, model_path)
    
    return best_model, best_params, feature_importances, grid_search.best_score_

def visualize_results(feature_importances, method_name, is_classification=True):
    if feature_importances is None:
        print(f"Нет доступных данных для визуализации метода {method_name}")
        return
    
    plt.figure(figsize=(12, 6))
    
    if len(feature_importances.shape) == 1:
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.xlabel('Признаки')
        plt.ylabel('Важность/Маска')
        plt.title(f'Важность признаков для {method_name} {"(Классификация)" if is_classification else "(Регрессия)"}')
    else:
        plt.imshow(feature_importances, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.xlabel('Признаки')
        plt.ylabel('Компоненты')
        plt.title(f'Компоненты для {method_name} {"(Классификация)" if is_classification else "(Регрессия)"}')
    
    plt.tight_layout()
    plt.show()

>>>>>>> f4857c11c792fcf9fa1f50ea343e5c95c0680e6f
>>>>>>> 0ac15ee36cfc8f296ec947bb7ab89d5fc3686dfe
