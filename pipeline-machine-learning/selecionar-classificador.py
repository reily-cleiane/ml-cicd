import pandas as pd
import string
import nltk
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, make_scorer
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
import wandb
import joblib

# ----------- Pré-processador seguro para GridSearchCV -----------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        stop_words = set(stopwords.words('portuguese'))  # carregado aqui!
        return X.apply(lambda text: self._preprocess(text, stop_words))

    def _preprocess(self, text, stop_words):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return ' '.join([word for word in text.split() if word not in stop_words])


# ----------- Embedding Transformer (com suporte a trust_remote_code) -----------
class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', trust_remote_code=False):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    def fit(self, X, y=None): return self

    def transform(self, X):
        return self.model.encode(X.tolist(), show_progress_bar=False)

def carregar_dataset_treinamento():
    # Mudar para usar valor vindo de variável de ambiente.
    artefato = wandb.use_artifact('cleiane-projetos/intencao-dialogar/dataset_treinamento:latest', type='dataset')
    # Mudar para usar valor vindo de variável de ambiente.
    csv_path = artefato.get_path("dataset-treinamento.csv").download()
    df = pd.read_csv(csv_path, keep_default_na=False)
    return df

def gerar_params_grid_tfidf():
    # ----------- Classificadores -----------
    logreg = LogisticRegression(max_iter=1000,class_weight='balanced')
    nb = MultinomialNB()
    rf = RandomForestClassifier()
    svc = CalibratedClassifierCV(LinearSVC())
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    pipeline_tfidf = Pipeline([
        ('prep', TextPreprocessor()),
        ('vect', TfidfVectorizer()),
        ('clf', logreg)
    ])
    
    param_grids_tfidf = [
        {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'vect__max_df': [1.0,0.8],
            'vect__min_df': [1],
            'clf': [logreg],
            'clf__C': [0.1, 1.0, 5.0, 10.0],
        },
        {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'vect__max_df': [1.0,0.8],
            'vect__min_df': [1],
            'clf': [nb],
            'clf__alpha': [0.5, 1.0]
        },
        {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'vect__max_df': [1.0,0.8],
            'vect__min_df': [1],
            'clf': [rf],
            'clf__n_estimators': [100, 200]
        },
        {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'vect__max_df': [1.0,0.8],
            'vect__min_df': [1],
            'clf': [svc],
        },
        {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'vect__max_df': [1.0,0.8],
            'vect__min_df': [1],
            'clf': [xgb],
            'clf__n_estimators': [100],
            'clf__max_depth': [3, 5]
        }
    ]
    return param_grids_tfidf, pipeline_tfidf

def gerar_params_grid_embeddings():
    logreg = LogisticRegression(max_iter=1000,class_weight='balanced')
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    pipeline_embeddings = Pipeline([
        ('prep', TextPreprocessor()),
        ('vect', EmbeddingVectorizer()),
        ('clf', logreg)
    ])
    
    param_grids_embeddings = [
        {
            'vect__model_name': ['paraphrase-MiniLM-L6-v2'],
            'vect__trust_remote_code': [False],
            'clf': [logreg],
            'clf__C': [1.0, 5.0, 10.0],
        },
        {
            'vect__model_name': ['Alibaba-NLP/gte-multilingual-base'],
            'vect__trust_remote_code': [True],
            'clf': [xgb],
            'clf__n_estimators': [100],
            'clf__max_depth': [3],
        }
    ]
    return param_grids_embeddings, pipeline_embeddings

def rodar_grid_search(pipeline, param_grids, X, y):
    classificadores = []

    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'recall_macro': 'recall_macro',
        'precision_macro': 'precision_macro'
    }
    for param_grid in param_grids:

        nome_classificador = param_grid['clf'][0].__class__.__name__
        print(f"\n🔍 Rodando GridSearch com classificador: {nome_classificador}")

        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring=scoring, refit='f1_macro', n_jobs=1, verbose=1)
        grid.fit(X, y)

        best_estimator = grid.best_estimator_
        best_params = grid.best_params_

        classificadores.append((nome_classificador, best_estimator, best_params))

    return classificadores

def gerar_metricas_classificador(classificador, X, y, label_encoder):
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = cross_val_predict(classificador, X, y, cv=cv_strategy, n_jobs=1)

    # Relatório e matriz de confusão
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # Decodifica y e y_pred para os nomes originais
    y_true_named = label_encoder.inverse_transform(y)
    y_pred_named = label_encoder.inverse_transform(y_pred)

    # Gera a matriz com os rótulos originais
    labels = label_encoder.classes_
    cm = confusion_matrix(y_true_named, y_pred_named, labels=labels)
    
    f1 = round(report['macro avg']['f1-score'], 4)
    acuracia = round(report['accuracy'], 4)
    recall = round(report['macro avg']['recall'], 4)
    precisao = round(report['macro avg']['precision'], 4)

    metricas = {
        'acuracia': acuracia ,
        'f1-macro': f1,
        'recall-macro': recall,
        'precisao-macro': precisao,
        #'matriz_confusao': cm
    }

    return metricas

def retornar_melhor_classificador(classificadores, metricas):
    melhor_modelo = None
    melhor_f1 = -1
    melhor_metrica = {}
    for i, metrica in enumerate(metricas):
        if metrica['f1-macro'] > melhor_f1:
            melhor_f1 = metrica['f1-macro']
            melhor_modelo = classificadores[i]
            melhor_metrica = metrica
    return melhor_modelo, melhor_metrica
  

def logar_melhor_classificador(classificador,metricas, run):
    run.log(metricas)
    joblib.dump(classificador, 'classificador.pkl')

    artifact = wandb.Artifact('modelo', type='modelo')
    artifact.add_file('classificador.pkl')
    run.log_artifact(artifact)

    if metricas['f1-macro'] >= 0.82 and metricas['acuracia'] >= 0.85:
        artifact = wandb.Artifact('classificador', type='modelo')
        artifact.add_file('classificador.pkl')
        run.log_artifact(artifact)
  

def logar_metricas_classificadores(metricas, run):
     
    tabela = run.Table(columns=["vetorizador", "classificador", "params",
        "acurácia", "f1_macro","recall_macro", "precisão_macro"])
    
    for metrica in metricas:
        tabela.add_data(metrica['vetorizador'], metrica['classificador'], str(metrica['params']),
            metrica['acuracia'], metrica['f1-macro'], metrica['recall-macro'], metrica['precisao-macro'])
        
    run.log({"tabela_classificadores": tabela})

def main():
    nltk.download('stopwords')
    wandb.init(
        project="intencao-dialogar",
        job_type="selecao-modelo",
    )

    df = carregar_dataset_treinamento()

    X = df['texto']
    y = df['tipo']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    grid_tfidf, pipeline_tfidf = gerar_params_grid_tfidf()
    grid_embeddings, pipeline_embeddings = gerar_params_grid_embeddings()
    
    classificadores_tfidf = rodar_grid_search(pipeline_tfidf, grid_tfidf, X, y_encoded)
    classificadores_embeddings = rodar_grid_search(pipeline_embeddings, grid_embeddings, X, y_encoded)

    #cleaned_params = lambda params: { k.replace('clf__', '').replace('vect__', ''): v for k, v in params.items() if 'clf__' in k or 'vect__' in k}

    metricas_tfidf = [
        dict(
            classificador=nome_classificador,
            vetorizador='TF-IDF',
            params=str(param),
            **gerar_metricas_classificador(classificador, X, y_encoded, le)
        )
        for nome_classificador, classificador, param in classificadores_tfidf
    ]

    metricas_embeddings = [
        dict(
            classificador=nome_classificador,
            vetorizador='Embeddings',
            params=str(param),
            **gerar_metricas_classificador(classificador, X, y_encoded, le)
        )
        for nome_classificador, classificador, param in classificadores_embeddings
    ]

    logar_metricas_classificadores(metricas_tfidf+metricas_embeddings,wandb)

    melhor_classificador_tfidf, melhor_metrica_tfidf = retornar_melhor_classificador(classificadores_tfidf, metricas_tfidf)
    melhor_classificador_embeddings, melhor_metrica_embeddings = retornar_melhor_classificador(classificadores_embeddings, metricas_embeddings)

    melhor_classificador_geral, melhor_metrica_geral = retornar_melhor_classificador(
        [classificador for classificador in [melhor_classificador_tfidf, melhor_classificador_embeddings]],
        [metrica for metrica in [melhor_metrica_tfidf, melhor_metrica_embeddings]])
    
    logar_melhor_classificador(melhor_classificador_geral, melhor_metrica_geral, wandb)
    wandb.finish()


if __name__ == "__main__":
    main()
