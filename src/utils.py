import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def convert_scores_to_ranks(scores):
    n = len(scores)
    order = scores.argsort()[::-1]
    pos = np.empty_like(order)
    pos[order] = np.arange(1, n + 1)
    R = (pos - 0.5) / n
    return R


def train_xgb_simple(X, y, random_state=42):
    # Abilita l'uso della GPU se disponibile, usando il nuovo parametro 'device'
    device_param = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizzo di XGBoost con device='{device_param}'")
    
    base = XGBClassifier(
        random_state=random_state, 
        eval_metric="logloss",
        device=device_param
    )
    grid = GridSearchCV(base, param_grid={"n_estimators": [50, 100]}, scoring="f1", cv=3, refit=True)
    grid.fit(X, y)
    return {"model": grid.best_estimator_}

def preprocessing_data(df, y_name='income', dataset_name='adults'):
    """Preprocess a dataset for the fairness experiments.

    Supported datasets: 'adults', 'propublica', 'law', 'edu'.
    """
    df = df.copy()

    if dataset_name == 'adults':
        df['native-country'] = (df['native-country'] == 'United-States').astype(int)
        df['marital-status'] = df['marital-status'].apply(
            lambda x: 'Married' if 'Married' in x else 'Not-Married')
        df['workclass'] = df['workclass'].apply(
            lambda x: 'Private' if x == 'Private' else 'Non-Private')
        df['education'] = df['education'].apply(
            lambda x: 'Bachelors-or-Above'
            if x in ['Bachelors', 'Masters', 'Doctorate'] else 'Non-Bachelors')
        df = df.drop(columns=['relationship', 'occupation'])
        df[y_name] = (df[y_name].isin(['>50K', '>50K.'])).astype(int)

    elif dataset_name == 'propublica':
        df.drop(columns=['Asian', 'Hispanic', 'Native_American', 'Other'],
                inplace=True)
        df[y_name] = (df[y_name] == 1).astype(int)

    elif dataset_name == 'law':
        # Drop redundant race columns (keep race_nonwhite)
        cols_to_drop = [c for c in ['race', 'race_simpler'] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    elif dataset_name == 'edu':
        # Student performance dataset – no special preprocessing needed
        pass

    else:
        raise ValueError(
            f"dataset_name must be one of 'adults', 'propublica', 'law', 'edu'. "
            f"Got: '{dataset_name}'"
        )

    df = df.replace('?', np.nan).dropna().drop_duplicates()
    return df
    

def load_dataset(dataset_name, data_dir="/home/carloabrate/iii/data"):
    """Load and preprocess a dataset, returning (df, y_name, features_Z).

    Supported datasets: 'adults', 'propublica', 'law', 'edu'.
    """
    import os

    CONFIGS = {
        "adults": {
            "file": "adult.csv",
            "reader": lambda p: pd.read_csv(p, sep=','),
            "y_name": "income",
            "features_Z": [['gender'], ['race'], ['gender', 'race']],
        },
        "propublica": {
            "file": "propublica_data_for_fairml.csv",
            "reader": lambda p: pd.read_csv(p, sep=','),
            "y_name": "Two_yr_Recidivism",
            "features_Z": [['African_American'], ['Female'], ['African_American', 'Female']],
        },
        "law": {
            "file": "clean_LawSchool.csv",
            "reader": lambda p: pd.read_csv(p, sep='|'),
            "y_name": "LSAT",
            "features_Z": [['sex'], ['race_nonwhite'], ['sex', 'race_nonwhite']],
        },
        "edu": {
            "file": "student_performance.json",
            "reader": lambda p: pd.read_json(p, orient='records'),
            "y_name": "G3",
            "features_Z": [['sex'], ['address'], ['sex', 'address']],
        },
    }

    if dataset_name not in CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(CONFIGS.keys())}"
        )

    cfg = CONFIGS[dataset_name]
    path = os.path.join(data_dir, cfg["file"])
    df_raw = cfg["reader"](path)
    df = preprocessing_data(df_raw, y_name=cfg["y_name"], dataset_name=dataset_name)

    print(f"Dataset '{dataset_name}': {df.shape[0]} rows, {df.shape[1]} cols")
    return df, cfg["y_name"], cfg["features_Z"]


def provide_x_z(df_i, y_name='income', f_p=['race', 'gender', 'age'], sample_size_per_class=1000, fz = [], random_state=42):
    # Get balanced sample using pandas groupby and sample
    # get indices of a balanced shuffled sample (per class)
    sample_size_per_class = min(sample_size_per_class, df_i[y_name].value_counts().min())
    indices = (df_i.groupby(y_name, group_keys=False)
               .apply(lambda x: x.sample(n=sample_size_per_class, random_state=random_state)).index) # .to_numpy() is fine too
    print(df_i.columns)
    # get X, Z, y for the balanced sample
    y_all = df_i[y_name]
    # Use .loc for label-based indexing
    y = y_all.loc[indices] 
    
    # get X, Z for the entire dataset (for training the classifier)
    # Flatten fz (list of lists) and add y_name to get all columns to drop
    cols_to_drop = list({col for sublist in fz for col in sublist}) + [y_name]
    X_df = df_i.drop(columns=cols_to_drop)
    Z_df = df_i[f_p]
    X_np_all = pd.get_dummies(X_df, drop_first=True, dtype=int)
    Z_np_all = pd.get_dummies(Z_df, drop_first=True, dtype=int)
    
    # Use .loc for label-based indexing here as well
    X_np = X_np_all.loc[indices].values
    Z_np = Z_np_all.loc[indices].values

    return X_np, Z_np, y.values, X_np_all.values, Z_np_all.values, y_all.values