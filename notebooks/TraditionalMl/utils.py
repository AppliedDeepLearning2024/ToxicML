from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from pathlib import Path
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import tqdm

def split_by_features(df):
    X = df[["smiles"]]
    try:
        y = df[["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]]
        return X, y
    except:
        pass

    try:
        y = df[["HIV_active"]]
        return X, y
    except:
        pass

    try:
        y = df[["exp"]]
        return X, y
    except:
        raise ValueError("No valid target columns")
    

class ChemFeatureGenerator():
    def fit(self, X, y = None):
        self.droped = None
        return self
    
    def _getMolDescriptors(self, mol, missingVal=None):
        ''' calculate the full list of descriptors for a molecule
        
            missingVal is used if the descriptor cannot be calculated
        '''
        res = {}
        for nm,fn in Descriptors._descList:
            # some of the descriptor fucntions can throw errors if they fail, catch those here:
            try:
                val = fn(mol)
            except:
                # print the error message:
                import traceback
                traceback.print_exc()
                # and set the descriptor value to whatever missingVal is
                val = missingVal
            res[nm] = val
        return res
    
    def transform(self, X):
        df = X["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
        res = df.apply(lambda x: self._getMolDescriptors(x)).values.tolist()
        X = pd.DataFrame(res)
        X = X.replace([np.inf, -np.inf], np.nan)
        return X
    

def split_and_preprocess(base_dir: Path) -> dict:
    generator = ChemFeatureGenerator()
    train = pd.read_csv(base_dir / "train.csv")
    X_train, y_train = split_by_features(train)
    X_train = generator.transform(X_train)
    test = pd.read_csv(base_dir / "test.csv")
    X_test, y_test = split_by_features(test)
    X_test = generator.transform(X_test)
    eval = pd.read_csv(base_dir / "eval.csv")
    X_eval, y_eval = split_by_features(eval)
    X_eval = generator.transform(X_eval)

    return X_train, y_train, X_test, y_test, X_eval, y_eval

def fit_basic_models(X_train, y_train, models):
    pipelines = []
    for name, model in tqdm.tqdm(models):
        pipe = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaller', StandardScaler()),
            ('selector', VarianceThreshold(threshold=0.8*0.2)),
            (name, model),
        ])
        pipe.fit(X_train, y_train.values.ravel())
        pipelines.append( (name, pipe) )
    return pipelines

def evaluate_pipelines(pipelines, metrics, X, y):
    all_res = []
    for model_name, pipe in pipelines:
        y_pred = pipe.predict(X)
        res = {"model": model_name}
        for metric in metrics:
            res[metric.__name__] = metric(y, y_pred)
        all_res.append(res)
    return pd.DataFrame(all_res)


def models_grid_search(models_with_grids, X, y, cv=3, n_iter=10, scoring=None):
    pipelines = []
    for name, model, grid in tqdm.tqdm(models_with_grids):
        pipe = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaller', StandardScaler()),
            ('selector', VarianceThreshold(threshold=0.8*0.2)),
            (name, model),
        ])
        opt = BayesSearchCV(
            pipe,
            grid,
            cv=cv,
            n_iter=n_iter,
            scoring=scoring
        )
        opt.fit(X, y.values.ravel())
        pipelines.append( (name, opt) )
    return pipelines