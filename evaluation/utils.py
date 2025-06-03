import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


    
def lgbm_train (df, label,lgbm_params, seed: int):
    
    np.random.seed(seed)
  
    X = np.array(df["ebd"].tolist())  
    y = df[label].values 
    
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    default_params = {
    'objective': 'regression',
    'random_state': seed,
    'verbose': -1,
}

    model_params = default_params.copy()
    model_params.update(lgbm_params)

   
    lgb_model = lgb.LGBMRegressor(**model_params)
    lgb_model.fit(X_train, y_train)
    

    y_train_pred = lgb_model.predict(X_train)
    lgb_train_r2 = r2_score(y_train, y_train_pred)
    lgb_train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

   
    y_test_pred = lgb_model.predict(X_test)
    lgb_test_r2 = r2_score(y_test, y_test_pred)
    lgb_test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    
    results = {
        # "train_rmse": lgb_train_rmse,
        # "train_r2": lgb_train_r2,
        "test_r2": lgb_test_r2,
        "test_rmse": lgb_test_rmse,
        "y": y_test,
        "y_pred": y_test_pred
    }
 
 
    return results


def ridge_train ( df,label , seed: int, k: int = 5):

   
    np.random.seed(seed)

    X = np.array(df["ebd"].tolist())  
    y = df[label].values 
    
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
   
    model =  Ridge(alpha=1, max_iter=10000)  


    y_true_all = []  
    y_pred_all = [] 


    # 5-Fold cross validation
    for i, (train_index, val_index) in enumerate(kf.split(X)):  

        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]


        model.fit(X_train_fold, y_train_fold)


        y_val_pred = model.predict(X_val_fold)
        y_train_pred = model.predict(X_train_fold) 

        y_true_all.extend(y_val_fold)
        y_pred_all.extend(y_val_pred)


    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)


    final_r2 = r2_score(y_true_all, y_pred_all)
    final_rmse = mean_squared_error(y_true_all, y_pred_all, squared=False)


    results = {
        "test_r2" : final_r2,
        "test_rmse" : final_rmse,
        "y": y_true_all,
        "y_pred": y_pred_all
    }


    return results