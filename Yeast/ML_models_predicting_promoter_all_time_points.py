import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy import stats

#Load Data

df = pd.read_csv('/Users/khusbuagarwal/Downloads/yeast_db.csv')

df = df.iloc[1:].copy()

df['gene_pos'] = pd.to_numeric(df['gene_pos'], errors='coerce')

df = df.dropna(subset=['gene_pos'])

cols = pd.Series(df.columns)

for dup in cols[cols.duplicated()].unique():
    idx = cols[cols == dup].index.tolist()
    cols[idx] = [dup if i == 0 else f"{dup}.{i}" for i in range(len(idx))]

df.columns = cols

timestamps = ['', '1', '2', '3', '4', '5']

ptm_list = [
'H2AK5ac','H2AS129ph','H3K14ac','H3K18ac','H3K23ac','H3K27ac',
'H3K36me','H3K36me2','H3K36me3','H3K4ac','H3K4me','H3K4me2',
'H3K4me3','H3K56ac','H3K79me','H3K79me3','H3K9ac','H3S10ph',
'H4K12ac','H4K16ac','H4K20me','H4K5ac','H4K8ac','H4R3me',
'H4R3me2s','Htz1'
]

results = []

# CROSS VALIDATION

kf = KFold(n_splits=5, shuffle=True, random_state=42)



def get_rmse(model, X, y, cv):
    neg_mse_scores = cross_val_score(
        model, X, y, cv=cv, scoring='neg_mean_squared_error'
    )
    rmse = np.sqrt(-neg_mse_scores).mean()
    return rmse



for ptm in ptm_list:

    for t in timestamps:

        col_name = ptm if t == '' else f"{ptm}.{t}"

        if col_name not in df.columns:
            continue

        subset = df[['gene', 'gene_pos', col_name]]

        pivot = subset.pivot_table(
            index='gene',
            columns='gene_pos',
            values=col_name
        )

        if -1 not in pivot.columns:
            continue

        needed_positions = [-1, 1, 2, 3, 4, 5]

        for pos in needed_positions:
            if pos not in pivot.columns:
                pivot[pos] = 0

        pivot = pivot[needed_positions]

        pivot = pivot.dropna(subset=[-1])

        pivot[[1, 2, 3, 4, 5]] = pivot[[1, 2, 3, 4, 5]].fillna(0)

        z_scores = np.abs(stats.zscore(pivot[[-1, 1, 2, 3, 4, 5]]))

        pivot = pivot[(z_scores < 3).all(axis=1)]

  # LOG TRANSFORM TARGET
        
        X = pivot[[1, 2, 3, 4, 5]]

        y_raw = pivot[-1]

        # shift to positive before log transform if needed
        shift = 0
        if y_raw.min() <= 0:
            shift = abs(y_raw.min()) + 1e-6

        y = np.log1p(y_raw + shift)


        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        xgb_search = BayesSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1),
            {
                'n_estimators': Integer(100, 600),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'min_child_weight': Integer(1, 10),
                'gamma': Real(0, 5)
            },
            n_iter=20,
            cv=kf,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )

        rf_search = BayesSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 15),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 5),
                'max_features': Real(0.3, 1.0)
            },
            n_iter=20,
            cv=kf,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )

        svr_search = BayesSearchCV(
            SVR(kernel='rbf'),
            {
                'C': Real(0.1, 100, prior='log-uniform'),
                'epsilon': Real(0.01, 1.0, prior='log-uniform'),
                'gamma': Real(1e-4, 1.0, prior='log-uniform')
            },
            n_iter=20,
            cv=kf,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )

        xgb_search.fit(X, y)
        rf_search.fit(X, y)
        svr_search.fit(X_scaled, y)

        models = {
            'Linear': (LinearRegression(), X_scaled),
            'SVM':    (svr_search.best_estimator_, X_scaled),
            'RF':     (rf_search.best_estimator_, X),
            'XGB':    (xgb_search.best_estimator_, X)
        }

        scores_r2 = {}
        scores_rmse = {}

        for name, (model, X_input) in models.items():

            r2 = cross_val_score(
                model, X_input, y, cv=kf, scoring='r2'
            ).mean()

            rmse = get_rmse(model, X_input, y, kf)

            scores_r2[f"{name}_CV_R2"] = r2
            scores_rmse[f"{name}_CV_RMSE"] = rmse


        best_model = max(scores_r2, key=scores_r2.get)

        results.append({
            'PTM': ptm,
            'Timepoint': t if t else 0,
            'Genes': len(pivot),
            'Best_Model': best_model,
            'Best_R2': scores_r2[best_model],
            'Best_RMSE': scores_rmse[best_model.replace('CV_R2', 'CV_RMSE')],
            **scores_r2,
            **scores_rmse
        })

        print(f"{ptm} | time {t if t else 0} | Best: {best_model} "
              f"| R2: {scores_r2[best_model]:.4f} "
              f"| RMSE: {scores_rmse[best_model.replace('CV_R2', 'CV_RMSE')]:.4f}")


# RESULTS


summary_df = pd.DataFrame(results)

summary_df = summary_df.sort_values(by='Best_R2', ascending=False)

print(summary_df)

save_path = "/Users/khusbuagarwal/Downloads/PTM_Timepoint_Model_Results.xlsx"

summary_df.to_excel(save_path, index=False)

print("Saved:", save_path)
