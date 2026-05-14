import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('/Users/khusbuagarwal/Downloads/yeast_db.csv')

time_row = df.iloc[0]
data = df.iloc[1:].copy()

data['gene_pos'] = pd.to_numeric(data['gene_pos'], errors='coerce')


start_index = df.columns.get_loc("gene_pos") + 1

ptm_zero_cols = []

for col in df.columns[start_index:]:
    if col.lower().startswith("input"):
        continue
    try:
        if float(time_row[col]) == 0:
            ptm_zero_cols.append(col)
    except:
        pass

print("Number of PTMs:", len(ptm_zero_cols))

positions = [-1, 1, 2, 3, 4, 5]

subset = data[data['gene_pos'].isin(positions)]
subset = subset[['gene', 'gene_pos'] + ptm_zero_cols]

pivot_df = subset.pivot_table(index='gene', columns='gene_pos')

print("\nTotal genes:", pivot_df.shape[0])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

unique_ptms = pivot_df.columns.get_level_values(0).unique()

for ptm in unique_ptms:
    try:
        ptm_df = pivot_df[ptm]

        # Target must exist (-1)
        if -1 not in ptm_df.columns:
            continue

        # FEATURES: +1 to +5
        feature_positions = [1, 2, 3, 4, 5]

        if not all(pos in ptm_df.columns for pos in feature_positions):
            continue

        # Arrange data: target first
        ptm_df = ptm_df[[-1] + feature_positions]

        # Clean data
        ptm_df = ptm_df.dropna(subset=[-1])
        ptm_df[feature_positions] = ptm_df[feature_positions].fillna(0)

        genes_used = len(ptm_df)

        # X = downstream features
        X = ptm_df[feature_positions]

        # y = upstream (-1)
        y = ptm_df[-1]

       
        # POLYNOMIAL MODEL
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])

        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

        mean_r2 = np.mean(scores)
        std_r2 = np.std(scores)

        print(f"PTM: {ptm}")
        print(f"Genes used: {genes_used}")
        print(f"Predict (-1) from (+1 to +5) → R2: {mean_r2:.4f}\n")

        results.append({
            "PTM": ptm,
            "Model": "Poly_LR_deg2_downstream_to_upstream",
            "Genes": genes_used,
            "Mean_R2": mean_r2,
            "Std_R2": std_r2
        })

    except Exception as e:
        print(f"Skipping {ptm}: {e}")
        continue


# FINAL RESULTS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="Mean_R2", ascending=False)

print(results_df.head(20))

results_df.to_csv("PTM_downstream_to_upstream_results.csv", index=False)

print("\nResults saved to PTM_downstream_to_upstream_results.csv")
