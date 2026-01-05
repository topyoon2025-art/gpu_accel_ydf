import argparse
from treeple.datasets import make_trunk_classification
import pandas as pd

# 1) Parse CLI arguments
parser = argparse.ArgumentParser(description='Generate trunk classification dataset')
parser.add_argument('--rows', type=int, required=True, help='Number of rows to generate')
parser.add_argument('--cols', type=int, required=True, help='Number of columns (features) to generate')
args = parser.parse_args()

# 2) Generate data
X, y = make_trunk_classification(args.rows, args.cols, n_informative=5, seed=1)
n_rows, n_cols = X.shape

# 3) Build DataFrame with features + target
feature_names = [f'feature_{i}' for i in range(n_cols)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 4) Define output paths
#    Full dataset: features + target
full_path     = f'./{n_rows}x{n_cols}.csv'
#    Features only: drop the target column
features_path = f'./nl_{n_rows}x{n_cols}.csv'

# 5) Save both CSVs
df.to_csv(full_path, index=False)
#df.drop(columns=['target']).to_csv(features_path, index=False)

print(f"Full dataset written to      {full_path}")
#print(f"Features-only dataset written to {features_path}")
