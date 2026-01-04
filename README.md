# A病院 採血 待ち人数・待ち時間 予測（Streamlit Cloud）

Streamlit Cloudで `Error installing requirements` が出る場合は、
- `runtime.txt` がリポジトリ直下にあるか
- `packages.txt` に `libgomp1` があるか
- requirements の xgboost/numpy の組み合わせが合っているか
を確認してください。
