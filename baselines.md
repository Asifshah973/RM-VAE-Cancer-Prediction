# Baselines

The manuscript specifies SVC, decision tree, logistic regression, random forest, XGBoost, K-NN, VGG-16, ResNet-50, GA and PSO comparisons.

- Tabular models use the selected manuscript settings in `src/baselines/classical.py` and `src/baselines/xgb.py`.
- GA and PSO use the manuscript's stated population/swarm and iteration settings. Their exact fitness implementation should be checked against the original author code if available because the manuscript does not provide every implementation-level detail.
- VGG-16 and ResNet-50 are configured with random initialization (`weights=None`), matching the manuscript. They require genuine scanner-generated microarray image arrays and are not fed gene-expression vectors.
