# Reproducibility checklist

- Record Python, TensorFlow, CUDA and GPU versions.
- Keep the original dataset source and licence information.
- Keep fixed fold indices when redistribution is allowed.
- Save YAML configuration with every experiment.
- Save random seed.
- Save fold-level metrics, not only aggregate means.
- Do not normalize using the complete dataset before CV.
- Do not tune hyperparameters using the outer test folds.
- Do not report generated results as manuscript results until the exact data and architecture are verified.
