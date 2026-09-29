# Dataset preparation

The repository expects one row per sample and numeric gene-expression columns. A binary `label` column is required.

If the source dataset has identifiers or annotation columns, keep only the numeric expression matrix plus the binary target. Preserve gene ordering consistently across all experiments.
