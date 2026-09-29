# Data

Place legally obtained datasets here. Do not commit patient-identifiable or restricted data.

Recommended CSV format:

`sample_id,label,gene_1,...,gene_d`

The code treats `label` as the binary target and automatically excludes `sample_id` when it is non-numeric.
