# Methodology implemented

The implementation follows the manuscript's two-stage formulation:

1. Standardize each gene using training-fold statistics only.
2. For Breast, duplicate the training data with Gaussian perturbation sigma=0.01 when N<150; never augment the held-out fold.
3. Train RM-VAE using the standard isotropic Gaussian prior and the objective consisting of reconstruction error, beta-weighted KL divergence and lambda-weighted geometric smoothness.
4. Construct the decoder-induced pullback metric using a stochastic random-projection Jacobian estimator.
5. Extract the posterior mean `mu(x)` as the deterministic latent representation.
6. Freeze the RM-VAE encoder and train the DNN classifier on the embeddings.
7. Evaluate once on the untouched outer fold.

The manuscript gives formulas and global settings but does not numerically specify hidden-layer widths. Those are therefore configuration values, not claimed manuscript facts.
