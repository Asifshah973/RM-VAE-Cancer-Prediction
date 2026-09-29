import numpy as np

def gaussian_augment(X, y, sigma=0.01, seed=1234):
    rng=np.random.default_rng(seed)
    noise=rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return np.concatenate([X, X+noise]), np.concatenate([y,y])
