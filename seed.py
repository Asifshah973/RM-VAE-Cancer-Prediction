import os, random, numpy as np

def set_seed(seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try: tf.config.experimental.enable_op_determinism()
        except Exception: pass
    except Exception:
        pass
