import tensorflow as tf

@tf.function(reduce_retracing=True)
def projected_metric(decoder, z, P, epsilon=1e-3):
    """G ~= J^T P P^T J = (J^T P)(J^T P)^T using directional derivatives.
    P has shape [d,r], z [1,k]. This is intentionally explicit for reproducibility.
    """
    cols=[]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        xhat=decoder(z,training=True)
        for q in range(P.shape[1]):
            scalar=tf.reduce_sum(xhat*P[:,q][None,:],axis=1)
            cols.append(scalar)
    grads=[tape.gradient(s,z) for s in cols]
    del tape
    JtP=tf.concat([g for g in grads],axis=0) # [r,k] for batch size 1
    JtP=tf.reshape(JtP,[P.shape[1],-1])
    G=tf.matmul(JtP,JtP,transpose_a=True)
    k=tf.shape(G)[0]
    G=G+tf.cast(epsilon,G.dtype)*tf.eye(k,dtype=G.dtype)
    return G

def geometric_regularizer(decoder,z,P,delta=1e-3,epsilon=1e-3):
    """Finite-difference ||nabla_z G||_F^2 for a single latent point."""
    z=tf.reshape(z,[1,-1]); k=z.shape[-1]
    G0=projected_metric(decoder,z,P,epsilon)
    total=tf.constant(0.,dtype=z.dtype)
    for j in range(k):
        e=tf.one_hot(j,k,dtype=z.dtype)[None,:]
        gp=projected_metric(decoder,z+delta*e,P,epsilon)
        gm=projected_metric(decoder,z-delta*e,P,epsilon)
        total += tf.reduce_sum(tf.square((gp-gm)/(2.0*delta)))
    return total/tf.cast(k,z.dtype)
