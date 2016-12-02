

def gated_mean(x, p=0.5, axis=2):
    import theano.tensor as T
    thres = T.shape_padaxis((p * T.mean(x, axis=axis) + 
                            (1 - p) * T.max(x, axis=axis)), 
                            axis=-1)
    mask = T.ge(x, thres)
    g_values = mask*x
    g_means = T.sum(g_values, axis=-1) / T.sum(mask, axis=-1)
    return g_means
