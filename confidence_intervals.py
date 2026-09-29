import numpy as np
from scipy.stats import t

def mean_ci(values,confidence=.95):
    x=np.asarray(values,float); n=len(x); mean=x.mean(); se=x.std(ddof=1)/np.sqrt(n); crit=t.ppf((1+confidence)/2,n-1)
    return mean,mean-crit*se,mean+crit*se
