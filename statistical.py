import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def wilcoxon_holm(df,proposed='RM-VAE+DNN',metric='accuracy'):
    methods=[m for m in df.method.unique() if m!=proposed]; rows=[]
    for m in methods:
        a=df[df.method==proposed].sort_values('fold')[metric].to_numpy(); b=df[df.method==m].sort_values('fold')[metric].to_numpy()
        if len(a)!=len(b): continue
        try: stat,p=wilcoxon(a,b,zero_method='wilcox',alternative='two-sided')
        except ValueError: stat,p=float('nan'),1.0
        rows.append({'comparison':f'{proposed} vs {m}','wilcoxon_stat':stat,'p_raw':p})
    if rows:
        adj=multipletests([r['p_raw'] for r in rows],method='holm')[1]
        for r,p in zip(rows,adj): r['p_holm']=p
    return pd.DataFrame(rows)
