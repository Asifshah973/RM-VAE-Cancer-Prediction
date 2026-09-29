import argparse
import pandas as pd
from src.evaluation.statistical import wilcoxon_holm
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args(); df=pd.read_csv(a.input); out=wilcoxon_holm(df); out.to_csv(a.output,index=False); print(out.to_string(index=False))
