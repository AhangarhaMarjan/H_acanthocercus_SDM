from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

def partition(folder):
    df=pd.read_csv(folder/"V4_factorial_cells.csv")
    factors=["predictors","tuning","product","period","ssp","gcm"]
    sizes=df.groupby(factors).size()
    assert len(sizes)==np.prod(df[factors].nunique())==160 and sizes.eq(1).all()
    records=[]
    for metric in ["P10","weighted"]:
        center=df[metric]-df[metric].mean()
        total=center@center
        effects={}
        for n in range(1,len(factors)+1):
            for term in combinations(factors,n):
                effect=df.groupby(list(term))[metric].transform("mean").to_numpy()-df[metric].mean()
                for lower,values in effects.items():
                    if set(lower)<set(term):
                        effect-=values
                effects[term]=effect
                records.append([metric,n," × ".join(term),100*(effect@effect)/total])
        assert np.allclose(sum(effects.values()),center)
    result=pd.DataFrame(records,columns=["metric","order","term","percent_total_SS"])
    assert np.allclose(result.groupby("metric").percent_total_SS.sum(),100)
    old=pd.read_csv(folder/"V4_factorial_shares.csv")
    for metric in result.metric.unique():
        z=result[result.metric==metric]
        observed=z[z.order>1].percent_total_SS.sum()
        expected=old[(old.metric==metric)&(old.term=="all interactions")].percent_SS.item()
        assert np.isclose(observed,expected)
        print(metric,z[z.order>1].nlargest(2,"percent_total_SS").to_string(index=False))
    result.to_csv(folder/"V4_1_factorial_terms.csv",index=False)
if __name__=="__main__":
    partition(Path(__file__).resolve().parent / "generated")

