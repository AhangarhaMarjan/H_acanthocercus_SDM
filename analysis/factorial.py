"""Balanced full-fit sensitivity across predictor choice, tuning policy and climate inputs."""
from revision_analysis import *
sel=pd.read_csv(proc/"13_tuning_spatial_cv/selected_settings.csv")
rows=[]
for variant in ["main","predictors_original"]:
    for mode in ["common","specific"]:
        for b in bases:
            src=proc/"12_maxent_inputs"/variant/b
            row=sel[(sel.variant==variant)&(sel["mode"]==mode)&(sel.baseline==b)].iloc[0]
            p=proc/"14_final_models"/variant/mode/b/"full"
            if not p.exists():
                p=out/"fits"/"factorial"/b
                if not (p/(SPECIES+".lambdas")).exists():run_maxent(root/"maxent/maxent.jar",p,src/"samples.csv",src/"background.csv",row.fc,row.beta)
            cutoff=pd.read_csv(p/"maxentResults.csv")[cols[0]].iloc[0]
            names=list(pd.read_csv(src/"samples.csv",nrows=0).columns[3:])
            with xr.open_dataset(proc/"08_common_mask"/(b+"_original_candidates.nc")) as ds:
                static={k:ds[k].values[M] for k in names if not k.startswith("bio")}
            for ssp in cfg["future"]["ssps"]:
                for g in cfg["future"]["gcms"]:
                    with xr.open_dataset(proc/"05_future_delta"/f"{b[:2]}_future_present_{g}_{ssp}_2041_2060_original_bioclim.nc") as ds:
                        env={k:ds[k].values[M] for k in names if k.startswith("bio")}
                    pr=predict(p/(SPECIES+".lambdas"),dict(env,**static))
                    assert np.isfinite(pr).all()
                    rows.append([variant,mode,b[:2],b[3:],ssp,g,area[pr>=cutoff].sum(),area@pr,row.fc,row.beta])
df=pd.DataFrame(rows,columns=["predictors","tuning","product","period","ssp","gcm","P10","weighted","fc","rm"])
save(df,"V4_factorial_cells")
rows=[]
for k in ["P10","weighted"]:
    total=((df[k]-df[k].mean())**2).sum(); used=0
    for factor in list(df)[:6]:
        means=df.groupby(factor)[k].transform("mean");share=((means-df[k].mean())**2).sum()/total
        rows.append([k,factor,100*share]);used+=share
    rows.append([k,"all interactions",100*(1-used)])
save(pd.DataFrame(rows,columns=["metric","term","percent_SS"]),"V4_factorial_shares")

