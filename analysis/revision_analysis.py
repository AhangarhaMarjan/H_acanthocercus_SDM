"""Reproduce the additional pre-submission checks from the retained project files."""
from pathlib import Path
import sys, csv, json
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import yaml
root=Path(__file__).resolve().parents[1]/"workflow"
out=Path(__file__).resolve().parent
pkg=out.parent
sys.path.insert(0,str(root/"src"))
from hemiscorpius.grid import cell_area_km2,ref_grid
from hemiscorpius.lambdas import predict
from hemiscorpius.maxent import run_maxent,SPECIES
proc=root/"data_processed"
cfg=yaml.safe_load((root/"config/config.yaml").read_text())
bases=["WC_1970_1999","WC_2000_2024","TC_1970_1999","TC_2000_2024"]
keys=["P10","MTP","maxSSS"]
cols=["10 percentile training presence Cloglog threshold","Minimum training presence Cloglog threshold","Maximum training sensitivity plus specificity Cloglog threshold"]
def mask(f):
    with rasterio.open(f) as ds:return ds.read(1)==1
M=mask(proc/"08_common_mask/common_mask.tif") & mask(proc/"10_accessible_area_background/M_main.tif")
area=cell_area_km2(ref_grid(cfg))[M]
def save(df, name):
    destination = out / "generated"
    destination.mkdir(exist_ok=True)
    df.to_csv(destination / (name + ".csv"), index=False)
def summary(x):return [np.mean(x),*np.percentile(x,[2.5,97.5])]
def curves():
    a=pd.read_csv(proc/"14_final_models/response_curves.csv")
    a=a[(a.variable=="bio12")&(a.kind=="marginal")&(a.variant=="main")&(a["mode"]=="common")&a.source.str.startswith("boot")]
    v={(b,r):np.interp(150,g.sort_values("x").x,g.sort_values("x").y) for (b,r),g in a.groupby(["baseline","source"])}
    return [(v[bases[1],r]+v[bases[3],r]-v[bases[0],r]-v[bases[2],r])/2 for r in sorted(a.source.unique())]
def thresholds():
    records=[]; classmaps=[]; coef=[]
    for b in bases:
        p=proc/"14_final_models/main/common"/b
        for line in (p/"full"/(SPECIES+".lambdas")).read_text().splitlines():
            feature,val,*rest=line.split(",")
            if feature not in ["densityNormalizer","linearPredictorNormalizer","numBackgroundPoints","entropy"] and float(val):coef.append([b,feature,float(val)])
        t=np.array([pd.read_csv(f)[cols].iloc[0].values for f in sorted(p.glob("paired_bootstrap/rep_*/model/maxentResults.csv"))],float)
        assert t.shape==(30,3)
        fixed=pd.read_csv(p/"full/maxentResults.csv")[cols].iloc[0].values
        with xr.open_dataset(proc/"15_projections/main/common"/(b+".nc")) as ds:
            classmaps.append(ds.reference_2000_2024.values>=fixed[0]);lat=ds.lat.values;lon=ds.lon.values
        with xr.open_dataset(proc/"15_projections/main/common"/(b+"_bootstrap.nc")) as ds:
            for ssp in cfg["future"]["ssps"]:
                surf=np.mean([ds["fut_present_"+g+"_"+ssp].values[:,M] for g in cfg["future"]["gcms"]],axis=0)
                assert np.isfinite(surf).all()
                for i in range(30):
                    for k,name in enumerate(keys):records.append([b,i,ssp,name,t[i,k],area[surf[i]>=t[i,k]].sum(),area[surf[i]>=fixed[k]].sum()])
    dat=pd.DataFrame(records,columns=["baseline","rep","scenario","threshold","resample_cutoff","area","fixed_area"]);save(dat,"V4_resample_areas")
    rows=[]
    for (ssp,k),g in dat.groupby(["scenario","threshold"]):
        for v in ["area","fixed_area"]:
            a=g.pivot(index="rep",columns="baseline",values=v)
            x=(a[bases[1]]+a[bases[3]]-a[bases[0]]-a[bases[2]])/2
            rows.append([ssp,k,v,*summary(x)])
    save(pd.DataFrame(rows,columns=["scenario","threshold","policy","mean","lower","upper"]),"V4_threshold_contrasts")
    save(pd.DataFrame(coef,columns=["baseline","feature","lambda"]),"V4_active_features")
    occ=pd.read_csv(proc/"12_maxent_inputs/main"/bases[0]/"samples_index.csv")
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    colors=["#e1e4e4","#216c86","#dc8735","#506c51"]
    iy,ix=np.where(M)
    for j,ax in enumerate(axs):
        a=classmaps[j*2].astype(int)+2*classmaps[j*2+1].astype(int)
        ax.pcolormesh(lon,lat,np.ma.masked_where(~M,a),cmap=ListedColormap(colors),vmin=-.5,vmax=3.5,shading="nearest",rasterized=True)
        ax.scatter(occ.lon,occ.lat,s=6,c="black",label="Calibration locality");ax.set(xlim=(lon[ix].min()-.2,lon[ix].max()+.2),ylim=(lat[iy].min()-.2,lat[iy].max()+.2),xlabel="Longitude (°E)",title=["WorldClim","TerraClimate"][j]);ax.set_aspect(1/np.cos(np.radians(28)))
    axs[0].set_ylabel("Latitude (°N)");fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,["Neither","Earlier only","Recent only","Both"])]+[axs[0].collections[-1]],loc="outside lower center",ncol=5,fontsize=8)
    fig.savefig(out/"C2000.png",dpi=220);plt.close(fig)
    x=curves();assert len(x)==30;save(pd.DataFrame([summary(x)],columns=["mean","lower","upper"]),"V4_BIO12_paired")
def layer(path):
    with xr.open_dataset(path) as ds:return {k:ds[k].values for k in fields if k in ds}
fields=list(pd.read_csv(proc/"12_maxent_inputs/main"/bases[0]/"samples.csv",nrows=0).columns[3:])
def candidate(b):return layer(proc/"08_common_mask"/(b+"_original_candidates.nc"))
def annual(path,years):
    with xr.open_dataset(path) as ds:
        arr=ds.prec.sel(time=slice(str(years[0]),str(years[1])))
        assert len(arr.time)==12*(years[1]-years[0]+1)
        return arr.mean("time",skipna=False).values*12
def fits():
    stats=[]
    ch=proc/"06_independent_precip_check/chirps_v3_monthly_prec_1981_2024.nc"
    chir=[annual(ch,y) for y in [(1981,1999),(2000,2024)]]
    for j,b in enumerate(bases):
        src=proc/"12_maxent_inputs/main"/b; prod=b[:2]; idx=j%2
        base=candidate(b);rc=candidate(prod+"_2000_2024")
        matched=annual(proc/"02_crop_and_units"/(prod.lower()+"_monthly_prec_1970_2024.nc"),[(1981,1999),(2000,2024)][idx])
        for spec in (["published"] if j==0 else [])+["RM1","matched","CHIRPS"]:
            env=dict(base)
            if spec=="published":env.update(layer(proc/"02_crop_and_units/wc21_bio_1970_2000.nc"))
            if spec in ["matched","CHIRPS"]:env["bio12"]= matched if spec=="matched" else chir[idx]
            wd=out/"fits"/spec/b;wd.mkdir(parents=True,exist_ok=True)
            for file in ["samples","background"]:
                tab=pd.read_csv(src/(file+".csv"));ix=pd.read_csv(src/(file+"_index.csv"))
                assert len(tab)==len(ix) and np.allclose(tab.lon,ix.lon) and np.allclose(tab.lat,ix.lat)
                for k in fields:
                    vals=env[k][ix.row,ix.col]
                    assert np.isfinite(vals).all(),(b,spec,k)
                    tab[k]=vals
                tab.to_csv(wd/(file+".csv"),index=False)
            dest=wd/"model";lam=dest/(SPECIES+".lambdas")
            if not lam.exists():run_maxent(root/"maxent/maxent.jar",dest,wd/"samples.csv",wd/"background.csv","LQHP",1 if spec=="RM1" else 5)
            cutoff=pd.read_csv(dest/"maxentResults.csv")[cols[0]].iloc[0]
            for ssp in cfg["future"]["ssps"]:
                preds=[]
                for g in cfg["future"]["gcms"]:
                    future=dict(rc);future.update(layer(proc/"05_future_delta"/f"{prod}_future_present_{g}_{ssp}_2041_2060_original_bioclim.nc"))
                    if spec=="CHIRPS":
                        future["bio12"]=chir[1]*future["bio12"]/np.where(rc["bio12"]>0,rc["bio12"],np.nan)
                    pr=predict(lam,{k:future[k][M].astype(float) for k in fields});assert np.isfinite(pr).all();preds.append(pr)
                pr=np.mean(preds,axis=0);stats.append([spec,b,ssp,cutoff,area[pr>=cutoff].sum(),area@pr])
            print(spec,b,flush=True)
    save(pd.DataFrame(stats,columns=["spec","baseline","ssp","P10","area","weighted"]),"V4_sensitivity")
if __name__=="__main__":
    thresholds();fits()

