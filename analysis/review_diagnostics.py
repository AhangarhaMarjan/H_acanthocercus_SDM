from pathlib import Path
import argparse,importlib.util,sys
import pandas as pd,numpy as np,xarray as xr,rasterio,yaml
def run(root,destination):
 sys.path.insert(0,str(root/'src'))
 from hemiscorpius.mess import mess
 from hemiscorpius.lambdas import predict
 from hemiscorpius.grid import ref_grid,cell_area_km2
 proc=root/'data_processed';cfg=yaml.safe_load((root/'config/config.yaml').read_text())
 sp=importlib.util.spec_from_file_location('projection',root/'scripts/15_projections.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 with rasterio.open(proc/'08_common_mask/common_mask.tif') as d:mask=d.read(1)==1
 with rasterio.open(proc/'10_accessible_area_background/M_main.tif') as d:mask &= d.read(1)==1
 area=cell_area_km2(ref_grid(cfg))[mask];cl=m.Climates(proc,cfg,mask)
 tab=pd.read_csv(destination/'T16_share_M_extrapolated_main.csv').set_index('baseline')
 bases=list(tab.index);active=['bio12','bio6','bio3','bio5','slope']
 samples={b:pd.read_csv(proc/'12_maxent_inputs/main'/b/'samples.csv') for b in bases}
 pooled=pd.concat(samples.values()); names=list(samples[bases[0]].columns[3:])
 fixed=pooled[names].median().to_dict(); fixed['bio12']=150.
 # Quartiles of the intersection of the observed BIO6 ranges, evaluated with identical other predictors.
 lo=max(x.bio6.min() for x in samples.values()); hi=min(x.bio6.max() for x in samples.values())
 temps=lo+np.array([.25,.5,.75])*(hi-lo)
 response=[];overlap=[];areas=[]
 for b in bases:
  model=proc/'14_final_models/main/common'/b/'full'
  cutoff=pd.read_csv(model/'maxentResults.csv')['10 percentile training presence Cloglog threshold'].iloc[0]
  env={k:np.full(3,v) for k,v in fixed.items()};env['bio6']=temps
  for i,val in enumerate(predict(next(model.glob('*.lambdas')),env)):response.append(dict(baseline=b,**{k:v[i] for k,v in env.items()},cloglog=val))
  x=pd.concat([samples[b],pd.read_csv(proc/'12_maxent_inputs/main'/b/'background.csv')]);ref={v:x[v].values for v in names}
  with xr.open_dataset(proc/'15_projections/main/common'/(b+'_bootstrap.nc')) as ds:
   for name,t in m.targets_for(b,'original',cfg,True):
    if not name.startswith('fut_present_'):continue
    target=m.env_for(cl,t,names,b);g=t[3];ssp=t[4]
    for group,variables in [('all',names),('active',active)]:
     novelty=mess({v:ref[v] for v in variables},target)[0]
     overlap.append([b,g,ssp,group,(novelty<0).mean()])
    areas.append((ds[name].values[:,mask]>=cutoff)@area)
 res=pd.DataFrame(overlap,columns=['baseline','gcm','scenario','variables','fraction_M'])
 for (b,ssp),sub in res[res.variables=='all'].groupby(['baseline','scenario']):assert abs(sub.fraction_M.mean()-tab.loc[b,'fut_present_'+ssp])<.00011
 a=np.array(areas);total=((a-a.mean())**2).sum();within=((a-a.mean(axis=1,keepdims=True))**2).sum()/total
 common=a.shape[0]*((a.mean(axis=0)-a.mean())**2).sum()/total
 assert abs(100*within-53.81)<.01
 res.to_csv(destination/'active_predictor_MESS.csv',index=False)
 pd.DataFrame(response).to_csv(destination/'common_environment_BIO12.csv',index=False)
 pd.DataFrame({'term':['shared replicate','replicate by design cell'],'percent_total_P10_SS':100*np.array([common,within-common])}).to_csv(destination/'paired_resampling_partition.csv',index=False)
 print(res.groupby(['baseline','scenario','variables']).fraction_M.mean(),pd.DataFrame(response)[['baseline','bio6','cloglog']],common,within,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('tables',type=Path);args=p.parse_args();run(args.project.resolve(),args.tables.resolve())
