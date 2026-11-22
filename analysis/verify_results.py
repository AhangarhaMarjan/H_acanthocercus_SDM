from revision_analysis import *
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location("proj",root/"scripts/15_projections.py");proj=module_from_spec(spec);spec.loader.exec_module(proj)
for p in (out/"fits").rglob(SPECIES+".lambdas"):
 model=p.parent
 if "factorial" in p.parts:src=proc/"12_maxent_inputs/predictors_original"/model.name
 else:src=model.parent
 bg=src/"background.csv";x=pd.read_csv(bg)
 py,n=proj.validate(p,bg,model/(SPECIES+"_backgroundPredictions.csv"),list(x.columns[3:]))
 assert n==len(x.drop_duplicates(["lon","lat"])) and py<1e-5,(p,n,py)
 print(str(model.relative_to(out)),n,py)
