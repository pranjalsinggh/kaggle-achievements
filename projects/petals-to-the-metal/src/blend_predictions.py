# Blend only saved, independently trained model predictions.
import glob,json,pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
reports=glob.glob('/kaggle/input/**/validation_report.json',recursive=True)
models=[]
for report_path in reports:
    root=pathlib.Path(report_path).parent
    if not (root/'validation_predictions.npz').exists() or not (root/'test_probabilities.npz').exists():continue
    report=json.loads(pathlib.Path(report_path).read_text())
    va=np.load(root/'validation_predictions.npz',allow_pickle=False)
    te=np.load(root/'test_probabilities.npz',allow_pickle=False)
    models.append((report['model'],va['labels'],va['probabilities'],te['ids'].astype(str),te['probabilities']))
assert len(models)==2,[(m[0],m[2].shape) for m in models]
a,b=models
assert np.array_equal(a[1],b[1]),'Validation order differs; align IDs before blending.'
assert a[2].shape==b[2].shape and a[2].shape[1]==104
assert set(a[3])==set(b[3]) and len(set(a[3]))==len(a[3])
index_b={key:i for i,key in enumerate(b[3])}
btest=b[4][[index_b[key] for key in a[3]]]
results=[]
for weight in [0.0,0.25,0.5,0.75,1.0]:
    probs=weight*a[2]+(1-weight)*b[2]
    score=float(f1_score(a[1],probs.argmax(1),labels=np.arange(104),average='macro',zero_division=0))
    results.append({'weight_a':weight,'validation_macro_f1':score})
    print('BLEND_RESULT',results[-1],flush=True)
best=max(results,key=lambda x:x['validation_macro_f1'])
w=best['weight_a']
ptest=w*a[4]+(1-w)*btest
sample=pd.read_csv(glob.glob('/kaggle/input/**/sample_submission.csv',recursive=True)[0],dtype={'id':str})
assert set(sample.id)==set(a[3]) and len(sample)==len(a[3])
pred=pd.Series(ptest.argmax(1),index=a[3])
out=pd.DataFrame({'id':sample.id,'label':pred.loc[sample.id].to_numpy()})
assert not out.isna().any().any() and out.label.between(0,103).all()
out.to_csv('/kaggle/working/submission.csv',index=False)
report={'models':[a[0],b[0]],'weight_grid':results,'selected':best,'rows':len(out),'note':'Blend selected on supplied validation split. Leaderboard result must be measured separately.'}
pathlib.Path('/kaggle/working/ensemble_report.json').write_text(json.dumps(report,indent=2))
print('ENSEMBLE_READY',report,flush=True)
