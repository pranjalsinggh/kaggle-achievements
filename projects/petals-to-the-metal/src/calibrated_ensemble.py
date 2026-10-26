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

# Guard class-prior calibration with stratified validation folds.
from sklearn.model_selection import StratifiedKFold
pval=w*a[2]+(1-w)*b[2]
y=a[1]
base_score=float(f1_score(y,pval.argmax(1),labels=np.arange(104),average='macro',zero_division=0))
folds=list(StratifiedKFold(3,shuffle=True,random_state=20261025).split(pval,y))
calibration=[]
for alpha in [0.0,0.1,0.2,0.3,0.4]:
    oof=np.zeros_like(pval)
    gains=[]
    for fit_index,eval_index in folds:
        prior=np.bincount(y[fit_index],minlength=104).astype(float)+5
        prior=prior/prior.mean()
        oof[eval_index]=pval[eval_index]*prior[None,:]**(-alpha)
        before=f1_score(y[eval_index],pval[eval_index].argmax(1),labels=np.arange(104),average='macro',zero_division=0)
        after=f1_score(y[eval_index],oof[eval_index].argmax(1),labels=np.arange(104),average='macro',zero_division=0)
        gains.append(float(after-before))
    score=float(f1_score(y,oof.argmax(1),labels=np.arange(104),average='macro',zero_division=0))
    row={'alpha':alpha,'out_of_fold_macro_f1':score,'fold_gains':gains}
    calibration.append(row)
    print('CALIBRATION_CANDIDATE',row,flush=True)
eligible=[x for x in calibration if x['alpha']>0 and min(x['fold_gains'])>=0 and x['out_of_fold_macro_f1']>=base_score+0.0005]
chosen=max(eligible,key=lambda x:x['out_of_fold_macro_f1']) if eligible else calibration[0]
alpha=chosen['alpha']
prior=np.bincount(y,minlength=104).astype(float)+5
prior=prior/prior.mean()
pval_cal=pval*prior[None,:]**(-alpha)
full_score=float(f1_score(y,pval_cal.argmax(1),labels=np.arange(104),average='macro',zero_division=0))
if full_score<base_score+0.0005:
    alpha=0.0
    full_score=base_score
out.to_csv('/kaggle/working/submission-uncalibrated.csv',index=False)
if alpha:
    calibrated=ptest*prior[None,:]**(-alpha)
    pred=pd.Series(calibrated.argmax(1),index=a[3])
    out['label']=pred.loc[sample.id].to_numpy()
    assert not out.isna().any().any() and out.label.between(0,103).all()
    out.to_csv('/kaggle/working/submission.csv',index=False)
cal_report={'base_validation_macro_f1':base_score,'candidates':calibration,'chosen':chosen,'applied_alpha':alpha,'full_validation_macro_f1':full_score,'calibration_accepted':bool(alpha),'rows':len(out)}
pathlib.Path('/kaggle/working/calibration_report.json').write_text(json.dumps(cal_report,indent=2))
print('CALIBRATION_RESULT',cal_report,flush=True)
