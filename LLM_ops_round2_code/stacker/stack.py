"""Stacker v8: LightGBM on CNN-blend probabilities + level features (scintillation) + track features (ETI).
Train rows = PKS-like OOF probabilities with features from PKS-like-noised train. Evaluated by 5-fold CV."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings, sys, os; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
C=os.environ.get('CACHE','cache/');D=os.environ.get('DATA','data/')
L=np.array(['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise'])
y=pd.read_csv(D+'train_meta.csv').label.map({l:i for i,l in enumerate(L)}).values
SPEC=[('effb2_fixup2',.3),('effb0_fixup2',.3),('effb2_fix',.1),('effb0_fix',.1),('effb2_fix45',.1),('effb3_fix',.1)]
if os.environ.get('SPEC'): SPEC=[(a,float(b)) for a,b in (s.split(':') for s in os.environ['SPEC'].split(','))]
P=sum(w*np.load(C+f'oofpks_{t}.npy') for t,w in SPEC); T=sum(w*np.load(C+f'test_{t}.npy') for t,w in SPEC)
FEATS=os.environ.get('FEATS','lvl,trk').split(',')
def F(split):
    out=[]
    for f in FEATS:
        if f=='': continue
        out.append(np.load(C+f'{f}_{split}.npy'))
    return out
INDIV=os.environ.get('INDIV','')
if INDIV:
    tags=INDIV.split(',')
    Xp=np.column_stack([np.log(P+1e-6)]+[np.log(np.load(C+f'oofpks_{t}.npy')+1e-6) for t in tags]+F(os.environ.get('FSPLIT','trainpks')))
    Xt=np.column_stack([np.log(T+1e-6)]+[np.log(np.load(C+f'test_{t}.npy')+1e-6) for t in tags]+F('test'))
else:
    Xp=np.column_stack([np.log(P+1e-6)]+F(os.environ.get('FSPLIT','trainpks'))); Xt=np.column_stack([np.log(T+1e-6)]+F('test'))
prm=dict(objective='multiclass',num_class=9,learning_rate=float(os.environ.get('LR',0.03)),num_leaves=int(os.environ.get('NL',15)),min_data_in_leaf=int(os.environ.get('MDL',40)),feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1,verbose=-1)
NR=int(os.environ.get('NR',400)); SEEDS=[0,1,2]
oof=np.zeros((len(y),9))
for a,b in StratifiedKFold(5,shuffle=True,random_state=1).split(Xp,y):
    for s in SEEDS: oof[b]+=lgb.train(dict(prm,seed=s),lgb.Dataset(Xp[a],y[a]),NR).predict(Xp[b])/len(SEEDS)
h=np.random.default_rng(0).random(len(y))<0.5
def rep(name,p):
    pc=f1_score(y,p,average=None); f=lambda m: f1_score(y[m],p[m],average='macro')
    print(f'{name:22s} {pc.mean():.4f} halves {f(h):.4f} {f(~h):.4f} | eti {pc[0]:.3f} scint {pc[3]:.3f} rfi_i {pc[6]:.3f} rfi_p {pc[7]:.3f} noise {pc[8]:.3f}')
pa,pb=P.argmax(1),oof.argmax(1)
rep('blend',pa); rep('stack '+','.join(FEATS),pb)
tag=os.environ.get('TAG','v8'); np.save(C+f'oofpks_stack_{tag}.npy',oof)
if os.environ.get('TEST','1')=='1':
    PT=sum(lgb.train(dict(prm,seed=s),lgb.Dataset(Xp,y),NR).predict(Xt) for s in SEEDS)/len(SEEDS); np.save(C+f'test_stack_{tag}.npy',PT)
    print('test counts stack', dict(zip([l[:6] for l in L],np.bincount(PT.argmax(1),minlength=9))), '| blend', dict(zip([l[:6] for l in L],np.bincount(T.argmax(1),minlength=9))))

if os.environ.get('OUT'):
    te=pd.read_csv(D+'test_meta.csv'); sub=pd.DataFrame({'id':te.id,'label':L[PT.argmax(1)]})
    assert len(sub)==10000 and sub.id.is_unique; sub.to_csv(os.environ['OUT'],index=False); print('wrote',os.environ['OUT'])
