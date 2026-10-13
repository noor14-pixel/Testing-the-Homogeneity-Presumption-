"""
Size-and-power simulation for the cluster-robust Wald homogeneity test
(Punjab MICS 2017-18, child discipline and Learning Index).

Usage:
    python replicate_simulation.py --data help_me.dta --check
    python replicate_simulation.py --data help_me.dta --reps-scale 0.1 --out quick.jsonl
    python replicate_simulation.py --data help_me.dta --out results.jsonl

Requires: numpy, pandas, scipy (no statsmodels needed).
--check only verifies that the baseline reproduces Table 4 (violent items 35.87 on 7 df,
three-domain 21.95 on 8 df, psychological p = 0.897, SD of violent coefficients 0.142).
A full run takes roughly 1 to 1.5 hours on one CPU; --reps-scale 0.1 is a quick test.
Each scenario appends one JSON line to --out, and rerunning skips scenarios already in it.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from scipy import stats

ap=argparse.ArgumentParser()
ap.add_argument('--data',default='help_me.dta'); ap.add_argument('--out',default='results.jsonl')
ap.add_argument('--reps-scale',type=float,default=1.0); ap.add_argument('--check',action='store_true')
ap.add_argument('--seed',type=int,default=20261012)
ARGS=ap.parse_args()

# ---------------- data ----------------
prac=['take_away_privileges','explain_behaviour','shook_child','shouted_yelled_child','gave_something_else','spanked_hit_slapped','hit_bottom_with_item','name_calling_dumb_lazy','hit_face_head_ears','hand_arm_leg','beat_as_hard_one_could']
_df=pd.read_stata(ARGS.data,convert_categoricals=False)
d=pd.DataFrame({'psu':_df.psu})
for p in prac: d[p]=_df[p]
for e in ['EC6','EC7','EC8']: d[e]=(_df[e]==1).astype(float)   # don't-know counts as not affirmative
d['y']=d[['EC6','EC7','EC8']].sum(axis=1)                         # Learning Index 0..3
d['age']=_df.cage; d['female']=(_df.HL4==2).astype(float)
d['disab']=(_df.cdisability==1).astype(float).where(_df.cdisability.notna())
d['urban']=_df.Urban.astype(float); d['school']=_df.school
d['mel']=_df.melevel; d['wq']=_df.windex5.where(_df.windex5>0)
c=d.dropna(subset=prac+['y','age','female','disab','urban','school','mel','wq']).reset_index(drop=True)
assert len(c)==15590, len(c)
def design(c):
    X=c[prac].copy()
    for v in ['age','female','disab','urban','school']: X[v]=c[v]
    for k in [1,2,3,4]: X[f'edu{k}']=(c.mel==k).astype(float)
    for k in [2,3,4,5]: X[f'w{k}']=(c.wq==k).astype(float)
    return X

# ---------------- ordered logit + cluster-robust Wald ----------------
def F(z): return 0.5*(1+np.tanh(z/2))
def f(z):
    e=F(z); return e*(1-e)
class OL:
    """ordered logit, 4 categories, y in 0..3; params = [beta(k), tau0,tau1,tau2]"""
    def __init__(s,X,y,g):
        s.X=X; s.y=y; s.k=X.shape[1]; s.n=len(y)
        s.G=g  # cluster ids 0..G-1
        s.ng=g.max()+1
    def _parts(s,th):
        b=th[:s.k]; t=th[s.k:]
        xb=s.X@b
        T=np.concatenate([[-40.],t,[40.]])
        u=T[s.y+1]-xb; l=T[s.y]-xb
        return xb,u,l,t
    def ll(s,th):
        xb,u,l,t=s._parts(th)
        p=np.clip(F(u)-F(l),1e-300,None); return np.log(p).sum()
    def scores(s,th):
        xb,u,l,t=s._parts(th)
        fu=f(u); fl=f(l); p=np.clip(F(u)-F(l),1e-300,None)
        gb=-(fu-fl)/p
        S=np.empty((s.n,s.k+3)); S[:,:s.k]=s.X*gb[:,None]
        for j in range(3):
            S[:,s.k+j]=np.where(s.y==j,fu/p,0)-np.where(s.y==j+1,fl/p,0)
        return S
    def grad(s,th): return s.scores(th).sum(0)
    def hess(s,th,h=1e-5):
        m=len(th); H=np.empty((m,m))
        for i in range(m):
            e=np.zeros(m); e[i]=h
            H[:,i]=(s.grad(th+e)-s.grad(th-e))/(2*h)
        return (H+H.T)/2
    def fit(s,th0,tol=1e-8,maxit=50):
        th=th0.copy()
        for it in range(maxit):
            g=s.grad(th); H=s.hess(th)
            step=np.linalg.solve(H,-g)  # H negative definite
            ll0=s.ll(th); a=1.
            while a>1e-4:
                new=th+a*step
                if np.all(np.diff(new[s.k:])>0) and s.ll(new)>=ll0-1e-9: break
                a/=2
            th=new
            if np.abs(a*step).max()<tol: break
        s.th=th; return th
    def cov(s,th=None):
        th=s.th if th is None else th
        H=s.hess(th); Hi=np.linalg.inv(-H)
        S=s.scores(th)
        Sg=np.zeros((s.ng,S.shape[1])); np.add.at(Sg,s.G,S)
        Gc=s.ng
        B=(Gc/(Gc-1))*Sg.T@Sg
        return Hi@B@Hi
def wald(b,V,R):
    r=R@b; W=float(r@np.linalg.solve(R@V@R.T,r)); return W,R.shape[0],stats.chi2.sf(W,R.shape[0])
def eqR(idx,m):
    R=np.zeros((len(idx)-1,m))
    for i in range(len(idx)-1): R[i,idx[0]]=1; R[i,idx[i+1]]=-1
    return R

# ---------------- simulation library ----------------
X0=design(c); names=list(X0.columns); Xv=X0.values.astype(float)
y_obs=c.y.astype(int).values
gid=pd.factorize(c.psu)[0]; G=gid.max()+1
K=len(names)
_m=OL(Xv,y_obs,gid)
th_hat=_m.fit(np.concatenate([np.zeros(K),[-0.5,0.5,1.5]]))
viol=['shook_child','shouted_yelled_child','spanked_hit_slapped','hit_bottom_with_item','name_calling_dumb_lazy','hit_face_head_ears','hand_arm_leg','beat_as_hard_one_could']
phys=['shook_child','spanked_hit_slapped','hit_bottom_with_item','hit_face_head_ears','hand_arm_leg','beat_as_hard_one_could']
psy=['shouted_yelled_child','name_calling_dumb_lazy']
nonv=['take_away_privileges','explain_behaviour','gave_something_else']
ix=lambda L:[names.index(v) for v in L]
M=K+3
Rs={'violent8':eqR(ix(viol),M),'physical':eqR(ix(phys),M),'psych':eqR(ix(psy),M),
    'domain3':np.vstack([eqR(ix(nonv),M),eqR(ix(psy),M),eqR(ix(phys),M)])}
def null_theta(kind):
    """true parameter vector under a null"""
    th=th_hat.copy()
    # refit restricted: average coefficients weighted by prevalence-adjusted information is overkill;
    # use restricted MLE by fitting a model with grouped regressors
    groups={'violent8':[viol],'domain3':[nonv,psy,phys]}[kind]
    cols=[j for j in range(K) if names[j] not in sum(groups,[])]
    Z=[Xv[:,cols]]+[Xv[:,ix(gp)].sum(1,keepdims=True) for gp in groups]
    Z=np.hstack(Z)
    m=OL(Z,y_obs,gid)
    st=np.concatenate([th_hat[cols],[th_hat[ix(gp)].mean() for gp in groups],th_hat[K:]])
    t=m.fit(st)
    out=np.zeros(M); out[cols]=t[:len(cols)]
    for gi,gp in enumerate(groups):
        for v in gp: out[names.index(v)]=t[len(cols)+gi]
    out[K:]=t[len(cols)+len(groups):]
    return out
def alt_theta(delta,base):
    th=base.copy(); b=th_hat[ix(viol)]
    th[ix(viol)]=base[ix(viol)]+delta*(b-b.mean())
    return th
def psy_alt(d,base):
    th=base.copy(); m=th[ix(psy)].mean()
    th[ix(psy)]=[m+d/2,m-d/2]; return th
def draw_y(Xs,gs,ng,th,sig,rng):
    xb=Xs@th[:K]
    if sig>0: xb=xb+sig*rng.standard_normal(ng)[gs]
    e=rng.logistic(size=len(xb))
    ystar=xb+e
    return np.searchsorted(th[K:],ystar).astype(int)
def sub_idx(N,rng):
    if N is None: return np.arange(len(y_obs)),gid,G
    cl=rng.permutation(G); sel=[];n=0
    sizes=np.bincount(gid)
    for k in cl:
        sel.append(k); n+=sizes[k]
        if n>=N: break
    mask=np.isin(gid,sel); idx=np.where(mask)[0]
    gs=pd.factorize(gid[idx])[0]; return idx,gs,gs.max()+1
def one_rep(th_true,sig,N,rng,start):
    idx,gs,ng=sub_idx(N,rng)
    Xs=Xv[idx]
    y=draw_y(Xs,gs,ng,th_true,sig,rng)
    m=OL(Xs,y,gs)
    try:
        th=m.fit(start); V=m.cov()
    except Exception as e:
        return None
    out={k:wald(th,V,R)[2] for k,R in Rs.items()}
    Vn=np.linalg.inv(-m.hess(th)); out['ser']=float(np.median(np.sqrt(np.diag(V)/np.diag(Vn))[:K]))
    return out

if ARGS.check:
    V=_m.cov(); th=th_hat
    print('N',len(c),'loglik',round(_m.ll(th),1))
    for k in ['violent8','physical','psych','domain3']:
        print(k,[round(x,4) for x in wald(th,V,Rs[k])])
    print('SD of 8 violent coefficients',round(float(th[ix(viol)].std(ddof=1)),3))
    raise SystemExit

# ---------------- run ----------------
base_v=null_theta('violent8'); base_d=null_theta('domain3')
jobs=[
 ('null_v8',base_v,None,1.3,None,600),
 ('null_d3',base_d,None,1.3,None,400),
 ('v8_d1.0',base_v,('v',1.0),1.3,None,200),
 ('psy_0.2',base_v,('p',0.2),1.3,None,200),
 ('psy_0.1',base_v,('p',0.1),1.3,None,200),
 ('v8_d0.5',base_v,('v',0.5),1.3,None,200),
 ('psy_0.3',base_v,('p',0.3),1.3,None,200),
 ('null_v8',base_v,None,1.3,5000,300),
 ('v8_d1.0',base_v,('v',1.0),1.3,5000,200),
 ('psy_0.2',base_v,('p',0.2),1.3,5000,200),
 ('null_v8',base_v,None,1.3,2000,300),
 ('v8_d1.0',base_v,('v',1.0),1.3,2000,200),
 ('psy_0.2',base_v,('p',0.2),1.3,2000,200),
]
rng=np.random.default_rng(ARGS.seed)
import os
done=set((json.loads(l)['scenario'],json.loads(l)['sigma'],json.loads(l)['N']) for l in open(ARGS.out)) if os.path.exists(ARGS.out) else set()
for name,base,tr,sig,N,R in jobs:
    R=max(10,int(round(R*ARGS.reps_scale)))
    if (name,sig,N or len(y_obs)) in done: continue
    th=base if tr is None else (alt_theta(tr[1],base) if tr[0]=='v' else psy_alt(tr[1],base))
    t=time.time(); rows=[]
    for _ in range(R):
        r=one_rep(th,sig,N,rng,th)
        if r: rows.append(r)
    P={k:np.array([r[k] for r in rows]) for k in ['violent8','physical','psych','domain3','ser']}
    res={'scenario':name,'sigma':sig,'N':N or len(y_obs),'R':len(rows),'secs':round(time.time()-t),
         'rej05':{k:float((P[k]<0.05).mean()) for k in Rs},'rej01':{k:float((P[k]<0.01).mean()) for k in Rs},
         'ser':float(P['ser'].mean())}
    open(ARGS.out,'a').write(json.dumps(res)+'\n')
    print(res,flush=True)
