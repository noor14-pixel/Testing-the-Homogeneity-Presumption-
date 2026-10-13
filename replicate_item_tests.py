"""
Homogeneity tests for the Punjab MICS 2017-18 child discipline module.

Reproduces (1) the main Wald tests in the ordered logit for the Learning Index
(violent items 35.87 on 7 df; three-domain 21.95 on 8 df; psychological p = 0.897)
and (2) the same tests with each ECDI item (EC6 letters, EC7 words, EC8 numbers)
as a separate binary outcome in a logit (Online Resource 1, Section S13, Table S21).

Usage:   python replicate_item_tests.py --data help_me.dta --out item_results.json
Requires: numpy, pandas, scipy (no statsmodels). Runs in a few seconds.
Standard errors are clustered on the PSU with a small-sample factor of G/(G-1).
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from scipy import stats

ap=argparse.ArgumentParser()
ap.add_argument('--data',default='help_me.dta'); ap.add_argument('--out',default='item_results.json')
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

# ---------------- binary logit with the same clustered sandwich ----------------
def logit_fit(X,y,g,maxit=50):
    n,k=X.shape; b=np.zeros(k); ng=g.max()+1
    for _ in range(maxit):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)
        H=-(X*W[:,None]).T@X; s=X.T@(y-p)
        step=np.linalg.solve(H,-s); b=b+step
        if np.abs(step).max()<1e-10: break
    p=1/(1+np.exp(-(X@b))); W=p*(1-p); Hi=np.linalg.inv((X*W[:,None]).T@X)
    S=X*(y-p)[:,None]; Sg=np.zeros((ng,k)); np.add.at(Sg,g,S)
    V=Hi@((ng/(ng-1))*Sg.T@Sg)@Hi
    return b,V

# ---------------- setup ----------------
X0=design(c); names=list(X0.columns)
gid=pd.factorize(c.psu)[0]
viol=['shook_child','shouted_yelled_child','spanked_hit_slapped','hit_bottom_with_item','name_calling_dumb_lazy','hit_face_head_ears','hand_arm_leg','beat_as_hard_one_could']
phys=['shook_child','spanked_hit_slapped','hit_bottom_with_item','hit_face_head_ears','hand_arm_leg','beat_as_hard_one_could']
psy=['shouted_yelled_child','name_calling_dumb_lazy']
nonv=['take_away_privileges','explain_behaviour','gave_something_else']
def tests(b,V,nm,m):
    ix=lambda L:[nm.index(v) for v in L]
    Rs={'violent8 (7 df)':eqR(ix(viol),m),
        'three-domain (8 df)':np.vstack([eqR(ix(nonv),m),eqR(ix(psy),m),eqR(ix(phys),m)]),
        'physical (5 df)':eqR(ix(phys),m),'psychological (1 df)':eqR(ix(psy),m),
        'non-violent (2 df)':eqR(ix(nonv),m)}
    out={k:wald(b,V,R) for k,R in Rs.items()}
    return out,float(b[ix(viol)].std(ddof=1))
res={}
# (1) Learning Index, ordered logit
Xv=X0.values.astype(float); y=c.y.astype(int).values
m=OL(Xv,y,gid); th=m.fit(np.concatenate([np.zeros(Xv.shape[1]),[-0.5,0.5,1.5]])); V=m.cov()
t,sd=tests(th,V,names,len(th)); res['Learning Index (ordered logit)']={'loglik':m.ll(th),'tests':t,'sd_violent':sd}
# (2) each ECDI item, logit with constant
Xc=np.column_stack([np.ones(len(c)),Xv]); nm=['const']+names
for item,lab in [('EC6','letters'),('EC7','words'),('EC8','numbers')]:
    b,V=logit_fit(Xc,c[item].values.astype(float),gid)
    t,sd=tests(b,V,nm,len(b)); res[f'{item} ({lab}), logit']={'share_one':float(c[item].mean()),'tests':t,'sd_violent':sd}
for k,r in res.items():
    print('\n'+k+('  loglik %.1f'%r['loglik'] if 'loglik' in r else '  share=1: %.3f'%r['share_one']))
    for tn,(W,df,p) in r['tests'].items(): print('   %-22s chi2 = %7.2f  df = %d  p = %.4f'%(tn,W,df,p))
    print('   SD of the 8 violent coefficients: %.3f'%r['sd_violent'])
json.dump(res,open(ARGS.out,'w'),indent=1,default=float)
