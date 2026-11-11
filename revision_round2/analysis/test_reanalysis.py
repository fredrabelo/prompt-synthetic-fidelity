"""Independent numerical checks; no model calls and no replacement of real data."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
p=Path(__file__).with_name('reanalyze.py')
tree=ast.parse(p.read_text()); functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
ns={'np':np,'pd':pd,'Path':Path,'SEED':20261111};exec(compile(ast.Module(body=functions,type_ignores=[]),str(p),'exec'),ns)
metrics,components,covariance=ns['metrics'],ns['components'],ns['covariance']
rng=np.random.default_rng(42)
# Identical item-specific constants cannot create respondent dispersion.
h=rng.normal(size=(200,11));s=np.tile(np.arange(11)*10,(200,1));v=metrics(h,s)
assert v['sd_pct'][0]<1e-6
# Perfect predictions give zero MAE, unit correlation and SD ratio.
v=metrics(h,h)
assert np.isclose(v['mae'][0],0) and np.isclose(v['r'][0],1) and np.isclose(v['sd_pct'][0],100)
# Independent direct calculations on distinct per-item masks.
s=h+rng.normal(size=h.shape);s[:30,0]=np.nan;h[50:80,2]=np.nan
v=metrics(h,s);manual=[]
for j in range(11):
 ok=np.isfinite(h[:,j])&np.isfinite(s[:,j]);hh=h[ok,j];ss=s[ok,j]
 manual.append([abs(hh-ss).mean(),np.corrcoef(hh,ss)[0,1],100*ss.std()/hh.std()])
assert np.allclose([v['mae'][0],v['r'][0],v['sd_pct'][0]],np.mean(manual,axis=0))
# Orthogonal row, item, interaction projections recover known deterministic terms.
n,j=200,11;r=rng.normal(size=n);r-=r.mean();i=np.arange(j)-5
u=rng.normal(size=(n,j));u=u-u.mean(1)[:,None]-u.mean(0)[None,:]+u.mean()
d=2+r[:,None]+i[None,:]+u;v=components(d,np.zeros((n,j,j)))
assert np.allclose([v['grand_squared'],v['respondent'],v['item'],v['respondent_item']],[4,np.mean(r*r),np.mean(i*i),np.mean(u*u)])
# Noise correction must include correlations among scores within each call.
# Known covariance: common-call noise + item noise. Average over 250 independent simulated studies.
acc=[]
for _ in range(250):
 a=rng.normal(size=(80,1,5))*3+rng.normal(size=(80,11,5))*2
 b=rng.normal(size=(80,1,5))*3+rng.normal(size=(80,11,5))*2
 v=components(b.mean(2)-a.mean(2),(covariance(a)+covariance(b))/5)
 acc.append([v[k] for k in ['grand_squared','respondent','item','respondent_item','total_squared']])
assert np.max(np.abs(np.mean(acc,axis=0)))<.15,np.mean(acc,axis=0)
# Frozen OOF predictions reproduce an independent least-squares fit on training rows only.
root=p.parents[1];release=ns['locate_release'](root);a=pd.read_csv(release/'data/respondents.csv').set_index('external_key');fold=pd.read_csv(root/'results/cv_folds.csv').set_index('external_key').loc[a.index,'fold'].to_numpy()
lev={c:sorted(a[c].unique()) for c in a if c!='age'};x,_=ns['design'](a,lev)
hh=pd.read_csv(release/'data/human_benchmark.csv').pivot(index='external_key',columns='thermometer',values='human_value').reindex(a.index)
keys=__import__('json').loads((release/'data/instrument.json').read_text())['items'];hh=hh[[k['key'] for k in keys]].to_numpy()
pred=np.load(root/'results/predictions.npz')['OLS (OOF)']
for f in range(5):
 for t in range(11):
  tr=(fold!=f)&np.isfinite(hh[:,t]);te=fold==f
  beta=np.linalg.lstsq(x[tr],hh[tr,t],rcond=None)[0]
  assert np.allclose(x[te]@beta,pred[te,t])
# Algebraic MSE decomposition is exact for every model/item on paired masks.
z=pd.read_csv(root/'results/mse_decomposition.csv')
assert np.allclose(z.mse,z.bias_squared+z.scale_mismatch+z.imperfect_association)
# Null ICC should be approximately zero, with negative estimates retained.
ic=pd.read_csv(root/'results/icc_summary.csv');assert (ic[ic.condition=='A'].icc1_mean.abs()<.03).all()
print('PASS: per-item metrics, missing masks, orthogonal components, correlated-noise null, all 55 OOF fits, MSE identity, null ICC.')
# Mirror changes exactly the designated columns; synthetic known-reference ratio equals one.
z=np.load(root/'results/mirror_design.npz');cols=z['columns'];ix=[i for i,c in enumerate(cols) if not c.startswith(('pid=','ideo='))]
assert np.array_equal(z['X'][:,ix],z['X_mirror'][:,ix])
q=rng.normal(5,1,size=(100,11));ss=q.copy();ss[3,:]=np.nan
ns['eh']=np.ones_like(q);ns['eh'][8,2]=np.nan;ns['sgn']=np.r_[np.ones(50),-np.ones(50)]
v,sm,hm=ns['mirror_metrics'](ss,q,np.ones((1,100)))
assert np.allclose(sm,hm) and np.allclose(v['ratio_all'],1) and np.allclose(v['ratio_party'],1)
assert np.allclose(v['spill_model'],v['spill_human'])
print('PASS: nine fixed mirror attributes, aligned human/model estimands, paired missingness and spillover.')
# Longitudinal grouping: no person's outcomes cross folds, cluster draws retain waves.
clusters=pd.read_csv(root/'data/person_clusters.csv').set_index('external_key').person_id
foldtable=pd.read_csv(root/'results/cv_folds.csv')
assert foldtable.groupby('person_id').fold.nunique().max()==1
assert len(clusters)==7526 and clusters.nunique()==6027
ns['person_ids']=clusters;ns['B']=10
for sampled in ns['cluster_indices'](clusters.index):
 counts=np.bincount(sampled,minlength=len(clusters))
 assert pd.DataFrame({'person':clusters.to_numpy(),'count':counts}).groupby('person')['count'].nunique().max()==1
# Full crossed finite-design response components: exact deterministic partition and /4 coding.
n,j,k=80,11,5
r0=rng.normal(size=n);r0-=r0.mean();it=np.arange(j)-5
mid=20+r0[:,None]+it[None,:];delta=2+3*r0[:,None]+np.zeros((n,j))
x=np.repeat((mid-delta/2)[:,:,None],k,axis=2);y=np.repeat((mid+delta/2)[:,:,None],k,axis=2)
parts,total=ns['response_components'](x,y)
assert np.isclose(parts['condition_respondent'],np.var(delta.mean(1))/4)
assert np.isclose(total,np.var(np.stack([x,y])))
assert np.isclose(parts['condition'],1)
rr=pd.read_csv(root/'results/response_variance_decomposition.csv')
assert np.allclose(rr.groupby('model').share.sum(),1)
assert np.allclose(rr.value/rr.total_single_call_variance,rr.share)
cc=pd.read_csv(root/'results/format_decomposition.csv')
for model in rr.model.unique():
 v=rr[(rr.model==model)&(rr.component=='condition_respondent')].value.iloc[0]
 d=cc[(cc.model==model)&(cc.component=='respondent')].value.iloc[0]
 assert np.isclose(v,d/4)
av=pd.read_csv(root/'results/five_draw_fidelity.csv')
assert (av.ols_mae<av.llm_mae).all() and (av.ols_r>av.llm_r).all()
print('PASS: longitudinal folds, cluster multiplicities, crossed variance identity, quarter-scale coding, eight shares, five-call sensitivity.')
