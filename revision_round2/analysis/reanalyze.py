"""Round-two analyses from frozen model draws. No API calls or model generation.
Run from any directory: OPENBLAS_NUM_THREADS=1 python3 reanalyze.py
All resampling uses explicit numpy generators. CV intervals condition on OOF fits.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def locate_release(root):
    """Directory holding data/ and analysis/common.py of the first-version package.
    In a repository clone it is the parent of revision_round2/; in the replication
    archive it is the sibling folder github_release/."""
    for cand in (root.parent, root.parent/'github_release'):
        if (cand/'analysis/common.py').exists() and (cand/'data/respondents.csv').exists():
            return cand
    raise FileNotFoundError('first-version data and analysis/common.py not found next to revision_round2/')
RELEASE=locate_release(ROOT)
sys.path.insert(0,str(RELEASE/'analysis'))
from common import load_responses,load_human_and_attrs,wide,MODELS,THERMS,sub250_keys
from a4_reliability import draws_array,noise_prep,noise_stats
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
B=1000; SEED=20261111

def save(name,rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)

def ci(a):return np.nanpercentile(a,[2.5,97.5])
def corr(x,y):
    x=x-x.mean();y=y-y.mean();d=np.sqrt((x*x).sum()*(y*y).sum())
    return (x*y).sum()/d if d>1e-12 else np.nan

def metrics(h,s,w=None):
    mask=np.isfinite(h)&np.isfinite(s); hh=np.where(mask,h,0);ss=np.where(mask,s,0)
    if w is None:w=np.ones((1,len(h)))
    n=w@mask;mh=w@hh/n;ms=w@ss/n
    vh=np.maximum(w@(hh*hh)/n-mh*mh,0);vs=np.maximum(w@(ss*ss)/n-ms*ms,0)
    cov=w@(hh*ss)/n-mh*ms
    with np.errstate(divide='ignore',invalid='ignore'):
        cr=cov/np.sqrt(vh*vs)
    cr=np.where((vh>1e-10)&(vs>1e-10),cr,np.nan)
    return {'mae':(w@np.abs(hh-ss)/n).mean(1), 'r':np.nanmean(cr,axis=1),
            'sd_pct':(100*np.sqrt(vs/vh)).mean(1), 'abs_mean_bias':np.abs(ms-mh).mean(1)}

def cluster_indices(keys,seed=SEED):
    codes,unique=pd.factorize(person_ids.loc[list(keys)].to_numpy())
    members=[np.flatnonzero(codes==i) for i in range(len(unique))]
    rng=np.random.default_rng(seed)
    for _ in range(B):
        yield np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])

def weights(keys,seed=SEED):
    keys=list(keys);batch=[]
    for ids in cluster_indices(keys,seed):
        batch.append(np.bincount(ids,minlength=len(keys)).astype(float))
        if len(batch)==100:yield np.stack(batch);batch=[]
    if batch:yield np.stack(batch)

def design(a,levels,interaction=False):
    cols=[np.ones(len(a)),a.age.to_numpy(float)/100];names=['intercept','age/100']
    for c,lv in levels.items():
        for v in lv[1:]:cols.append((a[c]==v).to_numpy(float));names.append(c+'='+str(v))
    if interaction:
        pc=[i for i,n in enumerate(names) if n.startswith('pid=')];ic=[i for i,n in enumerate(names) if n.startswith('ideo=')]
        for p in pc:
            for i in ic:cols.append(cols[p]*cols[i]);names.append(names[p]+':'+names[i])
    return np.stack(cols,axis=1),names

print('loading',flush=True)
L=load_responses();hw,attrs=load_human_and_attrs();H=hw.to_numpy(float);N=len(attrs)
levels={c:sorted(attrs[c].unique()) for c in attrs.columns if c!='age'}
X,names=design(attrs,levels)
person_ids=pd.read_csv(ROOT/'data/person_clusters.csv').set_index('external_key').person_id
assert person_ids.index.is_unique and person_ids.loc[attrs.index].nunique()==6027
codes,unique=pd.factorize(person_ids.loc[attrs.index].to_numpy())
rng=np.random.default_rng(SEED);perm=rng.permutation(len(unique));groupfold=np.empty(len(unique),int);groupfold[perm]=np.arange(len(unique))%5
fold=groupfold[codes]
save('cv_folds',{'external_key':attrs.index,'person_id':person_ids.loc[attrs.index].to_numpy(),'fold':fold})
P={k:np.full_like(H,np.nan) for k in ['Year mean (OOF)','OLS (OOF)','Year median (OOF)','Year mean (in sample)']}
for t in range(11):
    valid=np.isfinite(H[:,t])
    for yr in attrs.year.unique():
        ids=(attrs.year.to_numpy()==yr)
        P['Year mean (in sample)'][ids,t]=H[ids&valid,t].mean()
    for f in range(5):
        train=(fold!=f)&valid;test=fold==f
        beta=np.linalg.lstsq(X[train],H[train,t],rcond=None)[0]
        P['OLS (OOF)'][test,t]=X[test]@beta
        for yr in attrs.year.unique():
            tr=train&(attrs.year.to_numpy()==yr);te=test&(attrs.year.to_numpy()==yr)
            P['Year mean (OOF)'][te,t]=H[tr,t].mean();P['Year median (OOF)'][te,t]=np.median(H[tr,t])
for m in MODELS:
    for c in 'ABC':P[m+'/'+c]=wide(L.resp,m,c,keys=attrs.index,block='full').to_numpy(float)
np.savez_compressed(OUT/'predictions.npz',**{k.replace('/','_'):v for k,v in P.items()})
print('benchmarks and per-item fidelity',flush=True)
rows=[];items=[];yearrows=[];comparison=[];decomp=[]
for label,S in P.items():
    pt=metrics(H,S);acc={k:[] for k in pt}
    for w in weights(attrs.index):
        for k,v in metrics(H,S,w).items():acc[k].extend(v)
    row={'predictor':label}
    for k in pt:
        row[k]=pt[k][0]
        if k!='abs_mean_bias':row[k+'_lo'],row[k+'_hi']=ci(acc[k])
    rows.append(row)
    for t,key in enumerate(THERMS):
        ok=np.isfinite(H[:,t])&np.isfinite(S[:,t]);h=H[ok,t];s=S[ok,t]
        items.append({'predictor':label,'item':key,'n':len(h),'mae':np.abs(s-h).mean(),'r':corr(h,s),'sd_human':h.std(),'sd_synthetic':s.std(),'sd_pct':100*s.std()/h.std(),'signed_bias':(s-h).mean(),'mse':((s-h)**2).mean()})
    for yr in [2016,2020]:
        ok=attrs.year.to_numpy()==yr;yearrows.append({'predictor':label,'year':yr,**{k:v[0] for k,v in metrics(H[ok],S[ok]).items()}})
for m in MODELS:
    for left,right in [(m+'/B',m+'/C'),(m+'/B','OLS (OOF)'),(m+'/C','OLS (OOF)'),(m+'/B','Year mean (OOF)'),(m+'/C','Year mean (OOF)')]:
        s1=P[left].copy();s2=P[right].copy();ok=np.isfinite(s1)&np.isfinite(s2);s1[~ok]=np.nan;s2[~ok]=np.nan
        v1=metrics(H,s1);v2=metrics(H,s2);acc={k:[] for k in v1}
        for w in weights(attrs.index):
            a,b=metrics(H,s1,w),metrics(H,s2,w)
            for k in acc:acc[k].extend(a[k]-b[k])
        r={'left':left,'right':right}
        for k in acc:
            r[k]=float(v1[k][0]-v2[k][0])
            if k!='abs_mean_bias':r[k+'_lo'],r[k+'_hi']=ci(acc[k])
        comparison.append(r)
    s1=P[m+'/B'];s2=P[m+'/C']
    for t,key in enumerate(THERMS):
        ok=np.isfinite(H[:,t])&np.isfinite(s1[:,t])&np.isfinite(s2[:,t]);h=H[ok,t]
        for c,S in [('B',s1),('C',s2)]:
            s=S[ok,t];r=corr(h,s);bias2=(s.mean()-h.mean())**2;scale=(s.std()-h.std())**2;association=2*s.std()*h.std()*(1-r);mse=np.mean((s-h)**2)
            assert np.isclose(mse,bias2+scale+association,atol=1e-7)
            decomp.append({'model':m,'condition':c,'item':key,'n':len(h),'mse':mse,'bias_squared':bias2,'scale_mismatch':scale,'imperfect_association':association,'mae':abs(s-h).mean(),'r':r})
save('fidelity',rows);save('fidelity_by_item',items);save('fidelity_by_year',yearrows);save('paired_comparisons',comparison);save('mse_decomposition',decomp)
# Spearman uses average ranks, no scipy dependency. Full draw, item-specific pairs.
print('agreement and ICC',flush=True)
ag=[];agitems=[];ic=[];ics=[]
for m in MODELS:
    for c1,c2 in [('B','C'),('A','B'),('B','D'),('C','D')]:
        x=wide(L.resp,m,c1,keys=attrs.index,block='full').to_numpy(float);y=wide(L.resp,m,c2,keys=attrs.index,block='full').to_numpy(float)
        d=y-x;vals=d[np.isfinite(d)];rs=[];ps=[]
        for t,key in enumerate(THERMS):
            ok=np.isfinite(d[:,t]);xx=x[ok,t];yy=y[ok,t];dd=yy-xx
            sp=corr(pd.Series(xx).rank().to_numpy(),pd.Series(yy).rank().to_numpy());pr=corr(xx,yy);rs.append(sp);ps.append(pr)
            agitems.append({'model':m,'pair':c2+'-'+c1,'item':key,'n':len(dd),'pearson':pr,'spearman':sp,'median_abs':np.median(abs(dd)),'signed_mean':dd.mean(),**{f'signed_q{q}':np.percentile(dd,q) for q in [5,25,50,75,95]}})
        within=[]
        for xx,yy in zip(x,y):
            ok=np.isfinite(xx)&np.isfinite(yy)
            if ok.sum()>=3:within.append(corr(xx[ok],yy[ok]))
        ag.append({'model':m,'pair':c2+'-'+c1,'n_cells':len(vals),'identical':np.mean(vals==0),'within5':np.mean(abs(vals)<=5),'within10':np.mean(abs(vals)<=10),'mad':abs(vals).mean(),'median_abs':np.median(abs(vals)),'signed_mean':vals.mean(),'pearson_mean':np.nanmean(ps),'spearman_mean':np.nanmean(rs),'within_respondent_r':np.nanmean(within),**{f'signed_q{q}':np.percentile(vals,q) for q in [5,25,50,75,95]}})
    for c in 'ABCD':
        A=draws_array(L.resp,m,c,sub250_keys());vals=[]
        for t,key in enumerate(THERMS):
            Y=A[:,t,:];Y=Y[np.isfinite(Y).all(1)];n=len(Y);mu=Y.mean(1);msb=5*np.var(mu,ddof=1);msw=((Y-mu[:,None])**2).sum()/(n*4);icc=(msb-msw)/(msb+4*msw)
            vals.append(icc);ic.append({'model':m,'condition':c,'item':key,'n_respondents':n,'icc1':icc,'within_variance':msw,'between_variance_untruncated':(msb-msw)/5})
        ics.append({'model':m,'condition':c,'icc1_mean':np.mean(vals),'icc1_min':min(vals),'icc1_max':max(vals)})
save('agreement',ag);save('agreement_by_item',agitems);save('icc_by_item',ic);save('icc_summary',ics)
# Orthogonal method-of-moments decomposition of the latent C-B profile.
# Each call contains all items; preserve estimated within-call cross-item covariance.
def covariance(A):
    e=A-A.mean(2,keepdims=True)
    return np.einsum('ijk,ilk->ijl',e,e)/(A.shape[2]-1)
def components(D,V):
    n,j=D.shape;g=D.mean();rm=D.mean(1);im=D.mean(0);res=D-rm[:,None]-im[None,:]+g
    # Noise in grand, row, column, residual projections (normalization 1/nj).
    grand=V.sum()/(n*n*j*j);row0=V.sum((1,2)).mean()/(j*j);total=np.trace(V,axis1=1,axis2=2).mean()/j;col0=total/n
    obs={'grand_squared':g*g,'respondent':np.mean((rm-g)**2),'item':np.mean((im-g)**2),'respondent_item':np.mean(res**2)}
    noise={'grand_squared':grand,'respondent':row0-grand,'item':col0-grand,'respondent_item':total-row0-col0+grand}
    out={k:obs[k]-noise[k] for k in obs};out['total_squared']=np.mean(D*D)-total;out['signed_mean']=g;out['measurement_variance']=total
    assert np.isclose(sum(out[k] for k in obs),out['total_squared'])
    denom=sum(out[k] for k in ['respondent','item','respondent_item']);out['respondent_share']=out['respondent']/denom if denom>0 else np.nan
    return out
def response_components(x,y):
    vx=covariance(x);vy=covariance(y)
    mid=(x.mean(2)+y.mean(2))/2
    cm=components(mid,(vx+vy)/20)
    cd=components(y.mean(2)-x.mean(2),(vx+vy)/5)
    out={k:cm[k] for k in ['respondent','item','respondent_item']}
    for k in ['grand_squared','respondent','item','respondent_item']:
        out['condition' if k=='grand_squared' else 'condition_'+k]=cd[k]/4
    out['generation']=np.trace((vx+vy)/2,axis1=1,axis2=2).mean()/x.shape[1]
    total=sum(out.values())
    return out,total

print('format and order decomposition',flush=True)
vr=[];order=[];oc=[];response_rows=[];averaged=[];available=[]
for m in MODELS:
    x=draws_array(L.resp,m,'B',sub250_keys());y=draws_array(L.resp,m,'C',sub250_keys());ok=np.isfinite(x).all((1,2))&np.isfinite(y).all((1,2));x=x[ok];y=y[ok]
    D=y.mean(2)-x.mean(2);V=covariance(x)/5+covariance(y)/5;pt=components(D,V)
    acc={k:[] for k in pt};rc,total=response_components(x,y);ra={k:[] for k in rc};shares={k:[] for k in rc}
    for ids in cluster_indices(np.array(sub250_keys())[ok]):
        v=components(D[ids],V[ids]);rv,rt=response_components(x[ids],y[ids])
        for k in rc:ra[k].append(rv[k]);shares[k].append(rv[k]/rt)
        for k in acc:acc[k].append(v[k])
    for k in pt:
        lo,hi=ci(acc[k]);vr.append({'model':m,'component':k,'n_respondents':len(D),'value':pt[k],'lo':lo,'hi':hi})
    for k in rc:
        lo,hi=ci(ra[k]);sl,sh=ci(shares[k]);response_rows.append({'model':m,'component':k,'value':rc[k],'lo':lo,'hi':hi,'total_single_call_variance':total,'share':rc[k]/total,'share_lo':sl,'share_hi':sh})
    for c in 'BC':
        ar=draws_array(L.resp,m,c,sub250_keys());valid=np.isfinite(ar).all(2);mean=np.nanmean(ar,axis=2);mean[~valid]=np.nan
        ix=attrs.index.get_indexer(sub250_keys());ols=P['OLS (OOF)'][ix].copy();ols[~valid]=np.nan
        averaged.append({'model':m,'condition':c,**{'llm_'+k:v[0] for k,v in metrics(H[ix],mean).items()},**{'ols_'+k:v[0] for k,v in metrics(H[ix],ols).items()}})
    for c in 'ABCD':
        x=draws_array(L.resp,m,c,sub250_keys());y=wide(L.resp,m,c,0,'R',sub250_keys(),block='rev').to_numpy(float)
        with np.errstate(invalid='ignore'):
            av=y-np.nanmean(x,axis=2);row=np.nanmean(av,axis=1)
        available.append({'model':m,'condition':c,'n_records':int(np.isfinite(row).sum()),'signed_mean':np.nanmean(row)})
        ok=np.isfinite(x).all((1,2))&np.isfinite(y).all(1);x=x[ok];y=y[ok];D=y-x.mean(2);cov=covariance(x);v=D.mean(1)
        boot=np.array([v[ids].mean() for ids in cluster_indices(np.array(sub250_keys())[ok])])
        lo,hi=ci(boot);order.append({'model':m,'condition':c,'n_respondents':len(v),'signed_mean':v.mean(),'lo':lo,'hi':hi,'mad_vs_five_draw_mean':abs(D).mean()})
        for factor in [.5,1.,2.]:
            pt=components(D,cov*(factor+1/5))
            for k,val in pt.items():oc.append({'model':m,'condition':c,'reverse_noise_multiplier':factor,'component':k,'value':val})
save('response_variance_decomposition',response_rows);save('five_draw_fidelity',averaged);save('order_available_cases',available)
save('format_decomposition',vr);save('order_effects',order);save('order_decomposition_sensitivity',oc)
print('adjusted human mirror',flush=True)
# OLS conditional mean differences after simultaneously mirroring party AND ideology;
# other nine attributes unchanged. Descriptive, not causal. Same paired cells in both sources.
elig=attrs.pid.isin(['Democrat','Republican']).to_numpy();ea=attrs[elig].copy();eh=H[elig];ne=len(ea)
lev={c:sorted(ea[c].unique()) for c in ea.columns if c!='age'}
em=ea.copy();em['pid']=em.pid.map({'Democrat':'Republican','Republican':'Democrat'})
mp={'extremely liberal':'extremely conservative','liberal':'conservative','slightly liberal':'slightly conservative','moderate':'moderate','slightly conservative':'slightly liberal','conservative':'liberal','extremely conservative':'extremely liberal'}
em['ideo']=em.ideo.map(mp);sgn=np.where(ea.pid.to_numpy()=='Republican',1.,-1.)
x,cn=design(ea,lev);xm,_=design(em,lev);dx=xm-x
np.savez_compressed(OUT/'mirror_design.npz',X=x,X_mirror=xm,keys=ea.index.to_numpy(str),columns=np.array(cn))
fit=[];deltas=np.zeros_like(eh)
for t in range(11):
    ok=np.isfinite(eh[:,t]);beta=np.linalg.lstsq(x[ok],eh[ok,t],rcond=None)[0];deltas[:,t]=dx@beta
    fit.append({'item':THERMS[t],'n':int(ok.sum()),'rank':int(np.linalg.matrix_rank(x[ok])),'columns':x.shape[1]})
save('human_mirror_fit',fit)
mirrorS={m:wide(L.resp,m,'D',keys=ea.index,block='full').to_numpy(float)-wide(L.resp,m,'B',keys=ea.index,block='full').to_numpy(float) for m in MODELS}
def mirror_metrics(S,Q,W):
    # Equal weight to observed party groups, with item-specific paired observed cells.
    mask=np.isfinite(S)&np.isfinite(eh);mm=[];hh=[]
    for group in [-1.,1.]:
        v=mask&(sgn[:,None]==group);den=W@v
        mm.append(W@np.where(v,S*sgn[:,None],0)/den)
        hh.append(W@np.where(v,Q*sgn[:,None],0)/den)
    sm=(mm[0]+mm[1])/2;hm=(hh[0]+hh[1])/2
    # Absolute individual conditional shifts: not |mean signed shift|.
    spill_s=[];spill_h=[]
    for group in [-1.,1.]:
        v=mask&(sgn[:,None]==group);den=W@v
        spill_s.append((W@np.where(v,abs(S),0)/den)[:,2:].mean(1));spill_h.append((W@np.where(v,abs(Q),0)/den)[:,2:].mean(1))
    return {'ratio_all':abs(sm).mean(1)/abs(hm).mean(1),'ratio_party':abs(sm[:,:2]).mean(1)/abs(hm[:,:2]).mean(1),'spill_model':(spill_s[0]+spill_s[1])/2,'spill_human':(spill_h[0]+spill_h[1])/2},sm,hm
pts={};acc={};miritems=[]
for m,S in mirrorS.items():
    z,sm,hm=mirror_metrics(S,deltas,np.ones((1,ne)));pts[m]={k:v[0] for k,v in z.items()};acc[m]={k:[] for k in z}
    for t,key in enumerate(THERMS):miritems.append({'model':m,'item':key,'model_contrast':sm[0,t],'adjusted_human_contrast':hm[0,t]})
rng=np.random.default_rng(SEED)
# Bootstrap refits OLS and recomputes standardization on each resampled cohort.
for b,ids in enumerate(cluster_indices(ea.index)):
    w=np.bincount(ids,minlength=ne).astype(float);Q=np.empty_like(eh)
    for t in range(11):
        ok=np.isfinite(eh[:,t])&(w>0);xx=x[ok]*np.sqrt(w[ok,None]);yy=eh[ok,t]*np.sqrt(w[ok]);beta=np.linalg.lstsq(xx,yy,rcond=None)[0];Q[:,t]=dx@beta
    for m,S in mirrorS.items():
        z,_,_=mirror_metrics(S,Q,w[None,:])
        for k,v in z.items():acc[m][k].append(v[0])
    if b%200==0:print('mirror bootstrap',b,flush=True)
mr=[]
for m in MODELS:
    r={'model':m}
    for k,v in pts[m].items():r[k]=v;r[k+'_lo'],r[k+'_hi']=ci(acc[m][k])
    mr.append(r)
save('adjusted_mirror',mr);save('adjusted_mirror_by_item',miritems)
# Interaction sensitivity, with same-nine-attributes adjustment.
xi,_=design(ea,lev,True);xmi,_=design(em,lev,True);qi=np.empty_like(eh)
for t in range(11):
    ok=np.isfinite(eh[:,t]);qi[:,t]=(xmi-xi)@np.linalg.lstsq(xi[ok],eh[ok,t],rcond=None)[0]
save('mirror_interaction_sensitivity',[{'model':m,**{k:v[0] for k,v in mirror_metrics(S,qi,np.ones((1,ne)))[0].items()}} for m,S in mirrorS.items()])
save('party_ideology_support',ea.groupby(['pid','ideo']).size().reset_index(name='n'))
manifest={'seed':SEED,'bootstrap_resamples':B,'n_calls':len(L.calls),'n_person_wave_records':N,'n_unique_people':6027,'n_subsample_unique_people':int(person_ids.loc[sub250_keys()].nunique()),'cv_folds':5,'cv_intervals':'Conditional on frozen out-of-fold predictions; no nested refit.', 'mirror_intervals':'Person-cluster bootstrap retaining all observed waves, refitting human OLS in each resample.', 'format':'Orthogonal moment decomposition; cross-item covariance retained; negative component estimates not truncated.', 'order':'Single reversed call; variance decomposition depends on reverse/original noise covariance assumption. Multipliers .5,1,2.', 'person_clusters_sha256':hashlib.sha256((ROOT/'data/person_clusters.csv').read_bytes()).hexdigest(),'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (RELEASE/'data').glob('*') if p.is_file()},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE',flush=True)
