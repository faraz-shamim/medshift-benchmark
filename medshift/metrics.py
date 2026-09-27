from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score

EPS=1e-7

def clip(p): return np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
def entropy(p):
    p=clip(p); return -(p*np.log(p)+(1-p)*np.log1p(-p))
def nll_values(y,p):
    p=clip(p); y=np.asarray(y); return -(y*np.log(p)+(1-y)*np.log1p(-p))

def _observed(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    mask=np.isfinite(y)&np.isfinite(p)
    y=y[mask]; p=p[mask]
    if np.any((p<0)|(p>1)): raise ValueError('Probabilities must be between 0 and 1.')
    if not set(np.unique(y)).issubset({0,1}): raise ValueError('Binary labels must be 0, 1, or NaN.')
    return y,p,mask

def fit_temperature(y,z,min_events=20):
    y=np.asarray(y,dtype=float); z=np.asarray(z,dtype=float); mask=np.isfinite(y)&np.isfinite(z)
    y=y[mask]; z=z[mask]
    if min(np.sum(y==1),np.sum(y==0))<min_events: return {'T':np.nan,'status':'insufficient_events'}
    def objective(logT):
        v=z/np.exp(logT); return np.mean(np.logaddexp(0,v)-y*v)
    result=minimize_scalar(objective,bounds=(-3.,3.),method='bounded',options={'xatol':1e-6})
    return {'T':float(np.exp(result.x)),'status':'ok' if result.success else 'optimization_failed',
            'boundary':bool(abs(result.x)>2.99),'n':len(y),'positive':int(y.sum())}

def fit_logistic(y,z,min_events=20):
    y=np.asarray(y,dtype=float); z=np.asarray(z,dtype=float); mask=np.isfinite(y)&np.isfinite(z)
    y=y[mask]; z=z[mask]
    if min(np.sum(y==1),np.sum(y==0))<min_events:
        return {'a':np.nan,'b':np.nan,'status':'insufficient_events'}
    # Positive slope preserves rankings; the intercept can adjust source prevalence miscalibration.
    def obj(v):
        a=np.exp(v[0]); q=a*z+v[1]
        val=np.mean(np.logaddexp(0,q)-y*q)
        err=expit(q)-y
        grad=np.array([np.mean(err*a*z),np.mean(err)])
        return val,grad
    r=minimize(obj,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-5,5),(-15,15)])
    return {'a':float(np.exp(r.x[0])),'b':float(r.x[1]),'status':'ok' if r.success else 'optimization_failed',
            'boundary':bool(abs(r.x[0])>4.99 or abs(r.x[1])>14.99),'n':len(y),'positive':int(y.sum())}

def calibration_curve(y,p,bins=15,adaptive=False):
    y,p,_=_observed(y,p)
    if not len(y): return pd.DataFrame(columns=['bin','n','positives','mean_probability','observed_rate'])
    if adaptive:
        chunks=np.array_split(np.argsort(p,kind='stable'),min(bins,len(p)))
    else:
        ix=np.minimum((p*bins).astype(int),bins-1)
        chunks=[np.where(ix==i)[0] for i in range(bins)]
    rows=[]
    for i,idx in enumerate(chunks):
        if len(idx): rows.append({'bin':i,'n':len(idx),'positives':int(y[idx].sum()),
                                  'mean_probability':float(p[idx].mean()),'observed_rate':float(y[idx].mean())})
    return pd.DataFrame(rows)

def calibration_diagnostics(y,p):
    y,p,_=_observed(y,p)
    if min(np.sum(y==1),np.sum(y==0))<20:
        return dict(calibration_intercept=np.nan,calibration_slope=np.nan,calibration_in_large=np.nan)
    z=logit(clip(p))
    def obj(v):
        q=v[0]+v[1]*z; err=expit(q)-y
        return np.mean(np.logaddexp(0,q)-y*q),np.array([err.mean(),np.mean(err*z)])
    result=minimize(obj,np.array([0.,1.]),jac=True,method='BFGS')
    offset=minimize_scalar(lambda b:np.mean(np.logaddexp(0,z+b)-y*(z+b)),bounds=(-20,20),method='bounded')
    # Large slopes/intercepts can indicate separation or a near-constant predictor, not good calibration.
    valid=result.success and np.max(np.abs(result.x))<100
    return dict(calibration_intercept=float(result.x[0]) if valid else np.nan,
                calibration_slope=float(result.x[1]) if valid else np.nan,
                calibration_in_large=float(offset.x) if offset.success else np.nan)

def source_threshold(y,p,target_sensitivity=.9):
    y,p,_=_observed(y,p); pos=np.sort(p[y==1])
    if not len(pos): return np.nan
    k=min(len(pos)-1,int(np.floor((1-target_sensitivity)*len(pos)+1e-9)))
    return float(pos[k])

def binary_metrics(y,p,threshold=.5,source_prevalence=None):
    y,p,_=_observed(y,p); n=len(y)
    if n==0: return {'n':0,'positive':0,'negative':0}
    pos=int(y.sum()); neg=n-pos; decision=p>=threshold
    tp=int(np.sum(decision & (y==1))); tn=int(np.sum(~decision & (y==0)))
    fp=neg-tn; fn=pos-tp
    curve=calibration_curve(y,p); adaptive=calibration_curve(y,p,adaptive=True)
    ece=lambda c:float(np.sum(c.n*np.abs(c.mean_probability-c.observed_rate))/n)
    brier=float(np.mean((p-y)**2))
    result={'n':n,'positive':pos,'negative':neg,'prevalence':pos/n,
        'auroc':float(roc_auc_score(y,p)) if pos and neg else np.nan,
        'auprc':float(average_precision_score(y,p)) if pos and neg else np.nan,
        'brier':brier,'nll':float(np.mean(nll_values(y,p))),
        'ece15':ece(curve),'adaptive_ece15':ece(adaptive),
        'sensitivity':tp/pos if pos else np.nan,'specificity':tn/neg if neg else np.nan,
        'ppv':tp/(tp+fp) if tp+fp else np.nan,'npv':tn/(tn+fn) if tn+fn else np.nan,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
        'threshold':threshold,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
        'low_event_warning':min(pos,neg)<100}
    if source_prevalence is not None:
        baseline=float(np.mean((y-source_prevalence)**2))
        result['source_constant_brier']=baseline
        result['brier_skill_vs_source_constant']=1-brier/baseline if baseline>0 else np.nan
    result.update(calibration_diagnostics(y,p))
    return result

def risk_coverage(y,p,u,threshold=.5,points=100):
    y0=np.asarray(y,dtype=float); p0=np.asarray(p,dtype=float); u0=np.asarray(u,dtype=float)
    mask=np.isfinite(y0)&np.isfinite(p0)&np.isfinite(u0)
    y=y0[mask]; p=p0[mask]; u=u0[mask]
    if not len(y): return pd.DataFrame(), {'aurc_error':np.nan,'aurc_brier':np.nan,'error_detection_auroc':np.nan}
    order=np.argsort(u,kind='stable'); y=y[order]; p=p[order]; u=u[order]
    error=((p>=threshold)!=y).astype(float); brier=(y-p)**2; tp=((p>=threshold)&(y==1)).astype(float)
    k=np.arange(1,len(y)+1); cpos=np.cumsum(y)
    ce=np.cumsum(error)/k; cb=np.cumsum(brier)/k; ctp=np.cumsum(tp)
    selected=np.unique(np.maximum(0,np.ceil(np.linspace(.01,1,points)*len(y)).astype(int)-1))
    rows=[]
    for j in selected:
        rows.append({'coverage':(j+1)/len(y),'error_risk':float(ce[j]),'brier_risk':float(cb[j]),
                     'retained_prevalence':float(cpos[j]/(j+1)),
                     'positive_coverage':float(cpos[j]/max(1,y.sum())),
                     'retained_sensitivity':float(ctp[j]/cpos[j]) if cpos[j]>0 else np.nan})
    summary={'aurc_error':float(ce.mean()),'aurc_brier':float(cb.mean()),
             'error_detection_auroc':float(roc_auc_score(error,u)) if len(np.unique(error))==2 else np.nan}
    for coverage in [.5,.8,.9]:
        j=max(0,math.ceil(coverage*len(y))-1)
        summary[f'positive_coverage_at_{int(coverage*100)}']=float(cpos[j]/max(1,y.sum()))
    return pd.DataFrame(rows),summary

def cluster_bootstrap_mean(values,groups,B=2000,seed=20260926):
    """Image-weighted mean, resampling whole patients; conditional on the fitted models.

    values can be seed-averaged paired losses. The same bootstrap draw therefore pairs
    both methods, all images of a patient, and the three pre-specified training seeds.
    """
    values=np.asarray(values,dtype=float); groups=np.asarray(groups).astype(str)
    good=np.isfinite(values); values=values[good]; groups=groups[good]
    unique,codes=np.unique(groups,return_inverse=True)
    if len(unique)<2: return {'estimate':np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_centered_bootstrap':np.nan,'clusters':len(unique)}
    sums=np.bincount(codes,weights=values); counts=np.bincount(codes)
    point=float(values.mean()); rng=np.random.default_rng(seed); draws=[]
    for start in range(0,B,64):
        idx=rng.integers(0,len(unique),size=(min(64,B-start),len(unique)))
        draws.extend((sums[idx].sum(axis=1)/counts[idx].sum(axis=1)).tolist())
    boot=np.asarray(draws); lo,hi=np.quantile(boot,[.025,.975])
    # Approximate two-sided, null-centered bootstrap test. Not a randomized clinical trial test.
    p=(1+np.sum(np.abs(boot-point)>=abs(point)))/(B+1)
    return {'estimate':point,'ci_low':float(lo),'ci_high':float(hi),'p_centered_bootstrap':float(p),
            'clusters':len(unique),'B':B}

def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float); result=np.full(len(p),np.nan)
    valid=np.where(np.isfinite(p))[0]; order=valid[np.argsort(p[valid])]
    adjusted=np.maximum.accumulate([(len(order)-rank)*p[i] for rank,i in enumerate(order)])
    result[order]=np.clip(adjusted,0,1)
    return result

def sensitivity_labels(df,label,policy='primary'):
    y=df[label].to_numpy(dtype=float).copy()
    if policy=='primary': return y
    if df.domain.iloc[0]=='chexpert':
        raw=df[label+'_raw'].to_numpy(dtype=float)
        if policy=='uncertain_negative': return np.nan_to_num(raw,nan=0).clip(min=0)
        if policy=='uncertain_positive': return np.where(raw==-1,1,np.nan_to_num(raw,nan=0))
        if policy=='explicit_only': return np.where(np.isin(raw,[0,1]),raw,np.nan)
    if df.domain.iloc[0]=='vinbig' and policy=='unanimous_only':
        votes=df[label+'_votes'].to_numpy()
        return np.where(votes==3,1,np.where(votes==0,0,np.nan))
    raise ValueError(f'Unsupported label policy {policy}')


def _rank_plan(y,p):
    order=np.argsort(p,kind='stable'); pp=p[order]; yy=y[order]
    starts=np.r_[0,np.where(np.diff(pp)!=0)[0]+1]
    return order,yy,starts

def _weighted_rank_metrics(plan,weights):
    order,y,starts=plan; w=weights[order]
    positive=np.add.reduceat(w*y,starts); negative=np.add.reduceat(w*(1-y),starts)
    P=positive.sum(); N=negative.sum()
    if P==0 or N==0: return np.nan,np.nan
    auc=np.sum(positive*(np.cumsum(negative)-.5*negative))/(P*N)
    pos=positive[::-1]; neg=negative[::-1]
    denom=np.cumsum(pos+neg); precision=np.divide(np.cumsum(pos),denom,out=np.zeros_like(denom),where=denom>0)
    ap=np.sum((pos/P)*precision)
    return float(auc),float(ap)

def bootstrap_discrimination(y,probabilities,groups,B=1000,seed=20260926,subtract_probabilities=None):
    """CIs for seed-mean AUROC/AUPRC (or their paired difference), with patient-cluster draws."""
    y=np.asarray(y,dtype=float); ps=np.atleast_2d(probabilities).astype(float)
    other=None if subtract_probabilities is None else np.atleast_2d(subtract_probabilities).astype(float)
    mask=np.isfinite(y)&np.all(np.isfinite(ps),axis=0)
    if other is not None: mask &= np.all(np.isfinite(other),axis=0)
    y=y[mask]; ps=ps[:,mask]; groups=np.asarray(groups).astype(str)[mask]
    if other is not None: other=other[:,mask]
    unique,codes=np.unique(groups,return_inverse=True); m=len(unique)
    if m<2 or len(np.unique(y))<2:
        return {'auroc_low':np.nan,'auroc_high':np.nan,'auprc_low':np.nan,'auprc_high':np.nan}
    plans=[_rank_plan(y,p) for p in ps]
    plans2=[] if other is None else [_rank_plan(y,p) for p in other]
    rng=np.random.default_rng(seed); values=[]
    for _ in range(B):
        counts=np.bincount(rng.integers(0,m,size=m),minlength=m)
        weights=counts[codes].astype(float)
        val=np.nanmean([_weighted_rank_metrics(plan,weights) for plan in plans],axis=0)
        if plans2: val-=np.nanmean([_weighted_rank_metrics(plan,weights) for plan in plans2],axis=0)
        values.append(val)
    a=np.asarray(values); lo,hi=np.nanquantile(a,[.025,.975],axis=0)
    return {'auroc_low':float(lo[0]),'auroc_high':float(hi[0]),'auprc_low':float(lo[1]),'auprc_high':float(hi[1]),'rank_bootstrap_B':B}
