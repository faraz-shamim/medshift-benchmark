from __future__ import annotations
import itertools,json,math,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from .common import *
from .metrics import *


def load_evaluation(root):
    root=Path(root)
    if not (root/'INFERENCE_COMPLETE.json').exists():
        raise RuntimeError('Incomplete inference. Do not publish partial results as the pre-specified benchmark.')
    complete=read_json(root/'INFERENCE_COMPLETE.json')
    if complete['protocol_sha256']!=protocol_hash(): raise RuntimeError('Protocol mismatch.')
    if complete.get('code_sha256') and complete['code_sha256']!=code_hash():
        raise RuntimeError('Analysis code differs from the frozen inference pipeline.')
    for key in ['prediction_hashes','metadata_hashes']:
        for relative,digest in complete.get(key,{}).items():
            if sha256_file(root/relative)!=digest: raise RuntimeError(f'Inference content checksum mismatch: {relative}')
    metadata={p.stem:pd.read_csv(p,dtype={'patient_id':str,'image_uid':str,'original_id':str,'dhash':str})
              for p in sorted((root/'metadata').glob('*.csv'))}
    expected={'nih_calibration','nih_test','chexpert_test','vinbig_test'}
    if set(metadata)!=expected: raise RuntimeError('Missing inference cohort metadata.')
    raw={}
    for job in jobs():
        for context,df in metadata.items():
            p=root/'predictions'/job['job_id']/f'{context}.npz'
            with np.load(p,allow_pickle=False) as z:
                if not np.array_equal(z['image_uid'],df.image_uid.astype(str).to_numpy()):
                    raise AssertionError('Prediction row order / image ID mismatch.')
                if z['logits'].shape!=(len(df),len(LABELS)): raise AssertionError('Invalid prediction shape.')
                if not np.isfinite(z['logits']).all(): raise AssertionError('Non-finite logits.')
                if job['fraction']==1 and ('mc_logits' not in z or z['mc_logits'].shape!=(len(df),PROTOCOL['mc_head_samples'],len(LABELS))):
                    raise AssertionError('Full-data run is missing 20-sample head MC logits.')
                raw[(job['model'],job['fraction'],str(job['seed']),context)]={k:z[k].copy() for k in z.files}
    return metadata,raw


def build_variants(metadata,raw):
    """Fit all calibrators ONLY on NIH calibration; external labels are never inputs."""
    variants={}; calibrators=[]; cal=metadata['nih_calibration']
    cal_y=cal[LABELS].to_numpy(dtype=float)
    source_prev=np.nanmean(cal_y,axis=0)
    def add(model,fraction,seed,method,cal_fraction,probs,extras=None,fit_subset=None):
        cp=probs['nih_calibration']; fit_subset=np.ones(len(cal),dtype=bool) if fit_subset is None else fit_subset
        thresholds=[source_threshold(cal_y[fit_subset,j],cp[fit_subset,j],PROTOCOL['source_threshold_sensitivity']) for j in range(2)]
        for context,p in probs.items():
            key=(model,fraction,str(seed),method,cal_fraction,context)
            variants[key]={'p':p,'thresholds':thresholds,'source_prevalence':source_prev,
                           'extras':{} if extras is None else extras.get(context,{})}
    for model in MODELS:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                seed=str(seed)
                z={c:raw[(model,fraction,seed,c)]['logits'].astype(float) for c in metadata}
                p={c:expit(v) for c,v in z.items()}
                add(model,fraction,seed,'raw',1.,p)
                budgets=PROTOCOL['calibration_fractions'] if fraction==1 else [1.]
                for budget in budgets:
                    sub=nested_patient_subset(cal,budget,int(seed)+9000)
                    mask=cal.image_uid.isin(sub.image_uid).to_numpy()
                    tp={c:np.full_like(v,np.nan) for c,v in z.items()}
                    lp={c:np.full_like(v,np.nan) for c,v in z.items()}
                    for j,label in enumerate(LABELS):
                        ts=fit_temperature(cal_y[mask,j],z['nih_calibration'][mask,j])
                        lr=fit_logistic(cal_y[mask,j],z['nih_calibration'][mask,j])
                        for method,params in [('temperature',ts),('logistic',lr)]:
                            calibrators.append({'model':model,'fraction':fraction,'seed':seed,'cal_fraction':budget,
                                                'method':method,'label':label,'cal_images':int(mask.sum()),
                                                'cal_patients':sub.patient_id.nunique(),**params})
                        if budget==1 and (ts['status']!='ok' or lr['status']!='ok'):
                            raise RuntimeError(f'Failed full-calibration fit {model} {fraction} {seed} {label}. Inspect rather than silently substitute identity.')
                        for c in metadata:
                            if ts['status']=='ok': tp[c][:,j]=expit(z[c][:,j]/ts['T'])
                            if lr['status']=='ok': lp[c][:,j]=expit(lr['a']*z[c][:,j]+lr['b'])
                    add(model,fraction,seed,'temperature',budget,tp,fit_subset=mask)
                    add(model,fraction,seed,'logistic',budget,lp,fit_subset=mask)
                if fraction==1:
                    mc={c:expit(raw[(model,fraction,seed,c)]['mc_logits'].astype(float)) for c in metadata}
                    mp={c:v.mean(axis=1) for c,v in mc.items()}
                    extras={c:{'variance':v.var(axis=1),'mutual_information':np.maximum(0,entropy(v.mean(axis=1))-entropy(v).mean(axis=1))} for c,v in mc.items()}
                    add(model,fraction,seed,'head_mc',1.,mp,extras)
                    mt={c:np.full_like(v,np.nan) for c,v in mp.items()}
                    for j,label in enumerate(LABELS):
                        fit=fit_temperature(cal_y[:,j],logit(clip(mp['nih_calibration'][:,j])))
                        if fit['status']!='ok': raise RuntimeError('MC pooled-probability calibration failed.')
                        calibrators.append({'model':model,'fraction':1.,'seed':seed,'cal_fraction':1.,
                                            'method':'head_mc_temperature','label':label,**fit})
                        for c in metadata: mt[c][:,j]=expit(logit(clip(mp[c][:,j]))/fit['T'])
                    add(model,fraction,seed,'head_mc_temperature',1.,mt)
        # Ensembles only at 100% training: low-fraction ensembles would use the union of several subsets.
        members={c:np.stack([expit(raw[(model,1.,str(s),c)]['logits']) for s in SEEDS]) for c in metadata}
        ep={c:v.mean(axis=0) for c,v in members.items()}
        extras={c:{'variance':v.var(axis=0),'mutual_information':np.maximum(0,entropy(v.mean(axis=0))-entropy(v).mean(axis=0))} for c,v in members.items()}
        add(model,1.,'ensemble','ensemble',1.,ep,extras)
        et={c:np.full_like(v,np.nan) for c,v in ep.items()}
        for j,label in enumerate(LABELS):
            fit=fit_temperature(cal_y[:,j],logit(clip(ep['nih_calibration'][:,j])))
            if fit['status']!='ok': raise RuntimeError('Ensemble calibration failed.')
            calibrators.append({'model':model,'fraction':1.,'seed':'ensemble','cal_fraction':1.,
                                'method':'ensemble_temperature','label':label,**fit})
            for c in metadata: et[c][:,j]=expit(logit(clip(ep[c][:,j]))/fit['T'])
        add(model,1.,'ensemble','ensemble_temperature',1.,et)
    return variants,pd.DataFrame(calibrators)


def evaluate_variants(metadata,variants):
    rows=[]; uncertainty=[]; curves=[]; sensitivity=[]; subgroups=[]; prevalence=[]; failures=[]
    for key,value in variants.items():
        model,fraction,seed,method,budget,context=key
        if context=='nih_calibration': continue  # no resubstitution calibration-set performance claims
        df=metadata[context]; domain=df.domain.iloc[0]
        base={'model':model,'fraction':fraction,'seed':seed,'method':method,'cal_fraction':budget,'domain':domain}
        for j,label in enumerate(LABELS):
            y=df[label].to_numpy(dtype=float); p=value['p'][:,j]; threshold=value['thresholds'][j]
            row={**base,'label':label,**binary_metrics(y,p,threshold,value['source_prevalence'][j])}
            rows.append(row)
            if fraction!=1 or budget!=1: continue
            scores={'entropy':entropy(p)}
            scores.update({name:v[:,j] for name,v in value['extras'].items()})
            for score,u in scores.items():
                curve,summary=risk_coverage(y,p,u,threshold)
                uncertainty.append({**base,'label':label,'uncertainty_score':score,**summary})
                if not curve.empty:
                    for col,val in {**base,'label':label,'uncertainty_score':score}.items(): curve[col]=val
                    curves.append(curve)
            if method in ['raw','temperature','ensemble_temperature']:
                policies={'nih':[], 'chexpert':['uncertain_negative','uncertain_positive','explicit_only'],
                          'vinbig':['unanimous_only']}[domain]
                for policy in policies:
                    sy=sensitivity_labels(df,label,policy)
                    sensitivity.append({**base,'label':label,'label_policy':policy,
                                        **binary_metrics(sy,p,threshold,value['source_prevalence'][j])})
                groups={'female':df.sex.eq('Female').to_numpy(),'male':df.sex.eq('Male').to_numpy(),
                        'PA':df.view.eq('PA').to_numpy(),'AP':df.view.eq('AP').to_numpy(),
                        'age18_44':df.age.between(18,44).to_numpy(),'age45_64':df.age.between(45,64).to_numpy(),
                        'age65_100':df.age.between(65,100).to_numpy()}
                for group,mask in groups.items():
                    if mask.sum()<100: continue
                    subgroups.append({**base,'label':label,'subgroup':group,
                                      **binary_metrics(y[mask],p[mask],threshold,value['source_prevalence'][j])})
                # Outcome-based standardization is a descriptive evaluation, NOT target-data model recalibration.
                yy,pp,_=_observed_local(y,p)
                if np.any(yy==0) and np.any(yy==1):
                    pi=value['source_prevalence'][j]
                    prevalence.append({**base,'label':label,'standardized_prevalence':pi,
                        'prevalence_standardized_brier':float(pi*np.mean((pp[yy==1]-1)**2)+(1-pi)*np.mean(pp[yy==0]**2)),
                        'prevalence_standardized_nll':float(pi*np.mean(-np.log(clip(pp[yy==1])))+(1-pi)*np.mean(-np.log1p(-clip(pp[yy==0]))))})
            if method=='ensemble_temperature':
                for status,mask,reverse in [('reference_negative_high_probability',y==0,True),('reference_positive_low_probability',y==1,False)]:
                    idx=np.flatnonzero(mask); idx=idx[np.argsort(p[idx])]
                    if reverse: idx=idx[::-1]
                    for i in idx[:20]: failures.append({**base,'label':label,'discordance':status,
                        'image_uid':df.image_uid.iloc[i],'reference_label':y[i],'predicted_probability':p[i],
                        'note':'Discordance with dataset reference, not adjudicated clinical error.'})
    return {'metrics':pd.DataFrame(rows),'uncertainty':pd.DataFrame(uncertainty),
            'risk_curves':pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(),
            'label_sensitivity':pd.DataFrame(sensitivity),'subgroups':pd.DataFrame(subgroups),
            'prevalence_standardized':pd.DataFrame(prevalence),'discordant_cases':pd.DataFrame(failures)}


def _observed_local(y,p):
    mask=np.isfinite(y)&np.isfinite(p); return y[mask],p[mask],mask


def primary_comparisons(metadata,variants,B=2000):
    rows=[]
    for model,domain,j in itertools.product(MODELS,['chexpert','vinbig'],range(2)):
        context=domain+'_test'; df=metadata[context]; y=df[LABELS[j]].to_numpy(dtype=float)
        raw=np.stack([variants[(model,1.,str(s),'raw',1.,context)]['p'][:,j] for s in SEEDS])
        ts=np.stack([variants[(model,1.,str(s),'temperature',1.,context)]['p'][:,j] for s in SEEDS])
        delta=np.mean((ts-y[None,:])**2-(raw-y[None,:])**2,axis=0)
        stats=cluster_bootstrap_mean(delta,df.patient_id,B=B)
        rows.append({'model':model,'domain':domain,'label':LABELS[j],
                     'contrast':'temperature_minus_raw_Brier','resampling_unit':'image' if domain=='vinbig' else 'patient',
                     'seed_aggregation':'mean paired per-image loss over 3 fixed trained models',**stats})
    d=pd.DataFrame(rows)
    if len(d)!=12 or not d.p_centered_bootstrap.notna().all(): raise RuntimeError('The pre-specified 12-test primary family is incomplete.')
    d['p_holm_12']=holm_adjust(d.p_centered_bootstrap)
    return d


def main_summary(metrics,metadata,variants,out,B=2000,rank_B=1000):
    full=metrics[(metrics.fraction==1)&(metrics.cal_fraction==1)].copy()
    keys=['model','method','domain','label']
    numeric=['auroc','auprc','brier','nll','ece15','adaptive_ece15','calibration_intercept','calibration_slope',
             'sensitivity','specificity','ppv','npv','brier_skill_vs_source_constant']
    summary=full.groupby(keys,dropna=False)[numeric].agg(['mean','std']).reset_index()
    summary.columns=['_'.join([str(x) for x in c if x]) if isinstance(c,tuple) else c for c in summary.columns]
    cache=Path(out)/'bootstrap_cache'; cache.mkdir(parents=True,exist_ok=True)
    ci=[]; rank_cache={}
    for row in summary.to_dict('records'):
        model,method,domain,label=[row[k] for k in keys]; context=domain+'_test'; j=LABELS.index(label)
        seeds=['ensemble'] if method.startswith('ensemble') else list(map(str,SEEDS))
        ps=np.stack([variants[(model,1.,s,method,1.,context)]['p'][:,j] for s in seeds])
        df=metadata[context]; y=df[label].to_numpy(dtype=float)
        cache_file=cache/f'{model}__{method}__{domain}__{label}__b{B}_r{rank_B}.json'
        if cache_file.exists(): record=read_json(cache_file)
        else:
            boot=cluster_bootstrap_mean(np.mean((ps-y[None,:])**2,axis=0),df.patient_id,B=B)
            # Cache only exactly equal rank/tie structures; finite-precision saturation can break
            # the mathematical rank-invariance of monotone temperature/logistic transforms.
            rank_signature=[]
            for pp in ps:
                order=np.argsort(pp,kind='stable')
                starts=np.r_[0,np.where(np.diff(pp[order])!=0)[0]+1]
                rank_signature.append(stable_hash(order.tobytes().hex()+starts.tobytes().hex()))
            rank_key=(model,domain,label,tuple(rank_signature))
            if rank_key not in rank_cache:
                rank_cache[rank_key]=bootstrap_discrimination(y,ps,df.patient_id,B=rank_B)
            record={'brier_ci_low':boot['ci_low'],'brier_ci_high':boot['ci_high'],
                    'resampling_unit':'image' if domain=='vinbig' else 'patient',**rank_cache[rank_key]}
            write_json(cache_file,record)
        ci.append({**{k:row[k] for k in keys},**record})
    return summary.merge(pd.DataFrame(ci),on=keys,validate='one_to_one')


def data_efficiency_comparisons(metadata,variants,B=2000,rank_B=1000):
    rows=[]
    for model,domain,j in itertools.product(MODELS,['chexpert','vinbig'],range(2)):
        context=domain+'_test'; df=metadata[context]; y=df[LABELS[j]].to_numpy(dtype=float)
        ps={f:np.stack([variants[(model,f,str(s),'temperature',1.,context)]['p'][:,j] for s in SEEDS]) for f in [.1,1.]}
        difference=np.mean((ps[1.]-y[None,:])**2-(ps[.1]-y[None,:])**2,axis=0)
        boot=cluster_bootstrap_mean(difference,df.patient_id,B=B)
        rank=bootstrap_discrimination(y,ps[1.],df.patient_id,B=rank_B,subtract_probabilities=ps[.1])
        auc=np.mean([roc_auc_score(y[np.isfinite(y)],p[np.isfinite(y)]) for p in ps[1.]])-np.mean([roc_auc_score(y[np.isfinite(y)],p[np.isfinite(y)]) for p in ps[.1]])
        rows.append({'model':model,'domain':domain,'label':LABELS[j],
                     'contrast':'100_percent_minus_10_percent_training_after_source_temperature',
                     'brier_difference':boot['estimate'],'brier_ci_low':boot['ci_low'],'brier_ci_high':boot['ci_high'],
                     'auroc_difference':auc,**rank,
                     'interpretation':'Exploratory, pointwise CIs; no non-inferiority or plateau claim.'})
    return pd.DataFrame(rows)


def find_evaluation(search_root='/kaggle/input'):
    paths=list(Path(search_root).rglob('INFERENCE_COMPLETE.json'))
    if not paths: raise FileNotFoundError('Attach the completed 03_Inference notebook output.')
    hashes={sha256_file(p) for p in paths}
    if len(hashes)>1: raise RuntimeError('Multiple different completed inference datasets attached.')
    return paths[0].parent


def analyze_all(evaluation_root,output='/kaggle/working/manuscript_outputs',B=2000,rank_B=1000):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    fingerprint=sha256_file(Path(evaluation_root)/'INFERENCE_COMPLETE.json')
    request={'inference_fingerprint':fingerprint,'protocol_sha256':protocol_hash(),'B':B,'rank_B':rank_B}
    if (out/'analysis_request.json').exists() and read_json(out/'analysis_request.json')!=request:
        raise RuntimeError('Cached analysis belongs to different inputs/settings. Use a new output directory.')
    write_json(out/'analysis_request.json',request)
    metadata,raw=load_evaluation(evaluation_root)
    variants,calibrators=build_variants(metadata,raw)
    result=evaluate_variants(metadata,variants)
    tables=out/'tables'; tables.mkdir(exist_ok=True)
    write_table(calibrators,tables,'S1_calibrators_and_event_counts')
    for name,df in result.items():
        # Long curves are CSV only; shorter summary tables also receive LaTeX/Markdown exports.
        df.to_csv(tables/f'all_{name}.csv',index=False)
    lock=read_json(Path(evaluation_root)/'study_lock.json')
    if 'cohort' in lock: write_table(pd.DataFrame(lock['cohort']),tables,'T1_cohort')
    primary=primary_comparisons(metadata,variants,B=B)
    write_table(primary,tables,'T3_primary_paired_comparisons')
    full=main_summary(result['metrics'],metadata,variants,out,B=B,rank_B=rank_B)
    write_table(full,tables,'T2_full_data_performance')
    eff=data_efficiency_comparisons(metadata,variants,B=B,rank_B=rank_B)
    write_table(eff,tables,'T4_data_efficiency_contrasts')
    for name,stem in [('uncertainty','T5_uncertainty'),('label_sensitivity','S2_label_sensitivity'),
                      ('subgroups','S3_subgroups'),('prevalence_standardized','S4_prevalence_standardized'),
                      ('discordant_cases','S5_reference_discordant_cases')]:
        write_table(result[name],tables,stem)
    runtime=[]
    for p in sorted((Path(evaluation_root)/'training_records').glob('*/run.json')):
        cfg=read_json(p); done=read_json(p.parent/'done.json')
        runtime.append({k:cfg[k] for k in ['model','fraction','seed','n_train_images','n_train_patients','n_tune_images','n_total_development_images','parameters']}|
                       {'epochs_run':done['epochs_run'],'best_epoch':done['best_epoch'],
                        'training_wall_hours':done['total_training_wall_seconds']/3600})
    runtime=pd.DataFrame(runtime); write_table(runtime,tables,'S6_training_ledger_and_label_budget')
    budget=result['metrics'][(result['metrics'].fraction==1)&(result['metrics'].method.isin(['temperature','logistic']))]
    write_table(budget,tables,'S7_calibration_label_budget')
    from .plots import make_figures
    make_figures(metadata,variants,result,primary,runtime,out/'figures')
    write_manuscript_pack(out,primary,runtime,lock,B,rank_B)
    environment(out/'analysis_environment.json')
    write_json(out/'ANALYSIS_COMPLETE.json',{'protocol_sha256':protocol_hash(),'bootstrap_B':B,'rank_bootstrap_B':rank_B,
                                          'primary_tests':len(primary),'training_jobs':len(runtime),
                                          'warning':'Human literature, ethical, image-quality and scientific review still required.'})
    print('Analysis complete. Tables, figures, Methods draft, results facts, and review checklist:',out)
    return out


def write_manuscript_pack(out,primary,runtime,lock,B,rank_B):
    out=Path(out); m=out/'manuscript'; m.mkdir(exist_ok=True)
    methods=f'''# Generated Methods draft - verify before submission

## Design and prespecification
This retrospective, secondary-data benchmark assessed two radiographic reference labels
(cardiomegaly and pleural effusion) in three datasets. The protocol hash was
`{protocol_hash()}`. This benchmark is not a prospective deployment or clinical safety study.
The original data-use agreements and local ethics determination must be documented by the authors.

## Data and partitions
NIH ChestX-ray14 was the development domain. Eligible images with recorded ages 18-100 years
and AP/PA projections were divided by patient into 60% training, 10% tuning, 10% source
calibration, and 20% internal testing. Each patient was assigned to only one partition.
All eligible images of included source patients were retained after the pre-specified exact-pixel audit.
External evaluation used a label-blind, hash-selected cohort of up to 20,000 adult CheXpert
patients, one frontal image per patient, and the 15,000 publicly labeled VinBigData training
images, minus documented quality/duplicate exclusions. The competition's 3,000 unlabeled
test images were not evaluated. Exact analyzed counts appear in T1, not assumed from dataset cards.
VinBigData's public mirror does not expose patient identifiers: its uncertainty intervals resample
images and cannot account for undisclosed repeated patients.

NIH absent target labels were coded negative. In CheXpert, unmentioned labels were coded
negative and uncertain (-1) labels were masked per finding in the primary analysis. Explicit-only,
uncertain-negative, and uncertain-positive analyses were prespecified sensitivities. VinBigData
image-level labels were derived from >=2/3 distinct radiologist votes; unanimous-only analyses
were reported separately. Reference-standard and acquisition differences are both part of the
observed dataset shift and cannot be separated causally by this design.

## Models and training
The exact timm identifiers were `{MODELS['resnet18']}`, `{MODELS['densenet121']}` and
`{MODELS['deit_tiny']}`. These were ImageNet-pretrained models, not medical foundation models.
A dropout (p=0.2) and two-output linear head was trained with the whole backbone. Images were
converted to 8-bit grayscale, resized to 224x224 without cropping, repeated over three channels,
and ImageNet normalized. Augmentation was limited to affine rotation +/-5 degrees, translations
up to 2%, and brightness/contrast changes up to 10%; flips, class rebalancing, label smoothing,
MixUp and CutMix were not used.

For each architecture, nested 10%, 25%, 50% and 100% patient subsets were evaluated using
seeds 17, 43 and 101, for 36 training runs. Tuning and source-calibration cohorts were fixed;
therefore training-data efficiency is conditional on this fixed additional labeled-data budget.
AdamW used backbone/head learning rates 1e-4/1e-3, weight decay 1e-4, effective batch 64,
2 warmup epochs, cosine decay, and up to 20 epochs. Early stopping used tuning NLL with
patience 4 after at least 5 epochs; the checkpoint with lowest tuning NLL was retained.
Independent model runs were scheduled on separate GPUs without sharing model parameters,
gradients, or batch-normalization statistics. Each model retained microbatch 32 and effective
batch 64. Actual epochs, patient/image counts and per-model elapsed epoch times appear in S6.
These elapsed times include loading and validation; overlapping jobs must not be summed and
reported as notebook allocation hours. Hardware and concurrent-session timing are recorded in
the execution logs.

## Calibration and uncertainty
Raw sigmoid probabilities, finding-specific temperature scaling, and positive-slope affine
logistic recalibration were evaluated. Calibrators used only the NIH calibration partition and
were frozen before test evaluation. At full training, 20 stochastic evaluations of the trained
dropout head were made using a frozen deterministic backbone, and three-seed probability
ensembles were formed. Post-hoc temperature scaling of pooled probabilities used the logit
of their mean probability, not the mean of member logits. Head-only MC dropout is an
approximate uncertainty baseline, not a full-network Bayesian posterior.

## Evaluation and statistical analysis
The 12 primary contrasts were external-test Brier score differences (temperature minus raw)
at full training, one per architecture, finding and external dataset. A negative difference favors
temperature scaling. Paired per-image loss differences were averaged over the three fixed
trained seeds, then patients were resampled with replacement (images for VinBigData), using
{B} replicates. Percentile 95% confidence intervals and approximate two-sided null-centered
bootstrap p values were reported; Holm correction controlled the 12-test family.
Intervals are conditional on these fitted models and source calibrators; they do not include
all training-population or calibration-population uncertainty. Seed SD was reported separately.

AUROC, AUPRC, NLL, Brier score, 15-bin fixed/adaptive label-wise calibration errors, diagnostic
calibration intercept/slope, and sensitivity/specificity at a source-calibration threshold targeting
90% sensitivity were reported. Rank-metric intervals used {rank_B} paired cluster-bootstrap
replicates and the seed-mean statistic. Brier is a proper probability score, not a pure calibration
measure; prevalence and a source-constant-probability baseline were reported alongside it.
Risk-coverage analysis used entropy, and additionally variance/mutual-information proxies for
MC/ensemble outputs, with retained positive-case coverage to expose majority-class deferral artifacts.
Training-fraction, calibration-fraction, subgroup, reference-label-policy and prevalence-standardized
analyses were exploratory. Pointwise secondary intervals were not interpreted as evidence of
non-inferiority or of an annotation-efficiency plateau.
'''
    (m/'METHODS.generated.md').write_text(methods)
    facts=['# Results facts generated from executed analysis','',
           'These are numerical facts, not an interpretation or a claim of clinical utility.',
           f'Completed training jobs: {len(runtime)}.',
           f'Sum of recorded training epoch wall time: {runtime.training_wall_hours.sum():.2f} hours.',
           '','## Primary paired differences (Brier: temperature minus raw)','']
    for r in primary.to_dict('records'):
        facts.append(f"{r['model']} / {r['domain']} / {r['label']}: {r['estimate']:.5f} "
                     f"(95% CI {r['ci_low']:.5f} to {r['ci_high']:.5f}); Holm-adjusted p={r['p_holm_12']:.4g}.")
    (m/'RESULTS_FACTS.generated.md').write_text('\n\n'.join(facts))
    template='''# Calibration, Uncertainty, and Data Efficiency of Medical Image Classifiers Under Cross-Dataset Distribution Shift: A Multi-Dataset Benchmark

## Authors and affiliations
[Insert only qualifying contributors and verified affiliations.]

## Abstract
[Write last: objective, retrospective design, actual cohorts, principal effect sizes with CIs,
limitations and a restrained conclusion. Do not report planned counts as analyzed counts.]

## Introduction
[Update the literature search. Explain what this specific low-compute, two-finding benchmark
adds beyond existing calibration-under-shift studies. Do not claim to be the first without evidence.]

## Methods
[Incorporate METHODS.generated.md after checking every statement against the code, logs, original
licenses, ethics determination, and frozen protocol. Cite the original dataset/method papers.]

## Results
[Use RESULTS_FACTS.generated.md and T1-T5/S1-S7. Report unsuccessful calibration,
non-convergence, boundary fits, missingness, seed variability and negative results.]

## Discussion
[Interpret findings rather than restating rankings. Separate probability accuracy from calibration;
explain prevalence, projection, reference-label and compressed-image effects. Address limited
model families, only two findings, one training institution, unknown VinBig patient clustering,
fixed calibration/tuning labels, conditional bootstrap inference, and no prospective validation.]

## Conclusion
[Write only after results are reviewed. Do not infer safe autonomous diagnosis or clinical deployment.]

## Ethics, data availability, code availability, funding, conflicts and AI assistance
[Authors must supply real statements. Public/de-identified data do not automatically establish
an exemption. Share code and aggregate results; do not publish restricted images/identifiers.]

## References
[Verify and import original sources from docs/REFERENCES.md.]
'''
    (m/'MANUSCRIPT_TEMPLATE.md').write_text(template)
    (m/'HUMAN_REVIEW_REQUIRED.md').write_text('''# Submission gates

- Document original dataset permissions and an institutional ethics determination; never invent approval.
- Review compressed-image rendering, label mappings, repeated patients, exact/dHash audit and cohort exclusions.
- Reconcile the 36-run ledger and 144 prediction files with the locked protocol and all protocol amendments.
- Examine calibration boundaries, non-convergence, event counts, seed variability and subgroup instability.
- Distinguish statistical uncertainty from model uncertainty, and dataset reference labels from diagnostic truth.
- Review reference-discordant examples with a qualified imaging collaborator where feasible.
- Complete the official CLAIM 2024 checklist with manuscript page references; this file is not that checklist.
- Verify every citation and write original scientific interpretation. Run a final search before submission.
- Publish only code/aggregate outputs permitted by the data agreements; review notebook outputs for images and identifiers.
''')
    (out/'REPORT.html').write_text('<!doctype html><html><meta charset="utf-8"><title>MedShift outputs</title>'
        '<h1>MedShift completed analysis</h1><p>Open tables/ for CSV, Markdown and LaTeX outputs; figures/ for PNG/PDF/SVG.</p>'
        '<p>Methods and Results facts are in manuscript/. Human scientific and ethical review remains required.</p>'
        +primary.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</html>')
