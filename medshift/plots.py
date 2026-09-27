from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve,precision_recall_curve
from .common import LABELS,MODELS,SEEDS
from .metrics import calibration_curve


def save(fig,out,stem):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if all((out/f'{stem}.{ext}').exists() for ext in ['png','pdf','svg']):
        plt.close(fig); return
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def make_figures(metadata,variants,result,primary,runtime,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    # F1: Counts are populated from actual prepared manifests and executed run metadata.
    fig,ax=plt.subplots(figsize=(12,6)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    n_train=int(runtime[runtime.fraction==1].n_train_images.iloc[0]); n_tune=int(runtime.n_tune_images.iloc[0])
    boxes=[(.2,4.5,2.5,1.,f'NIH development\nTraining: {n_train:,} images\nTuning: {n_tune:,}'),
           (3.2,4.5,2.5,1.,'3 architectures\n4 training fractions\n3 seeds = 36 fits'),
           (6.2,4.5,2.5,1.,f'Source calibration only\n{len(metadata["nih_calibration"]):,} images\nFreeze calibrators'),
           (9.2,4.5,2.5,1.,'Freeze checkpoints\nand protocol\nNo target tuning'),
           (.4,1.9,3.3,1.2,f'NIH internal test\n{len(metadata["nih_test"]):,} images\nPatient-disjoint from development'),
           (4.3,1.9,3.3,1.2,f'CheXpert external test\n{len(metadata["chexpert_test"]):,} images\nOne image per selected patient'),
           (8.2,1.9,3.3,1.2,f'VinBigData external test\n{len(metadata["vinbig_test"]):,} images\nPatient IDs unavailable'),
           (2.4,.1,7.2,1.,'Discrimination + calibration + uncertainty + label budget\nPaired cluster bootstrap; Holm correction for 12 primary contrasts')]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor='none',edgecolor='black'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center')
    for a,b in [((2.7,5),(3.2,5)),((5.7,5),(6.2,5)),((8.7,5),(9.2,5)),
                ((10.4,4.5),(2.0,3.15)),((10.4,4.5),(5.9,3.15)),((10.4,4.5),(9.8,3.15)),
                ((2.0,1.9),(4,.95)),((5.9,1.9),(6,.95)),((9.8,1.9),(8,.95))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','linewidth':1.0})
    save(fig,out,'F1_study_flow')
    metrics=result['metrics']
    # F2: Full calibration budget held fixed; error bars are training-seed SD, not 95% CIs.
    for label in LABELS:
        for domain in ['nih','chexpert','vinbig']:
            d=metrics[(metrics.label==label)&(metrics.domain==domain)&(metrics.method=='temperature')&(metrics.cal_fraction==1)]
            for metric in ['auroc','brier','nll','ece15']:
                fig,ax=plt.subplots(figsize=(6.8,4.8))
                for model in MODELS:
                    g=d[d.model==model].groupby('fraction')[metric].agg(['mean','std'])
                    ax.errorbar(100*g.index,g['mean'],yerr=g['std'].fillna(0),marker='o',capsize=3,label=model)
                ax.set(xlabel='Training patients (% of fixed training partition)',ylabel=metric.upper(),
                       title=f'{label.replace("_"," ").title()} - {domain}\nSource temperature scaling; mean +/- seed SD')
                ax.set_xticks([10,25,50,100]); ax.legend(); ax.grid(alpha=.2)
                save(fig,out,f'F2_learning_{domain}_{label}_{metric}')
    # F3: Reliability plots use a PRE-SPECIFIED seed 17; averaging probabilities would create an ensemble.
    for model in MODELS:
        for context,df in metadata.items():
            if context=='nih_calibration': continue
            domain=df.domain.iloc[0]
            for j,label in enumerate(LABELS):
                fig,ax=plt.subplots(figsize=(6,5.3)); ax.plot([0,1],[0,1],'--',label='Perfect calibration')
                for method,seed,text in [('raw','17','Raw, seed 17'),('temperature','17','Temperature, seed 17'),
                                         ('ensemble_temperature','ensemble','Calibrated 3-seed ensemble')]:
                    p=variants[(model,1.,seed,method,1.,context)]['p'][:,j]
                    c=calibration_curve(df[label],p,bins=15)
                    c.to_csv(out/f'F3_bins_{model}_{domain}_{label}_{method}.csv',index=False)
                    ax.plot(c.mean_probability,c.observed_rate,marker='o',label=text)
                ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed reference-positive fraction',
                       title=f'{model}: {label.replace("_"," ")} - {domain}')
                ax.legend(fontsize=8); ax.grid(alpha=.2)
                save(fig,out,f'F3_reliability_{model}_{domain}_{label}')
    # F4: The primary comparison, not a winner-selection plot.
    for label in LABELS:
        d=primary[primary.label==label].reset_index(drop=True)
        fig,ax=plt.subplots(figsize=(8,4.8)); y=np.arange(len(d))
        ax.errorbar(d.estimate,y,xerr=np.vstack([d.estimate-d.ci_low,d.ci_high-d.estimate]),fmt='o',capsize=4)
        ax.axvline(0,linestyle='--'); ax.set_yticks(y,labels=d.model+' / '+d.domain)
        ax.set(xlabel='Brier difference: temperature minus raw (negative favors temperature)',
               title=f'{label.replace("_"," ").title()}: primary paired contrasts\n95% cluster-bootstrap intervals, fixed trained seeds')
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
        save(fig,out,f'F4_primary_contrasts_{label}')
    # F5: Show both error retention and positive-case retention, avoiding majority-class-only conclusions.
    curves=result['risk_curves']
    for model,domain,label in [(m,d,l) for m in MODELS for d in ['nih','chexpert','vinbig'] for l in LABELS]:
        c=curves[(curves.model==model)&(curves.domain==domain)&(curves.label==label)&(curves.uncertainty_score=='entropy')]
        for metric in ['error_risk','positive_coverage']:
            fig,ax=plt.subplots(figsize=(6.8,4.8))
            for method in ['raw','temperature','head_mc','ensemble_temperature']:
                g=c[c.method==method].groupby('coverage')[metric].mean()
                ax.plot(g.index,g.values,label=method)
            ax.set(xlabel='Retained image fraction',ylabel=metric.replace('_',' '),
                   title=f'{model}: {label.replace("_"," ")} - {domain}\nEntropy ranking; source-frozen decision threshold')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
            save(fig,out,f'F5_retention_{model}_{domain}_{label}_{metric}')
    # F6: Calibration-label-budget sensitivity at full training, not mixed with training-data efficiency.
    for domain in ['nih','chexpert','vinbig']:
        for label in LABELS:
            fig,ax=plt.subplots(figsize=(6.8,4.8))
            d=metrics[(metrics.domain==domain)&(metrics.label==label)&(metrics.fraction==1)&(metrics.method=='temperature')]
            for model in MODELS:
                g=d[d.model==model].groupby('cal_fraction').brier.agg(['mean','std'])
                ax.errorbar(g.index*100,g['mean'],yerr=g['std'].fillna(0),marker='o',capsize=3,label=model)
            ax.set(xlabel='Source calibration patients (%)',ylabel='Brier score',
                   title=f'Calibration-label budget: {label.replace("_"," ")} - {domain}\nFull training data; failed low-event fits omitted and tabulated')
            ax.set_xticks([10,25,50,100]); ax.legend(); ax.grid(alpha=.2)
            save(fig,out,f'F6_calibration_budget_{domain}_{label}')
    # Supplement: ROC and precision-recall curves for calibrated full-data ensembles.
    for context,df in metadata.items():
        if context=='nih_calibration': continue
        for j,label in enumerate(LABELS):
            for kind in ['ROC','PR']:
                fig,ax=plt.subplots(figsize=(6,5))
                for model in MODELS:
                    p=variants[(model,1.,'ensemble','ensemble_temperature',1.,context)]['p'][:,j]
                    y=df[label].to_numpy(dtype=float); good=np.isfinite(y)&np.isfinite(p)
                    if len(np.unique(y[good]))<2: continue
                    if kind=='ROC': x,v,_=roc_curve(y[good],p[good])
                    else: v,x,_=precision_recall_curve(y[good],p[good])
                    ax.plot(x,v,label=model)
                ax.set(xlabel='False-positive rate' if kind=='ROC' else 'Recall',
                       ylabel='True-positive rate' if kind=='ROC' else 'Precision',
                       title=f'{kind}: {label.replace("_"," ")} - {context}')
                ax.legend(); ax.grid(alpha=.2)
                save(fig,out,f'SF1_{kind}_{context}_{label}')
    captions='''# Figure captions and selection guide

F1: Executed study flow. Counts are image counts, not claimed independent patients.
F2: Learning curves. Training patients vary; tuning and calibration labels are fixed. Bars are
SD over three trained seeds, not confidence intervals. Main text: select Brier and AUROC panels;
NLL and ECE panels can be supplementary. No plateau/non-inferiority inference is automated.
F3: Reliability diagrams (15 equal-width probability bins). Seed 17 is prespecified for individual
models; the three-seed ensemble is explicitly labeled. Empty bins are omitted. Bin counts are
saved as CSV. Sparse high-probability bins should not be overinterpreted.
F4: Primary paired Brier-score changes, temperature minus raw. Negative values favor
source-fitted temperature scaling. Intervals resample test patients (images for VinBigData),
conditional on fitted models/calibrators. Multiplicity-adjusted p values are in Table T3.
F5: Selective error and positive-case coverage using entropy. Decision thresholds were selected
in source calibration to target 90% sensitivity and then frozen. These are retrospective
reference-label assessments, not a validated clinical referral policy.
F6: Calibration-label-budget sensitivity at 100% training. Failed low-event fits are tabulated,
not replaced with an identity calibrator or treated as successful.
SF1: ROC and PR curves for calibrated three-seed ensembles. PR curves depend on cohort prevalence.

All figure panels are exported individually as 300-dpi PNG, vector PDF and editable SVG.
Choose a readable subset for the main manuscript; retain remaining panels in a supplement.
'''
    (out/'FIGURE_CAPTIONS.md').write_text(captions)
