"""Finite-population probability sampling and design-aware teaching inference.

No population values are invented here. Only sample selections are random.
Systematic single-start and convenience designs deliberately do not receive
an unsupported design-based standard error.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SRS='Simple random (without replacement)'
STRAT_PROP='Stratified (proportional allocation)'
STRAT_EQUAL='Stratified (equal allocation)'
SYSTEMATIC='Systematic (one random start)'
CLUSTER='Cluster (whole groups)'
TWO_STAGE='Two-stage (groups then units)'
CONVENIENCE='Convenience (accessible early records)'
METHODS=[SRS,STRAT_PROP,STRAT_EQUAL,SYSTEMATIC,CLUSTER,TWO_STAGE,CONVENIENCE]
DATA_DIR=Path(__file__).parent/'sampling_data'
MAX_RECORDED_DRAWS=200_000


def source_catalog():return json.loads((DATA_DIR/'sources.json').read_text())


def load_population(case):
    meta=source_catalog()[case]
    f=pd.read_csv(DATA_DIR/meta['file'],dtype={'unit_id':str,'cluster':str,'stratum':str})
    if not f.unit_id.is_unique or not np.isfinite(f.value).all():raise ValueError('Invalid bundled population.')
    return f,meta


def allocate_strata(sizes,n,equal=False):
    sizes=np.asarray(sizes,int);base=np.minimum(sizes,2)
    if n<base.sum():raise ValueError(f'This stratified design requires at least {base.sum()} units (two per non-singleton stratum).')
    if n>sizes.sum():raise ValueError('Sample size exceeds population size.')
    target=np.full(len(sizes),n/len(sizes)) if equal else n*sizes/sizes.sum()
    result=base.copy()
    for _ in range(n-int(base.sum())):
        deficit=np.where(result<sizes,target-result,-np.inf);result[int(np.argmax(deficit))]+=1
    return result


def interval_and_test(estimate,se,df,truth,level,mu0,alternative):
    if not np.isfinite(se):return dict(lower=np.nan,upper=np.nan,covers=np.nan,p_value=np.nan)
    if se==0:
        p=1. if np.isclose(estimate,mu0,rtol=1e-12,atol=1e-10) else (0. if alternative=='Two-sided' else float((estimate<mu0) if alternative=='Greater' else (estimate>mu0)))
        return dict(lower=estimate,upper=estimate,covers=bool(np.isclose(estimate,truth,rtol=1e-12,atol=1e-10)),p_value=p)
    q=stats.t.ppf((1+level)/2,df);z=(estimate-mu0)/se
    p=2*stats.t.sf(abs(z),df) if alternative=='Two-sided' else stats.t.sf(z,df) if alternative=='Greater' else stats.t.cdf(z,df)
    return dict(lower=estimate-q*se,upper=estimate+q*se,covers=bool(estimate-q*se<=truth<=estimate+q*se),p_value=float(p))


class SamplingFrame:
    def __init__(self,population):
        self.pop=population.reset_index(drop=True).copy();self.N=len(self.pop);self.y=self.pop.value.to_numpy(float)
        self.truth=float(self.y.mean())
        self.strata={str(k):np.array(v,int) for k,v in self.pop.groupby('stratum',sort=False).indices.items()}
        self.clusters={str(k):np.array(v,int) for k,v in self.pop.groupby('cluster',sort=False).indices.items()}
        self.cluster_names=list(self.clusters);self.M=len(self.clusters)

    def draw(self,config,rng,confidence=.95,mu0=None,alpha=.05,alternative='Two-sided'):
        mu0=self.truth if mu0 is None else float(mu0);method=config['method'];N=self.N
        n=int(config.get('n',30));m=int(config.get('clusters',2));per=int(config.get('per_cluster',5))
        if method not in METHODS:raise ValueError('Unknown sampling method.')
        if not 2<=n<=N and method not in [CLUSTER,TWO_STAGE]:raise ValueError(f'Choose 2–{N} units.')
        if method in [CLUSTER,TWO_STAGE] and not 2<=m<=self.M:raise ValueError(f'Choose 2–{self.M} clusters.')
        if method==TWO_STAGE and per<2:raise ValueError('Use at least two units per selected non-singleton cluster.')
        indices=[];pis=[];variance=np.nan;df=np.nan;valid=True;notes='';chosen=[]
        if method==SRS:
            indices=rng.choice(N,n,replace=False);pis=np.full(n,n/N)
            variance=0. if n==N else (1-n/N)*np.var(self.y[indices],ddof=1)/n;df=n-1
        elif method in [STRAT_PROP,STRAT_EQUAL]:
            items=list(self.strata.items());sizes=[len(ix) for _,ix in items]
            allocation=allocate_strata(sizes,n,method==STRAT_EQUAL);components=[];dofs=[]
            for (key,ix),nh in zip(items,allocation):
                sel=rng.choice(ix,int(nh),replace=False);indices.extend(sel);pis.extend([nh/len(ix)]*nh)
                term=0. if nh==len(ix) else (len(ix)/N)**2*(1-nh/len(ix))*np.var(self.y[sel],ddof=1)/nh
                components.append(term);dofs.append(max(1,nh-1))
            variance=float(sum(components));den=sum(v*v/d for v,d in zip(components,dofs))
            df=variance**2/den if den else np.inf
            notes='Stratum weights use known population sizes; unweighted and design estimates can differ.'
        elif method==SYSTEMATIC:
            order=config.get('order','Source order')
            if order=='Shuffled frame':ordered=np.random.default_rng(config.get('order_seed',902)).permutation(N)
            elif order=='Sorted by value':ordered=np.argsort(self.y,kind='stable')
            elif order=='Grouped by stratum':ordered=self.pop.sort_values(['stratum','order']).index.to_numpy()
            else:ordered=self.pop.sort_values('order').index.to_numpy()
            # Generalized interval N/n with one random start: exact size and equal first-order inclusion.
            j=int(rng.integers(N));positions=(j+np.arange(n)*N)//n
            indices=ordered[positions];pis=np.full(n,n/N);valid=False
            notes='One random-start systematic sample has no general unbiased variance estimator from the sample alone. Working SRS intervals ignore ordering and periodicity.'
        elif method in [CLUSTER,TWO_STAGE]:
            selected=rng.choice(self.M,m,replace=False);totals=[];within=[]
            for k in selected:
                key=self.cluster_names[int(k)];chosen.append(key);ix=self.clusters[key];Nh=len(ix)
                nh=Nh if method==CLUSTER else min(per,Nh)
                sel=ix.copy() if nh==Nh else rng.choice(ix,nh,replace=False)
                indices.extend(sel);pis.extend([(m/self.M)*(nh/Nh)]*nh)
                totals.append(Nh*self.y[sel].mean())
                within.append(0. if nh==Nh else Nh**2*(1-nh/Nh)*np.var(self.y[sel],ddof=1)/nh)
            variance=(self.M**2*(1-m/self.M)*np.var(totals,ddof=1)/m + (self.M/m)*sum(within))/N**2
            df=m-1
            notes='Horvitz–Thompson total divided by known N. Clusters are selected with equal probability; unequal cluster sizes are accounted for.'
        else:
            # Explicit simulated access restriction, not a claim about the source survey.
            pool_size=max(n,int(np.ceil(.35*N)));pool=self.pop.sort_values('order').index.to_numpy()[:pool_size]
            indices=rng.choice(pool,n,replace=False);pis=np.full(n,np.nan);valid=False
            notes='Random selection only within an artificial accessible first-35% frame (expanded if n is larger). Units outside it have zero selection probability; this is not a population probability sample.'
        ix=np.asarray(indices,int);pi=np.asarray(pis,float);values=self.y[ix];actual_n=len(ix)
        weights=1/pi
        estimate=float(np.sum(weights*values)/N) if valid or method==SYSTEMATIC else float(values.mean())
        sample_mean=float(values.mean());se=np.sqrt(max(0.,variance)) if np.isfinite(variance) else np.nan
        interval=interval_and_test(estimate,se,df,self.truth,confidence,mu0,alternative)
        working_se=0. if actual_n==N else float(np.sqrt((1-actual_n/N)*values.var(ddof=1)/actual_n))
        working=interval_and_test(sample_mean,working_se,actual_n-1,self.truth,confidence,mu0,alternative)
        out=dict(method=method,n=actual_n,population_size=N,estimate=estimate,sample_mean=sample_mean,
                 weighted_ratio=float(np.sum(weights*values)/np.sum(weights)) if np.isfinite(weights).all() else np.nan,
                 standard_error=se,degrees_freedom=df,true_mean=self.truth,estimation_error=estimate-self.truth,
                 confidence=confidence,null_mean=mu0,alpha=alpha,alternative=alternative,design_inference_available=valid,probability_sample=(method!=CONVENIENCE),
                 selected_clusters=json.dumps(chosen),analysis_note=notes,**interval)
        out['reject']=bool(interval['p_value']<alpha) if np.isfinite(interval['p_value']) else np.nan
        for k,v in working.items():out['working_'+k]=v
        out['working_standard_error']=working_se;out['working_reject']=bool(working['p_value']<alpha)
        sample=self.pop.iloc[ix].copy();sample.insert(0,'draw',np.arange(1,actual_n+1))
        sample['inclusion_probability']=pi;sample['sampling_weight']=weights
        return sample,pd.Series(out)


def collect_sample(population,config,seed,**inference):
    return SamplingFrame(population).draw(config,np.random.default_rng(seed),**inference)


def repeat_samples(population,config,repetitions,seed,**inference):
    if not 1<=repetitions<=3000:raise ValueError('Use 1–3,000 independent samples.')
    frame=SamplingFrame(population);method=config['method']
    expected=(config.get('clusters',2)*frame.N/frame.M if method==CLUSTER else
              sum(min(config.get('per_cluster',5),len(g)) for g in frame.clusters.values())/frame.M*config.get('clusters',2) if method==TWO_STAGE else config.get('n',30))
    if repetitions*expected>MAX_RECORDED_DRAWS:raise ValueError('Reduce sample size or repetitions (200,000 recorded-unit budget).')
    rows=[];raw=[];count=0
    for i,stream in enumerate(np.random.SeedSequence(seed).spawn(repetitions),1):
        sample,result=frame.draw(config,np.random.default_rng(stream),**inference);count+=len(sample)
        if count>MAX_RECORDED_DRAWS:raise ValueError('Realized sample-unit budget exceeded. Reduce repetitions or group count.')
        rows.append({'replicate':i,**result.to_dict()})
        raw.append(sample[['unit_id','value','stratum','cluster','inclusion_probability','sampling_weight']].assign(replicate=i))
    return pd.DataFrame(rows),pd.concat(raw,ignore_index=True)


def group_variation(population,column='stratum'):
    table=population.groupby(column,sort=False).value.agg(['count','mean','std','min','max']).reset_index()
    mu=population.value.mean();total=float(((population.value-mu)**2).sum())
    between=float((table['count']*(table['mean']-mu)**2).sum())
    return table,(between/total if total else 0.)
