"""Sample-mean calculations from recorded practical experiment outcomes."""
from __future__ import annotations
import numpy as np
import pandas as pd

WITH_REPLACEMENT='Resample recorded outcomes (with replacement)'
DISJOINT='Group recorded outcomes (no reuse)'


def empirical_sample_means(values,n,repetitions,seed,method=WITH_REPLACEMENT):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('At least two finite recorded outcomes are required.')
    if not isinstance(n,(int,np.integer)) or not 1<=n<=1000:raise ValueError('Sample size must be 1–1,000.')
    if not 1<=repetitions<=10000:raise ValueError('Use 1–10,000 repeated samples.')
    if method==WITH_REPLACEMENT:
        if n*repetitions>1_000_000:raise ValueError('Reduce sample size or repetitions (1 million draw budget).')
        indices=np.random.default_rng(seed).integers(0,len(x),size=(repetitions,n))
    elif method==DISJOINT:
        count=len(x)//n
        if count<1:raise ValueError(f'Need at least {n} completed outcomes to form one sample of size {n}.')
        indices=np.arange(count*n).reshape(count,n)
    else:raise ValueError('Unknown sampling method.')
    samples=x[indices]
    means=samples.mean(axis=1)
    table=pd.DataFrame({'replicate':np.arange(1,len(samples)+1),'sample_size':n,
                        'sample_mean':means,'sample_sum':samples.sum(axis=1),
                        'source_observations':len(x),'source_mean':x.mean(),
                        'source_sd':x.std(ddof=0),'reference_standard_error':x.std(ddof=0)/np.sqrt(n),
                        'sampling_method':method})
    return table,samples,indices


def long_samples(samples,indices,source_trials=None):
    R,n=samples.shape
    trials=np.asarray(source_trials) if source_trials is not None else np.arange(indices.max()+1)+1
    return pd.DataFrame({'replicate':np.repeat(np.arange(1,R+1),n),'draw_in_sample':np.tile(np.arange(1,n+1),R),
                         'source_trial':trials[indices.ravel()],'value':samples.ravel()})
