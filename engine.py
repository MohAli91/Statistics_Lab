"""Pure, reproducible simulation models. No Streamlit or display dependencies."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats

MAX_DRAWS = 2_000_000
DISTRIBUTIONS = ['Bernoulli', 'Binomial', 'Geometric', 'Negative binomial', 'Hypergeometric',
                 'Poisson', 'Discrete uniform', 'Custom discrete', 'Uniform', 'Normal',
                 'Exponential', 'Gamma', 'Weibull', 'Beta', 'Lognormal', 'Bimodal', 'Cauchy']
EXPERIMENTS = ['Coins', 'Dice', 'Cards', 'Darts', 'Basketball', 'Urn sampling',
               'Manufacturing', 'Arrivals', 'Lifetimes', 'Measurement', 'Galton board', 'Random walk']

def probabilities(text, length=None):
    p = np.asarray([float(x.strip()) for x in text.split(',')], dtype=float)
    if (length is not None and len(p) != length) or not len(p):
        raise ValueError(f'Enter {length or "at least one"} comma-separated weights.')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, and have a positive sum.')
    return p / p.sum()

def distribution(c):
    """Return a frozen SciPy law; geometric counts trials, NB counts failures."""
    name = c['distribution']
    if name == 'Bernoulli': return stats.bernoulli(c['p'])
    if name == 'Binomial': return stats.binom(c['n'], c['p'])
    if name == 'Geometric': return stats.geom(c['p'])
    if name == 'Negative binomial': return stats.nbinom(c['r'], c['p'])
    if name == 'Hypergeometric': return stats.hypergeom(c['N'], c['K'], c['n'])
    if name == 'Poisson': return stats.poisson(c['lam'])
    if name == 'Discrete uniform': return stats.randint(c['low'], c['high'] + 1)
    if name == 'Custom discrete':
        p = probabilities(c['weights'])
        return stats.rv_discrete(values=(np.arange(len(p)), p))
    if name == 'Uniform': return stats.uniform(c['low'], c['high'] - c['low'])
    if name == 'Normal': return stats.norm(c['mu'], c['sigma'])
    if name == 'Exponential': return stats.expon(scale=1 / c['rate'])
    if name == 'Gamma': return stats.gamma(c['shape'], scale=c['scale'])
    if name == 'Weibull': return stats.weibull_min(c['shape'], scale=c['scale'])
    if name == 'Beta': return stats.beta(c['alpha'], c['beta'])
    if name == 'Lognormal': return stats.lognorm(c['sigma'], scale=np.exp(c['mu']))
    if name == 'Cauchy': return stats.cauchy(c['mu'], c['scale'])
    if name == 'Bimodal': return None
    raise ValueError(f'Unknown distribution: {name}')

def draw(c, size, rng):
    if c['distribution'] == 'Bimodal':
        choose = rng.random(size) < c['p']
        return rng.normal(np.where(choose, c['mu1'], c['mu2']), c['sigma'], size)
    return distribution(c).rvs(size=size, random_state=rng)

def moments(c):
    if c['distribution'] == 'Bimodal':
        p = c['p']; mu = p*c['mu1'] + (1-p)*c['mu2']
        var = c['sigma']**2 + p*(1-p)*(c['mu1']-c['mu2'])**2
        return float(mu), float(var)
    law = distribution(c)
    mu, var = law.stats(moments='mv')
    return float(mu), float(var)

def density(c, x):
    if c['distribution'] == 'Bimodal':
        return c['p']*stats.norm.pdf(x, c['mu1'], c['sigma']) + (1-c['p'])*stats.norm.pdf(x, c['mu2'], c['sigma'])
    law = distribution(c)
    return law.pmf(x) if hasattr(law, 'pmf') else law.pdf(x)

def is_discrete(c):
    return c['distribution'] in DISTRIBUTIONS[:8]

def pack(value, raw=None, **extra):
    return {'value': float(value), 'details': json.dumps(raw or {}, separators=(',', ':')), **extra}

def simulate_one(kind, c, rng):
    """One independent experiment; details retain underlying draws/events."""
    n = c.get('n', 1)
    if kind == 'Coins':
        x = (rng.random(n) < c['p']).astype(int)
        return pack(x.sum(), {'draws': x.tolist()}, successes=int(x.sum()), attempts=n)
    if kind == 'Dice':
        x = rng.choice(np.arange(1, 7), n, p=probabilities(c['weights'], 6))
        val = x.sum() if c['statistic'] == 'Sum' else x.max()
        return pack(val, {'draws': x.tolist()})
    if kind == 'Cards':
        # Composition is counts of each rank PER SUIT, preserving actual finite cards.
        counts = [int(v) for v in c['counts'].split(',')]
        if len(counts) != 13 or min(counts) < 0 or sum(counts) == 0:
            raise ValueError('Enter 13 nonnegative integer rank counts per suit; total must be positive.')
        if sum(counts)*4 > 100_000: raise ValueError('Deck size must not exceed 100,000 cards.')
        deck = [(rank+1, suit) for suit in range(4) for rank, k in enumerate(counts) for _ in range(k)]
        if not c['replace'] and n > len(deck): raise ValueError('Hand exceeds deck size without replacement.')
        indices = rng.choice(len(deck), n, replace=c['replace'])
        hand = [deck[int(i)] for i in indices]
        hits = sum(rank == 1 for rank, suit in hand) if c['target'] == 'Aces' else sum(suit < 2 for rank, suit in hand)
        return pack(hits, {'cards': hand}, successes=hits, attempts=n)
    if kind == 'Darts':
        if c['model'] == 'Uniform square': x, y = rng.uniform(-1, 1, 2)
        else: x, y = rng.normal([c['aim_x'], c['aim_y']], [c['sx'], c['sy']])
        hit = int(x*x+y*y <= c['radius']**2)
        return pack(hit, {'draws': [hit]}, x=float(x), y=float(y), distance=float(np.hypot(x,y)))
    if kind == 'Basketball':
        mode = c['mode']
        if mode == 'Fixed attempts':
            x = (rng.random(n) < c['p']).astype(int)
            return pack(x.sum(), {'draws': x.tolist()}, successes=int(x.sum()), attempts=n)
        if mode == 'Until target makes':
            failures = int(rng.negative_binomial(c['r'], c['p']))
            total = failures + c['r']
            if total > 100_000: raise ValueError('Too many attempts in one trial. Increase p or reduce the target.')
            x = np.zeros(total, dtype=int)
            x[-1] = 1
            if c['r'] > 1: x[rng.choice(total-1, c['r']-1, replace=False)] = 1
            return pack(total, {'draws': x.tolist(), 'failures': failures}, successes=c['r'], attempts=total)
        rate = c['rate']; T = c['T']
        k = int(rng.poisson(rate*T))
        if k > 100_000: raise ValueError('Too many events in one window. Reduce rate or duration.')
        times = np.sort(rng.uniform(0, T, k))
        made = np.ones(k, dtype=int) if mode == 'Made baskets over time' else (rng.random(k)<c['p']).astype(int)
        return pack(made.sum(), {'times': times.tolist(), 'draws': made.tolist()}, successes=int(made.sum()), attempts=k, duration=T)
    if kind == 'Urn sampling':
        counts = np.array([c['red'], c['blue'], c['green']])
        if counts.sum() == 0: raise ValueError('The urn must contain at least one object.')
        if not c['replace'] and n > counts.sum(): raise ValueError('Sample exceeds urn size without replacement.')
        if c['replace']: x = rng.choice(3, n, p=counts/counts.sum())
        else: x = rng.choice(np.repeat(np.arange(3), counts), n, replace=False)
        return pack((x==0).sum(), {'draws': x.tolist()}, successes=int((x==0).sum()), attempts=n)
    if kind == 'Manufacturing':
        defects = rng.random(n) < c['p']
        flagged = np.where(defects, rng.random(n)<c['sensitivity'], rng.random(n)>(c['specificity']))
        return pack(defects.sum(), {'draws': defects.astype(int).tolist(), 'flagged': flagged.astype(int).tolist()},
                    flagged=int(flagged.sum()), false_positives=int((flagged & ~defects).sum()),
                    false_negatives=int((~flagged & defects).sum()), accepted=int(flagged.sum() <= c['accept']))
    if kind == 'Arrivals':
        k = int(rng.poisson(c['rate']*c['T']))
        if k > 100_000: raise ValueError('Too many events. Reduce rate or duration.')
        times = np.sort(rng.uniform(0, c['T'], k))
        # Gaps within a fixed window are retained, but not claimed to be an unbiased iid sample.
        service = rng.exponential(1/c['service_rate'], k) if c['queue'] else np.zeros(k)
        starts = []; ends = []; last_end = 0.
        for t, s in zip(times, service):
            start = max(float(t), last_end) if c['queue'] else float(t)
            last_end = start + float(s); starts.append(start); ends.append(last_end)
        waits = np.array(starts)-times
        return pack(k, {'times': times.tolist(), 'service_times': service.tolist(), 'service_starts': starts, 'service_ends': ends},
                    duration=c['T'], mean_wait=float(waits.mean()) if k else 0.)
    if kind == 'Lifetimes':
        x = rng.exponential(c['scale']) if c['model']=='Exponential' else c['scale']*rng.weibull(c['shape'])
        return pack(x, {'draws': [float(x)]})
    if kind == 'Measurement':
        x = rng.normal(c['truth']+c['bias'], c['sigma'])
        return pack(x, {'draws': [float(x)]}, error=float(x-c['truth']))
    if kind == 'Galton board':
        steps = (rng.random(n)<c['p']).astype(int)
        return pack(steps.sum(), {'draws': steps.tolist()})
    if kind == 'Random walk':
        steps = np.where(rng.random(n)<c['p'], 1, -1)
        return pack(steps.sum(), {'draws': steps.tolist(), 'path': np.r_[0, steps.cumsum()].tolist()})
    raise ValueError(kind)

def simulate(kind, configs, repetitions, seed):
    if not 1 <= repetitions <= 20000: raise ValueError('Use 1–20,000 repetitions.')
    cost = repetitions*sum(c.get('n',1) for c in configs)
    if cost > MAX_DRAWS: raise ValueError(f'Reduce repetitions or observations: maximum {MAX_DRAWS:,} underlying draws.')
    if kind in ('Arrivals', 'Basketball'):
        expected = sum(c['r']/c['p'] if c.get('mode')=='Until target makes' else c.get('rate',0)*c.get('T',0) for c in configs)*repetitions
        if expected > MAX_DRAWS/2: raise ValueError('Reduce rate, duration, or repetitions (event budget exceeded).')
    streams = np.random.SeedSequence(seed).spawn(len(configs))
    records = []
    for j, (c, stream) in enumerate(zip(configs, streams)):
        rng = np.random.default_rng(stream)
        for i in range(repetitions):
            records.append({'population': chr(65+j), 'trial': i+1, **simulate_one(kind,c,rng)})
    return pd.DataFrame(records)

def sampling(configs, n, repetitions, seed, statistic='Mean'):
    if n*repetitions*len(configs)>MAX_DRAWS: raise ValueError('Reduce sample size or repetitions (2 million draw limit).')
    records=[]
    for j, stream in enumerate(np.random.SeedSequence(seed).spawn(len(configs))):
        x=draw(configs[j], (repetitions,n), np.random.default_rng(stream))
        funcs={'Mean':lambda a:a.mean(1),'Sum':lambda a:a.sum(1),'Median':lambda a:np.median(a,axis=1),
               'Variance':lambda a:a.var(1,ddof=1),'Proportion > threshold':lambda a:(a>configs[j].get('threshold',0)).mean(1)}
        values=funcs[statistic](x)
        for i,v in enumerate(values):
            records.append({'population':chr(65+j),'trial':i+1,**pack(v,{'draws':x[i].tolist()})})
    return pd.DataFrame(records)

def raw_records(df):
    """Long-form raw observations, including event times and card rank/suit."""
    out=[]
    for row in df.to_dict('records'):
        d=json.loads(row['details']); base={k:v for k,v in row.items() if k!='details'}
        if 'cards' in d:
            for i,(rank,suit) in enumerate(d['cards']): out.append({**base,'draw':i+1,'rank':rank,'suit':suit})
        elif 'times' in d:
            for i,t in enumerate(d['times']):
                item={**base,'draw':i+1,'time':t}
                for k in ['draws','service_times','service_starts','service_ends']:
                    if k in d: item[k if k!='draws' else 'outcome']=d[k][i]
                out.append(item)
        elif 'draws' in d:
            for i,v in enumerate(d['draws']):
                item={**base,'draw':i+1,'outcome':v}
                if 'group_b' in d: item['resampled_group']='A'
                if 'flagged' in d: item['flagged']=d['flagged'][i]
                out.append(item)
            if 'group_b' in d:
                for i,v in enumerate(d['group_b']): out.append({**base,'draw':i+1,'outcome':v,'resampled_group':'B'})
        else: out.append({**base,'draw':1,'outcome':row['value']})
    return pd.DataFrame(out)

def confidence_coverage(c,n,R,seed,level):
    if n*R>MAX_DRAWS: raise ValueError('Reduce sample size or repetitions.')
    x=draw(c,(R,n),np.random.default_rng(seed)); mu,var=moments(c)
    if not np.isfinite(mu): raise ValueError('This population has no finite mean.')
    means=x.mean(1); se=x.std(1,ddof=1)/np.sqrt(n); q=stats.t.ppf((1+level)/2,n-1)
    lo=means-q*se; hi=means+q*se
    return pd.DataFrame({'trial':np.arange(1,R+1),'mean':means,'lower':lo,'upper':hi,'covers':(lo<=mu)&(mu<=hi),'true_mean':mu})

def testing(p0,p1,n,R,alpha,seed):
    rng=np.random.default_rng(seed); x=rng.binomial(n,p1,R)
    lookup={int(k):stats.binomtest(int(k),n,p0).pvalue for k in np.unique(x)}
    pv=np.array([lookup[int(k)] for k in x])
    return pd.DataFrame({'trial':np.arange(1,R+1),'successes':x,'estimate':x/n,'p_value':pv,'reject':pv<alpha,'null_p':p0,'true_p':p1})

def resample(values, R, seed, other=None, method='Bootstrap'):
    a=np.asarray(values,float); rng=np.random.default_rng(seed); records=[]
    b=None if other is None else np.asarray(other,float)
    if (len(a)+(len(b) if b is not None else 0))*R>MAX_DRAWS: raise ValueError('Reduce data length or repetitions.')
    for i in range(R):
        if method=='Bootstrap':
            aa=rng.choice(a,len(a),replace=True)
            bb=None if b is None else rng.choice(b,len(b),replace=True)
        else:
            if b is None: raise ValueError('Permutation requires two groups.')
            pool=rng.permutation(np.r_[a,b]); aa=pool[:len(a)]; bb=pool[len(a):]
        v=aa.mean() if bb is None else aa.mean()-bb.mean()
        records.append({'population':'A','trial':i+1,**pack(v,{'draws':aa.tolist(), 'group_b':[] if bb is None else bb.tolist()})})
    return pd.DataFrame(records)
