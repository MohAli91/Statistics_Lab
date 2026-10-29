"""Applied Statistics Simulation Lab — run with: streamlit run app.py"""
from __future__ import annotations
import html
import io
import json
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from basketball_ui import render_basketball_experiment
from practical_clt_ui import render_practical_clt
from engine import (EXPERIMENTS, DISTRIBUTIONS, probabilities, draw, distribution, moments,
                    density, is_discrete, simulate, sampling, raw_records,
                    confidence_coverage, testing, resample)

st.set_page_config(page_title='Statistics Simulation Lab',page_icon='🎲',layout='wide')
COLORS={'A':'#0d9488','B':'#e87926'}
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1500px}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #dce5ed;border-radius:12px;padding:12px}
.scene{min-height:175px;border:1px solid #dce5ed;background:linear-gradient(140deg,#fff,#edf5f8);border-radius:16px;padding:20px;text-align:center;color:#12263a;overflow:hidden}
.scene .objects{font-size:40px;line-height:1.8;letter-spacing:5px;animation:appear .35s ease-out}
.scene .caption{font:15px sans-serif;letter-spacing:0}
.scene .coin{display:inline-block;animation:flip .7s ease-out}.scene .die{display:inline-block;animation:tumble .6s ease-out}.scene .ball{display:inline-block;animation:shot 1s ease-in-out}
@keyframes flip{0%{transform:rotateY(0)}100%{transform:rotateY(1080deg)}}
@keyframes tumble{0%{transform:translateY(-20px) rotate(-180deg)}100%{transform:translateY(0) rotate(0)}}
@keyframes shot{0%{transform:translate(-90px,30px)}50%{transform:translate(0,-40px)}100%{transform:translate(85px,0)}}
@keyframes appear{from{transform:translateY(-14px) rotate(-5deg);opacity:.3}to{transform:translateY(0);opacity:1}}
</style>''',unsafe_allow_html=True)

def number(label,value,min_value=None,max_value=None,key=None,step=None):
    kw={'value':value,'key':key}
    if min_value is not None: kw['min_value']=min_value
    if max_value is not None: kw['max_value']=max_value
    if step is not None: kw['step']=step
    return st.number_input(label,**kw)

def prob(label,key,value=.5,positive=False):
    return st.slider(label,.01 if positive else 0.,1.,value,.01,key=key)

def dist_controls(key):
    name=st.selectbox('Distribution',DISTRIBUTIONS,index=9,key=key+'dist')
    c={'distribution':name}
    def num(label,v,lo=None,hi=None): return number(label,v,lo,hi,key=key+label)
    if name in ['Bernoulli','Binomial','Geometric','Negative binomial','Bimodal']:
        c['p']=prob('Probability / mixture weight',key+'p',positive=name in ['Geometric','Negative binomial'])
    if name=='Binomial': c['n']=num('Binomial attempts',10,1,1000)
    if name=='Negative binomial': c['r']=num('Target successes',3,1,100)
    if name=='Hypergeometric':
        c['N']=num('Population size N',50,2,10000)
        c['K']=num('Successes in population K',min(20,c['N']),0,c['N'])
        c['n']=num('Draws without replacement',min(10,c['N']),1,c['N'])
    if name=='Poisson': c['lam']=num('Mean count λ',4.,.01,1000.)
    if name in ['Uniform','Discrete uniform']:
        integer=name=='Discrete uniform'
        c['low']=num('Lower endpoint',0 if integer else 0.,-1000 if integer else -1000.,999 if integer else 999.)
        c['high']=num('Upper endpoint',max(10 if integer else 1.,c['low']+(1 if integer else .01)),c['low']+(1 if integer else .001),1000 if integer else 1000.)
    if name=='Custom discrete':
        c['weights']=st.text_input('Weights for outcomes 0, 1, 2, …','1,2,3,2,1',key=key+'weights')
        st.caption('Edit the weights to construct your own discrete population. Weights are normalized.')
    if name in ['Normal','Lognormal','Cauchy']:
        c['mu']=num('Location μ / log-location',0.,-100.,100.)
    if name in ['Normal','Lognormal','Bimodal']: c['sigma']=num('Standard deviation / log-SD',1.,.01,20.)
    if name in ['Gamma','Weibull']: c['shape']=num('Shape',2.,.1,20.)
    if name in ['Gamma','Weibull','Cauchy']: c['scale']=num('Scale',1.,.01,100.)
    if name=='Exponential': c['rate']=num('Rate per time unit',1.,.01,100.)
    if name=='Beta':
        c['alpha']=num('Alpha',2.,.1,50.); c['beta']=num('Beta',5.,.1,50.)
    if name=='Bimodal':
        c['mu1']=num('First component mean',-3.,-100.,100.)
        c['mu2']=num('Second component mean',3.,-100.,100.)
    return c

def experiment_controls(kind,key):
    c={}
    def num(label,v,lo=None,hi=None): return number(label,v,lo,hi,key=key+label)
    if kind in ['Coins','Dice','Cards','Urn sampling','Manufacturing','Galton board','Random walk']:
        labels={'Coins':'Flips per trial','Dice':'Dice per trial','Cards':'Cards per hand','Urn sampling':'Draws per trial',
                'Manufacturing':'Products per batch','Galton board':'Peg rows','Random walk':'Steps per walk'}
        defaults={'Coins':1,'Dice':1,'Cards':5,'Urn sampling':10,'Manufacturing':20,'Galton board':12,'Random walk':50}
        c['n']=num(labels[kind],defaults[kind],1,1000)
    if kind in ['Coins','Manufacturing','Galton board','Random walk']:
        label={'Coins':'Probability of heads','Manufacturing':'Defect probability','Galton board':'Probability of moving right','Random walk':'Probability of +1 step'}[kind]
        c['p']=prob(label,key+'p',.05 if kind=='Manufacturing' else .5)
    if kind=='Dice':
        c['weights']=st.text_input('Weights for faces 1–6','1,1,1,1,1,1',key=key+'weights')
        c['statistic']=st.selectbox('Record','Sum Maximum'.split(),key=key+'stat')
    if kind=='Cards':
        c['counts']=st.text_input('Copies of A,2,…,10,J,Q,K per suit','1,1,1,1,1,1,1,1,1,1,1,1,1',key=key+'counts')
        c['replace']=st.checkbox('Replace each card after drawing',key=key+'replace')
        c['target']=st.selectbox('Count in each hand',['Aces','Red cards'],key=key+'target')
        st.caption('Each trial starts with a fresh deck. Suits 0–1 are red. Extra rank copies change deck composition.')
    if kind=='Darts':
        c['model']=st.selectbox('Hit model',['Clustered aim','Uniform square'],key=key+'model')
        c['aim_x']=num('Aim x',0.,-2.,2.); c['aim_y']=num('Aim y',0.,-2.,2.)
        c['sx']=num('Horizontal SD',.4,.01,2.); c['sy']=num('Vertical SD',.4,.01,2.)
        c['radius']=num('Target radius',.5,.05,1.)
        st.caption('Uniform square uses [−1,1]²; aim and SD apply only to clustered aim.')
    if kind=='Basketball':
        c['mode']=st.selectbox('Model',['Fixed attempts','Until target makes','Made baskets over time','Attempts over time'],key=key+'mode')
        c['p']=prob('Make probability',key+'p',.65,positive=c['mode']=='Until target makes')
        if c['mode']=='Fixed attempts': c['n']=num('Attempts per trial',10,1,1000)
        if c['mode']=='Until target makes': c['r']=num('Target makes',3,1,100)
        if 'time' in c['mode']:
            c['rate']=num('λ: makes/min' if c['mode']=='Made baskets over time' else 'Attempts/min',2.,.01,100.)
            c['T']=num('Window T (minutes)',5.,.1,100.)
            st.caption('λT is the expected count. Direct makes mode does not use p; attempts mode independently marks attempts as make/miss.')
    if kind=='Urn sampling':
        for color,default in [('red',20),('blue',20),('green',10)]: c[color]=num(color.title()+' objects',default,0,1000)
        c['replace']=st.checkbox('Replace after each draw',True,key=key+'replace')
    if kind=='Manufacturing':
        c['sensitivity']=prob('Inspection sensitivity',key+'sens',.95)
        c['specificity']=prob('Inspection specificity',key+'spec',.95)
        c['accept']=num('Accept batch if flagged ≤',1,0,c['n'])
    if kind=='Arrivals':
        c['rate']=num('Arrival rate λ (per minute)',2.,.01,100.)
        c['T']=num('Window T (minutes)',5.,.1,100.)
        c['queue']=st.checkbox('Add single-server queue',key=key+'queue')
        c['service_rate']=num('Service rate (per minute)',3.,.01,100.)
        st.caption('Independent Poisson windows. Queue starts empty in every trial; service can finish after the window ends.')
    if kind=='Lifetimes':
        c['model']=st.selectbox('Lifetime model',['Exponential','Weibull'],key=key+'model')
        c['scale']=num('Scale (time units)',10.,.01,1000.)
        c['shape']=num('Weibull shape',2.,.1,20.)
    if kind=='Measurement':
        c['truth']=num('True value',10.,-1000.,1000.)
        c['bias']=num('Instrument bias',0.,-100.,100.)
        c['sigma']=num('Measurement SD',1.,.01,100.)
    return c

def setup_populations(builder):
    compare=st.checkbox('Compare populations A and B',True)
    hidden=st.checkbox('Hide true parameters in results (detect mode)',False,
        help='Configure the populations, then collapse Settings before students investigate. This is a classroom display mode, not a security boundary.')
    configs=[]
    for j,col in enumerate(st.columns(2 if compare else 1)):
        with col:
            st.markdown(f'**Population {chr(65+j)}**')
            configs.append(builder(chr(65+j)))
    return configs, hidden

def start_run(frame,meta):
    meta={**meta,'created_utc':datetime.now(timezone.utc).isoformat(),'schema_version':1}
    st.session_state.reveal=False
    st.session_state.run={'data':frame,'meta':meta,'shown':0,'playing':False,'phase':0,'clock_fraction':0.}
    st.session_state.archive=st.session_state.get('archive',[])[-4:]
    # Archive keeps up to five independent saved runs, including complete generated outcomes.
    st.session_state.archive.append({'data':frame,'meta':meta})

def action_generate(fn,meta):
    if st.button('Generate experiment',type='primary'):
        try:
            with st.spinner('Preparing reproducible outcomes…'): start_run(fn(),meta)
        except (ValueError,TypeError,OverflowError) as exc: st.error(str(exc))

def common_trials(prefix=''):
    a,b=st.columns(2)
    with a: R=number('Number of independent trials / repetitions',500,1,20000,key=prefix+'R')
    with b: seed=number('Random seed',42,0,2**32-1,key=prefix+'seed')
    return R,seed

def figure_layout(fig,title='',height=320):
    fig.update_layout(title=title,height=height,margin=dict(l=25,r=25,t=45,b=30),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',font=dict(color='#12263a'),
        legend=dict(orientation='h',y=1.12),transition_duration=180)
    return fig

def theoretical(kind,c):
    """Exact reference law for supported recorded statistics."""
    n=c.get('n',1)
    if kind in ['Coins','Galton board','Manufacturing']: return stats.binom(n,c['p'])
    if kind=='Dice' and c['statistic']=='Sum':
        p=probabilities(c['weights'],6); mass=np.array([1.])
        for _ in range(n): mass=np.convolve(mass,p)
        return stats.rv_discrete(values=(np.arange(n,6*n+1),mass))
    if kind=='Dice':
        cdf=np.cumsum(probabilities(c['weights'],6))**n
        return stats.rv_discrete(values=(np.arange(1,7),np.diff(np.r_[0,cdf])))
    if kind=='Cards':
        counts=np.array([int(v) for v in c['counts'].split(',')]); N=4*counts.sum()
        K=4*counts[0] if c['target']=='Aces' else N//2
        return stats.binom(n,K/N) if c['replace'] else stats.hypergeom(N,K,n)
    if kind=='Urn sampling':
        N=c['red']+c['blue']+c['green']; K=c['red']
        return stats.binom(n,K/N) if c['replace'] else stats.hypergeom(N,K,n)
    if kind=='Basketball':
        if c['mode']=='Fixed attempts': return stats.binom(n,c['p'])
        if c['mode']=='Until target makes': return stats.nbinom(c['r'],c['p'],loc=c['r'])
        return stats.poisson(c['rate']*c['T']*(c['p'] if c['mode']=='Attempts over time' else 1))
    if kind=='Arrivals': return stats.poisson(c['rate']*c['T'])
    if kind=='Lifetimes': return stats.expon(scale=c['scale']) if c['model']=='Exponential' else stats.weibull_min(c['shape'],scale=c['scale'])
    if kind=='Measurement': return stats.norm(c['truth']+c['bias'],c['sigma'])
    if kind=='Random walk': return stats.rv_discrete(values=(2*np.arange(n+1)-n,stats.binom.pmf(np.arange(n+1),n,c['p'])))
    if kind=='Darts' and c['model']=='Uniform square': return stats.bernoulli(np.pi*c['radius']**2/4)
    return None

def scene(row,kind,c,phase=0,fraction=1.):
    d=json.loads(row['details']); draws=d.get('draws',[])
    caption=f'Trial {int(row.trial)} · recorded value = {row.value:g}' if fraction>=1. else f'Trial {int(row.trial)} · window in progress'
    if kind in ['Darts','Random walk','Arrivals','Galton board','Lifetimes','Measurement']:
        fig=go.Figure()
        if kind=='Darts':
            fig.add_shape(type='circle',x0=-c['radius'],y0=-c['radius'],x1=c['radius'],y1=c['radius'],fillcolor='#ccfbf1',line_color='#0d9488')
            fig.add_trace(go.Scatter(x=[row.x],y=[row.y],mode='markers',marker=dict(size=14,color='#e87926',symbol='x')))
            lim=max(1.2,abs(row.x)+.1,abs(row.y)+.1); fig.update_xaxes(range=[-lim,lim]);fig.update_yaxes(range=[-lim,lim],scaleanchor='x',scaleratio=1)
        elif kind=='Random walk':
            path=d['path']; fig.add_trace(go.Scatter(x=list(range(len(path))),y=path,mode='lines+markers',line_color=COLORS[row.population]))
        elif kind=='Galton board':
            x=0; xs=[0]; ys=[0]
            for i,s in enumerate(draws): x+=1 if s else -1;xs.append(x);ys.append(-(i+1))
            for i in range(min(c['n'],40)+1):
                fig.add_trace(go.Scatter(x=list(range(-i,i+1,2)),y=[-i]*(i+1),mode='markers',marker=dict(size=4,color='#94a3b8'),showlegend=False))
            fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines+markers',line_color=COLORS[row.population],showlegend=False))
        elif kind=='Arrivals':
            ts=np.array(d['times']); now=c['T']*fraction; visible=ts[ts<=now]
            fig.add_trace(go.Scatter(x=visible,y=np.ones(len(visible)),mode='markers',marker=dict(symbol='line-ns',size=25,color=COLORS[row.population]),name='Arrivals'))
            if c['queue']:
                for t,s,e in zip(ts,d['service_starts'],d['service_ends']):
                    if t<=now: fig.add_trace(go.Scatter(x=[s,e],y=[.5,.5],mode='lines',line=dict(width=8,color=COLORS[row.population]),showlegend=False))
            fig.add_vline(x=now,line_dash='dash');fig.update_xaxes(range=[0,max(c['T'],max(d['service_ends'],default=0))],title='Simulated minutes')
            fig.update_yaxes(range=[0,1.5],tickvals=[.5,1],ticktext=['Service','Arrival'])
            caption+=f' · clock {now:.2f}/{c["T"]:g} min'
        elif kind=='Lifetimes':
            fig.add_trace(go.Scatter(x=[0,row.value],y=[1,1],mode='lines+markers',line=dict(width=12,color=COLORS[row.population])))
            fig.update_xaxes(title='Lifetime');fig.update_yaxes(visible=False)
        else:
            fig.add_vline(x=c['truth'],line_color='#64748b',annotation_text='True value')
            fig.add_trace(go.Scatter(x=[row.value],y=[1],mode='markers',marker=dict(size=20,color=COLORS[row.population])))
            span=max(4*c['sigma'],abs(c['bias'])+2*c['sigma']);fig.update_xaxes(range=[c['truth']-span,c['truth']+span]);fig.update_yaxes(visible=False)
        st.plotly_chart(figure_layout(fig,caption,230),use_container_width=True,key=f'scene{row.population}')
        return
    if kind=='Coins': objects=' '.join('🟡' if v else '⚪' for v in draws[:24]); caption+=' · 🟡 heads / ⚪ tails'
    elif kind=='Dice': objects=' '.join('⚀⚁⚂⚃⚄⚅'[int(v)-1] for v in draws[:24])
    elif kind=='Cards':
        ranks=['','A','2','3','4','5','6','7','8','9','10','J','Q','K']; suits=['♥','♦','♣','♠']
        objects=' '.join(f'{ranks[r]}{suits[s]}' for r,s in d['cards'][:16])
    elif kind=='Basketball':
        if 'times' in d:
            now=c['T']*fraction; ix=np.array(d['times'])<=now; show=np.array(draws)[ix].tolist()
            objects='🏀 '+' '.join('✓' if x else '×' for x in show[-20:]); caption+=f' · {sum(show)} makes at {now:.2f} min'
        else: objects='🏀 '+' '.join('✓' if x else '×' for x in draws[:24]) if draws else f'🏀 {int(row.successes)} makes / {int(row.attempts)} attempts'
    elif kind=='Urn sampling': objects=' '.join(['🔴','🔵','🟢'][int(v)] for v in draws[:24])
    elif kind=='Manufacturing':
        objects=' '.join('🔴' if v else '🟩' for v in draws[:24]);caption+=f' · flagged {int(row.flagged)} · '+('accepted' if row.accepted else 'rejected')
    else: objects=' '.join(f'{v:.2g}' for v in draws[:12])
    if len(draws)>24: caption+=f' · showing 24 of {len(draws)} outcomes'
    # CSS transition replays on each newly rendered result; all displayed text is escaped.
    animation_class={'Coins':'coin','Dice':'die','Basketball':'ball'}.get(kind,'')
    st.markdown(f'<div class="scene"><div class="objects"><span class="{animation_class}">{html.escape(objects)}</span></div><div class="caption">{html.escape(caption)}</div></div>',unsafe_allow_html=True)


def hist_and_summary(visible,meta,show_truth):
    if visible.empty: return
    a,b=st.columns([1.25,1])
    with a:
        vals=visible.value.to_numpy(); discrete=meta['section']=='Experiments' and meta['kind'] not in ['Lifetimes','Measurement']
        fig=go.Figure()
        if discrete:
            low=int(vals.min());high=int(vals.max());width=max(1,int(np.ceil((high-low+1)/80)))
            edges=np.arange(low-.5,high+width+.5,width)
        else:
            lo,hi=np.quantile(vals,[0,1]);edges=np.linspace(lo-.5 if lo==hi else lo,hi+.5 if lo==hi else hi,31)
        for pop,g in visible.groupby('population'):
            fig.add_trace(go.Histogram(x=g.value,name=f'Population {pop}',histnorm='probability' if discrete else 'probability density',
                xbins=dict(start=float(edges[0]),end=float(edges[-1]),size=float(edges[1]-edges[0])),opacity=.55,marker_color=COLORS[pop]))
            if show_truth and meta['section']=='Experiments':
                law=theoretical(meta['kind'],meta['configs'][ord(pop)-65])
                if law is not None:
                    xx=np.arange(low,high+1) if hasattr(law,'pmf') else np.linspace(edges[0],edges[-1],250)
                    yy=law.pmf(xx) if hasattr(law,'pmf') else law.pdf(xx)
                    if discrete and width>1: continue
                    fig.add_trace(go.Scatter(x=xx,y=yy,name=f'{pop} theory',line=dict(color=COLORS[pop],dash='dash')))
        fig.update_layout(barmode='overlay');fig.update_xaxes(title='Recorded outcome / statistic')
        st.plotly_chart(figure_layout(fig,'Observed distributions'),use_container_width=True,key='hist')
    with b:
        table=visible.groupby('population').value.agg(['count','mean','std','min','max']).reset_index()
        st.dataframe(table,hide_index=True,use_container_width=True)
        if 'successes' in visible and 'attempts' in visible:
            for pop,g in visible.groupby('population'):
                total=g.attempts.sum()
                if total>0: st.caption(f'{pop}: observed success fraction = {g.successes.sum()/total:.4g} ({int(g.successes.sum())}/{int(total)})')
        if visible.population.nunique()==2:
            ga=visible[visible.population=='A'].value;gb=visible[visible.population=='B'].value
            st.metric('Observed mean A − mean B',f'{ga.mean()-gb.mean():.4g}')
            if len(ga)>1 and len(gb)>1:
                v1=ga.var()/len(ga);v2=gb.var()/len(gb);se=np.sqrt(v1+v2)
                if se>0:
                    df=(v1+v2)**2/(v1*v1/(len(ga)-1)+v2*v2/(len(gb)-1)); q=stats.t.ppf(.975,df)
                    delta=ga.mean()-gb.mean();st.caption(f'Approximate Welch 95% interval for mean difference: [{delta-q*se:.4g}, {delta+q*se:.4g}]. Treat cautiously for tiny or highly skewed samples.')
        if show_truth and meta['section']=='Experiments':
            for j,c in enumerate(meta['configs']):
                law=theoretical(meta['kind'],c)
                if law is not None: st.caption(f'{chr(65+j)} theoretical mean = {law.mean():.4g}; SD = {law.std():.4g}')
    trend=go.Figure()
    for pop,g in visible.groupby('population'):
        y=g.value.expanding().mean()
        stride=max(1,len(g)//1500);trend.add_trace(go.Scatter(x=g.trial.iloc[::stride],y=y.iloc[::stride],name=pop,line_color=COLORS[pop]))
        if show_truth and meta['section']=='Experiments':
            law=theoretical(meta['kind'],meta['configs'][ord(pop)-65])
            if law is not None: trend.add_hline(y=law.mean(),line_dash='dash',line_color=COLORS[pop])
    trend.update_xaxes(title='Independent trials');trend.update_yaxes(title='Running average')
    st.plotly_chart(figure_layout(trend,'Law of large numbers · running average',250),use_container_width=True,key='trend')


def clt_view(visible,meta,show_truth):
    stat=meta['statistic'];n=meta['n']
    if visible.empty:return
    cols=st.columns(len(meta['configs']))
    for j,col in enumerate(cols):
        c=meta['configs'][j];pop=chr(65+j);g=visible[visible.population==pop]
        if g.empty:continue
        with col:
            last=g.iloc[-1];sample=np.array(json.loads(last.details)['draws'])
            st.markdown(f'**Population {pop} · sample {int(last.trial)}**')
            a=go.Figure()
            if show_truth:
                demo=draw(c,10000,np.random.default_rng(912+j));lo,hi=np.quantile(demo,[.005,.995])
                xx=np.arange(np.floor(lo),np.ceil(hi)+1) if is_discrete(c) else np.linspace(lo,hi,250)
                a.add_trace(go.Scatter(x=xx,y=density(c,xx),name='Population',mode='lines+markers' if is_discrete(c) else 'lines',line_color=COLORS[pop]))
                a.update_yaxes(title='PMF' if is_discrete(c) else 'Density')
            else:
                a.add_trace(go.Histogram(x=sample,name='Current sample',marker_color=COLORS[pop]))
            st.plotly_chart(figure_layout(a,'1 · Original population' if show_truth else '1 · Population hidden',210),use_container_width=True,key='pop'+pop)
            current=go.Figure(go.Scatter(x=sample,y=np.zeros(n),mode='markers',marker=dict(color=COLORS[pop],size=9,opacity=.7)))
            current.add_vline(x=sample.mean(),line_color='#e87926',annotation_text='sample mean')
            current.update_yaxes(visible=False)
            st.plotly_chart(figure_layout(current,f'2 · Draw n={n} observations → {stat.lower()} = {last.value:.4g}',190),use_container_width=True,key='sample'+pop)
            vals=g.value.to_numpy();mu,var=moments(c);standard=meta.get('standardize',False)
            valid=np.isfinite(mu) and np.isfinite(var) and var>0 and stat in ['Mean','Sum']
            center=mu if stat=='Mean' else n*mu
            sd=np.sqrt(var/n) if stat=='Mean' else np.sqrt(n*var)
            if standard and valid: vals=(vals-center)/sd; center=0.;sd=1.
            hist=go.Figure(go.Histogram(x=vals,histnorm='probability density',marker_color=COLORS[pop],opacity=.6,name=f'{stat} statistics'))
            if valid and show_truth:
                xx=np.linspace(center-4*sd,center+4*sd,250)
                hist.add_trace(go.Scatter(x=xx,y=stats.norm.pdf(xx,center,sd),name='Normal approximation',line_color='#e87926'))
            st.plotly_chart(figure_layout(hist,f'3 · Sampling distribution · R={len(g)}',260),use_container_width=True,key='clthist'+pop)
            st.caption(f'Observed mean of statistics: {g.value.mean():.4g}; observed SD: {g.value.std():.4g}')
            if show_truth and valid: st.caption(f'Theoretical center: {mu if stat=="Mean" else n*mu:.4g}; SD: {np.sqrt(var/n) if stat=="Mean" else np.sqrt(n*var):.4g}')
            if not np.isfinite(var):st.warning('This population has no finite variance: the usual CLT guarantee does not apply. Cauchy sample means do not concentrate normally.')
    st.info('One completed sample contributes ONE statistic. Increasing n changes its sampling distribution; increasing R improves the simulated picture. Normal approximation can require large n for strongly skewed populations. Median and variance do not use the mean/sum normal overlay.')


def downloads(visible,run):
    meta=run['meta'];safe_meta={**meta}
    if meta.get('hidden') and not st.session_state.get('reveal',False):
        safe_meta.pop('configs',None);safe_meta['seed']='withheld in detect mode'
    if not st.checkbox('Enable downloads for the current playback position',False,key='enable_downloads'):
        st.caption('Enable downloads when you are ready to export. Pause first to keep the export position fixed.')
        return
    st.download_button('Download recorded results · CSV',visible.assign(experiment=meta.get('kind',meta['section'])).to_csv(index=False).encode(),
        'recorded_results.csv','text/csv',key='results_csv')
    if st.checkbox('Prepare raw observations and reproducibility package',False,key='prepare_export'):
        raw=raw_records(visible)
        st.download_button('Download raw draws / events · CSV',raw.to_csv(index=False).encode(),'raw_observations.csv','text/csv',key='raw_csv')
        z=io.BytesIO()
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as out:
            out.writestr('results.csv',visible.to_csv(index=False));out.writestr('raw_observations.csv',raw.to_csv(index=False))
            out.writestr('settings.json',json.dumps(safe_meta,indent=2))
        st.download_button('Download results + raw data + settings · ZIP',z.getvalue(),'experiment_results.zip','application/zip',key='run_zip')
    st.caption('Downloads include only trials already revealed by playback. Finish instantly to export the complete run. Data remain in this browser session until downloaded; hosting does not create a permanent student database.')


@st.fragment(run_every=.25)
def playback():
    run=st.session_state.get('run')
    if not run:return
    meta=run['meta'];data=run['data'];R=int(data.trial.max())
    st.divider();st.subheader('Simulation workbench')
    if meta['section']!=st.session_state.page:
        st.caption(f'Showing the last generated run: {meta.get("kind",meta["section"])}. Generate on this page to replace it.')
    c1,c2,c3,c4,c5=st.columns(5)
    if c1.button('▶ Play',key='play'): run['playing']=True
    if c2.button('⏸ Pause',key='pause'): run['playing']=False
    if c3.button('⏭ One trial',key='step'):run['playing']=False;run['shown']=min(R,run['shown']+1);run['clock_fraction']=1.
    if c4.button('⏩ Finish instantly',key='finish'):run['shown']=R;run['playing']=False;run['clock_fraction']=1.
    if c5.button('↺ Replay',key='replay'):run['shown']=0;run['playing']=False;run['clock_fraction']=0.;run['phase']=0
    with st.expander('Jump to a completed trial (replay recorded outcomes)'):
        seek=number('Completed trials to reveal',0,0,R,key='seek')
        if st.button('Jump',key='jump'):
            run['shown']=seek;run['playing']=False;run['clock_fraction']=1.
    left,right=st.columns(2)
    with left: speed=st.select_slider('Playback speed · trials per tick',[1,2,5,10,25,50,100,500],value=5,key='speed')
    timed=meta.get('kind')=='Arrivals' or (meta.get('kind')=='Basketball' and any('time' in c.get('mode','') for c in meta['configs']))
    with right:
        time_mode=st.checkbox('Animate the simulated clock within each window',False,key='clock_mode',disabled=not timed)
        if timed and time_mode: clock_speed=st.slider('Simulated window fraction per tick',.01,1.,.1,.01,key='clock_speed')
    if run['playing'] and run['shown']<R:
        if timed and time_mode:
            run['clock_fraction']+=clock_speed
            if run['clock_fraction']>=1:
                run['shown']+=1;run['clock_fraction']=0.
        else: run['shown']=min(R,run['shown']+speed);run['clock_fraction']=1.
        run['phase']+=1
    if run['shown']>=R:run['playing']=False
    if speed>1: st.caption('Batch playback shows the most recent trial; every completed trial is recorded. Use one trial per tick or One trial to inspect individual outcomes.')
    st.progress(run['shown']/R,text=f'{run["shown"]:,} / {R:,} completed trials per population'+(' · playing' if run['playing'] else ' · paused'))
    if meta.get('hidden'):st.checkbox('Reveal true parameters and theory',False,key='reveal')
    show_truth=not meta.get('hidden',False) or st.session_state.get('reveal',False)
    if show_truth and meta.get('hidden'):st.json(meta['configs'],expanded=False)
    visible=data[data.trial<=run['shown']]
    if meta['section']=='Experiments':
        cols=st.columns(len(meta['configs']))
        for j,col in enumerate(cols):
            pop=chr(65+j)
            with col:
                st.markdown(f'**Population {pop}**')
                if timed and time_mode and run['shown']<R:
                    row=data[(data.population==pop)&(data.trial==run['shown']+1)].iloc[0]
                    local_timed=meta['kind']=='Arrivals' or 'time' in meta['configs'][j].get('mode','')
                    scene(row,meta['kind'],meta['configs'][j],run['phase'],run['clock_fraction'] if local_timed else 1.)
                elif not visible.empty:
                    scene(visible[visible.population==pop].iloc[-1],meta['kind'],meta['configs'][j],run['phase'])
                else: st.info('Press Play or One trial to begin.')
        if meta['kind']=='Darts' and not visible.empty:
            fig=px.scatter(visible.tail(4000),x='x',y='y',color='population',color_discrete_map=COLORS,opacity=.45)
            for j,c in enumerate(meta['configs']):fig.add_shape(type='circle',x0=-c['radius'],y0=-c['radius'],x1=c['radius'],y1=c['radius'],line_color=COLORS[chr(65+j)])
            fig.update_yaxes(scaleanchor='x',scaleratio=1)
            st.plotly_chart(figure_layout(fig,'Accumulated dart hits',350),use_container_width=True,key='all_darts')
        if meta['kind']=='Lifetimes' and not visible.empty:
            fig=go.Figure()
            for pop,g in visible.groupby('population'):
                x=np.sort(g.value);fig.add_trace(go.Scatter(x=np.r_[0,x],y=np.r_[1,1-np.arange(1,len(x)+1)/len(x)],name=pop,line=dict(color=COLORS[pop],shape='hv')))
            st.plotly_chart(figure_layout(fig,'Empirical survival probability',260),use_container_width=True,key='survival')
        hist_and_summary(visible,meta,show_truth)
        render_practical_clt(visible,prefix='experiment_clt',label=meta['kind']+' recorded outcome',default_enabled=True)
    elif meta['section'] in ['CLT & sampling','Distribution explorer']:
        clt_view(visible,meta,show_truth)
        hist_and_summary(visible,meta,False)
    elif meta['section']=='Resampling':
        if not visible.empty:
            sample=np.array(json.loads(visible.iloc[-1].details)['draws'])
            st.plotly_chart(figure_layout(go.Figure(go.Scatter(x=sample,y=np.zeros(len(sample)),mode='markers')),'Current resampled group A',180),use_container_width=True,key='resample_scene')
            hist_and_summary(visible,meta,False)
            if meta['method']=='Bootstrap':
                lo,hi=np.quantile(visible.value,[.025,.975]);st.write(f'Percentile bootstrap 95% interval: [{lo:.4g}, {hi:.4g}] (stabilizes with more resamples).')
            else:
                p=(1+(abs(visible.value)>=abs(meta['observed'])).sum())/(len(visible)+1)
                st.write(f'Two-sided Monte Carlo permutation p-value: {p:.4g}; observed mean difference: {meta["observed"]:.4g}')

    if meta['section']=='Distribution explorer' and not visible.empty:
        with st.expander('Empirical CDF and normal Q–Q diagnostic'):
            ecdf=go.Figure();qq=go.Figure()
            for pop,g in visible.groupby('population'):
                x=np.sort(g.value.to_numpy());ecdf.add_trace(go.Scatter(x=x,y=np.arange(1,len(x)+1)/len(x),name=pop,line=dict(color=COLORS[pop],shape='hv')))
                if len(x)>2:
                    (theory,ordered),(slope,intercept,r)=stats.probplot(x,dist='norm')
                    qq.add_trace(go.Scatter(x=theory,y=ordered,mode='markers',name=pop,marker=dict(color=COLORS[pop],size=4)))
                    qq.add_trace(go.Scatter(x=theory,y=intercept+slope*theory,mode='lines',name=pop+' reference',line=dict(color=COLORS[pop],dash='dash')))
            st.plotly_chart(figure_layout(ecdf,'Empirical cumulative distribution'),use_container_width=True,key='ecdf')
            st.plotly_chart(figure_layout(qq,'Normal Q–Q plot · curvature indicates departure from normality'),use_container_width=True,key='qq')
    if not visible.empty:
        with st.expander('Recorded results and downloads',expanded=False):
            st.dataframe(visible.tail(1000),hide_index=True,use_container_width=True)
            downloads(visible,run)
    else:st.caption('Results appear here as trials finish.')

st.title('🎲 Applied Statistics Simulation Lab')
st.caption('Watch randomness. Collect evidence. Compare populations.')
with st.sidebar:
    page=st.radio('Laboratory',['Experiments','Distribution explorer','CLT & sampling','Confidence intervals','Testing & power','Resampling','Sampling bias','Regression'],key='page')
    st.divider()
    st.markdown('**Suggested workflow**\n\n1. Predict the outcome.\n2. Set populations and sample size.\n3. Run a small experiment.\n4. Increase the trials.\n5. Explain the evidence and download it.')
    st.caption('Playback is separate from simulated time. A/B populations use independent random-number streams.')
    if st.button('Clear current results',key='clear'):
        st.session_state.pop('run',None)
        st.session_state.pop('bb_run',None)
    if st.button('Clear all session data',key='clear_all'):
        for data_key in ['run','archive','ci','testing','bias','regression','waits','clt_compare','bb_run']:
            st.session_state.pop(data_key,None)
    st.caption('For classroom projection, use browser fullscreen and collapse this sidebar.')

if page=='Experiments':
    kind=st.selectbox('Choose an experiment',EXPERIMENTS+['Knicks vs. 76ers'])
    if kind=='Knicks vs. 76ers':
        render_basketball_experiment()
        st.stop()
    with st.expander('Settings · define the experiment and populations',expanded=True):
        configs,hidden=setup_populations(lambda k:experiment_controls(kind,kind+k))
        R,seed=common_trials()
        st.caption('Each trial is independent. For cards/urns, replacement controls draws within a trial; the population resets between trials.')
        action_generate(lambda:simulate(kind,configs,R,seed),dict(section=page,kind=kind,configs=configs,R=R,seed=seed,hidden=hidden))
    definitions={'Coins':'One observation = heads in the selected number of flips.',
        'Dice':'One observation = sum or maximum of the selected number of dice.',
        'Cards':'One observation = target cards in one independently dealt hand.',
        'Darts':'One observation = a target hit (1) or miss (0); x and y are also recorded.',
        'Basketball':'One observation = makes in a fixed trial/window, or attempts to reach target makes.',
        'Urn sampling':'One observation = red objects in one sample.',
        'Manufacturing':'One observation = actual defects per batch; inspection flags and acceptance are recorded separately.',
        'Arrivals':'One observation = arrivals in one time window; all event times are retained.',
        'Lifetimes':'One observation = one component lifetime.',
        'Measurement':'One observation = one instrument reading.',
        'Galton board':'One observation = final bin (number of right turns).',
        'Random walk':'One observation = endpoint after the selected number of steps.'}
    st.info(definitions[kind])
    if kind=='Arrivals':
        with st.expander('Companion experiment · exponential interarrival times'):
            st.caption('These are independent, uncensored waiting times, sampled directly. Gaps selected only from a finite observation window have boundary/selection effects.')
            if st.button('Generate independent waiting times'):
                frames=[]
                for j,c in enumerate(configs):
                    rng=np.random.default_rng(np.random.SeedSequence([seed,812,j]))
                    frames.append(pd.DataFrame({'population':chr(65+j),'waiting_time':rng.exponential(1/c['rate'],R),'rate':c['rate']}))
                st.session_state.waits=pd.concat(frames,ignore_index=True)
            if 'waits' in st.session_state:
                f=st.session_state.waits
                st.plotly_chart(figure_layout(px.histogram(f,x='waiting_time',color='population',color_discrete_map=COLORS,histnorm='probability density',barmode='overlay',opacity=.5),'Independent exponential waiting times'),use_container_width=True)
                st.download_button('Download waiting times CSV',f.to_csv(index=False),'interarrival_times.csv','text/csv')


elif page in ['Distribution explorer','CLT & sampling']:
    with st.expander('Settings · populations and repeated samples',expanded=True):
        configs,hidden=setup_populations(lambda k:dist_controls('dist'+k))
        n=number('Observations per sample n',1 if page=='Distribution explorer' else 30,1,1000,key=page+'n')
        statistic=st.selectbox('Statistic recorded for each sample',['Mean','Sum','Median','Variance','Proportion > threshold'])
        if statistic=='Variance' and n<2:st.warning('Sample variance requires n ≥ 2.');n=2
        if statistic=='Proportion > threshold':
            threshold=number('Threshold',0.,-10000.,10000.)
            for c in configs:c['threshold']=threshold
        standardize=st.checkbox('Standardize means / sums using population moments',False,disabled=statistic not in ['Mean','Sum'])
        R,seed=common_trials('dist')
        action_generate(lambda:sampling(configs,n,R,seed,statistic),dict(section=page,configs=configs,n=n,R=R,seed=seed,statistic=statistic,hidden=hidden,standardize=standardize))
    if page=='Distribution explorer':
        st.info('Set n=1 for ordinary random sampling. Increase n to explore sampling distributions. Custom discrete lets you construct a population by editing category weights.')
        if not hidden:
            with st.expander('Theoretical probabilities and moments',expanded=True):
                aa,bb=st.columns(2)
                with aa:lo=number('Interval lower bound a',-1.,-10000.,10000.)
                with bb:hi=number('Interval upper bound b',1.,-10000.,10000.)
                for j,c in enumerate(configs):
                    try:
                        mu,var=moments(c)
                        if c['distribution']=='Bimodal':
                            F=lambda x:c['p']*stats.norm.cdf(x,c['mu1'],c['sigma'])+(1-c['p'])*stats.norm.cdf(x,c['mu2'],c['sigma'])
                            pr=F(hi)-F(lo)
                        else:
                            law=distribution(c);pr=law.cdf(hi)-law.cdf(np.ceil(lo)-1 if is_discrete(c) else lo)
                        st.write(f'{chr(65+j)}: mean = {mu:.4g}; variance = {var:.4g}; P(a ≤ X ≤ b) = {max(0.,pr) if hi>=lo else 0.:.4g}')
                        demo=draw(c,5000,np.random.default_rng(123));l,h=np.quantile(demo,[.005,.995])
                        xx=np.arange(np.floor(l),np.ceil(h)+1) if is_discrete(c) else np.linspace(l,h,300)
                        fig=go.Figure(go.Scatter(x=xx,y=density(c,xx),line_color=COLORS[chr(65+j)],name='PMF / density'))
                        yy=density(c,xx);fig.add_trace(go.Scatter(x=xx,y=np.where((xx>=lo)&(xx<=hi),yy,0),fill='tozeroy',name='Selected interval',line_color=COLORS[chr(65+j)]))
                        st.plotly_chart(figure_layout(fig,f'Population {chr(65+j)} · theoretical distribution',230),use_container_width=True,key='theory'+str(j))
                    except ValueError as exc:st.error(str(exc))
    else:
        st.latex(r'\operatorname{SD}(\bar X)=\sigma/\sqrt n,\qquad \operatorname{SD}(S_n)=\sigma\sqrt n')
        st.caption('The usual CLT assumes independent, identically distributed observations and finite, nonzero variance. It gives approximate normality as n grows, not a guarantee at n=30.')
        with st.expander('Compare several sample sizes side by side'):
            sizes_text=st.text_input('Sample sizes','1,5,30,100')
            if st.button('Compare sample sizes'):
                try:
                    sizes=[int(s) for s in sizes_text.split(',')]
                    if not 1<=len(sizes)<=6 or any(n<1 or n>1000 for n in sizes):raise ValueError('Use 1–6 sizes between 1 and 1000.')
                    if statistic=='Variance' and min(sizes)<2:raise ValueError('Sample variance requires every size to be at least 2.')
                    if sum(sizes)*min(R,2000)>2_000_000:raise ValueError('Reduce the sample sizes or repetitions.')
                    frames=[]
                    for size in sizes:
                        f=sampling(configs[:1],size,min(R,2000),seed,statistic);f['sample_size']=str(size);frames.append(f)
                    st.session_state.clt_compare=pd.concat(frames,ignore_index=True)
                except ValueError as exc:st.error(str(exc))
            if 'clt_compare' in st.session_state:
                f=st.session_state.clt_compare
                fig=px.histogram(f,x='value',facet_col='sample_size',facet_col_wrap=3,histnorm='probability density',nbins=35)
                st.plotly_chart(figure_layout(fig,'Population A · repeated sample statistics',500),use_container_width=True)
                st.download_button('Download sample-size comparison CSV',f.to_csv(index=False),'clt_sample_sizes.csv','text/csv')

elif page=='Confidence intervals':
    st.subheader('Repeated confidence intervals')
    st.caption('Two-sided Student t intervals for a population mean. Exact coverage under normal sampling; approximate otherwise. Changing the population can expose poor coverage.')
    with st.expander('Settings',expanded=True):
        c=dist_controls('ci');n=number('Sample size n',30,2,1000);R,seed=common_trials('ci');level=st.slider('Confidence level',.5,.99,.95,.01)
        if st.button('Simulate confidence intervals',type='primary'):
            try:st.session_state.ci=(confidence_coverage(c,n,R,seed,level),{'distribution':c,'n':n,'R':R,'seed':seed,'level':level})
            except ValueError as exc:st.error(str(exc))
    if 'ci' in st.session_state:
        f,meta=st.session_state.ci
        shown=st.slider('Reveal completed intervals',1,len(f),min(100,len(f))) if len(f)>1 else 1
        g=f.iloc[:shown];view=g.tail(200)
        fig=go.Figure()
        for cover,color in [(True,'#0d9488'),(False,'#ef4444')]:
            h=view[view.covers==cover];xs=[];ys=[]
            for row in h.itertuples():xs.extend([row.lower,row.upper,None]);ys.extend([row.trial,row.trial,None])
            fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',name='Covers μ' if cover else 'Misses μ',line_color=color))
        fig.add_vline(x=f.true_mean.iloc[0],line_dash='dash');fig.update_xaxes(title='Mean interval');fig.update_yaxes(title='Repeated sample')
        st.plotly_chart(figure_layout(fig,'Intervals across independent samples',500),use_container_width=True)
        st.metric('Observed coverage',f'{g.covers.mean():.1%}');st.caption('Plot shows the latest 200 revealed intervals. Download includes all revealed intervals.')
        st.download_button('Download confidence intervals CSV',g.to_csv(index=False),'confidence_intervals.csv','text/csv')
        st.download_button('Download CI settings JSON',json.dumps(meta,indent=2),'confidence_settings.json','application/json')

elif page=='Testing & power':
    st.subheader('Repeated studies · exact binomial test')
    a,b=st.columns(2)
    with a:p0=prob('Null success probability p₀','testp0',.5,True);p1=prob('True success probability p₁','testp1',.6)
    with b:n=number('Attempts per study',50,1,5000);alpha=st.slider('Significance level α',.01,.2,.05,.01)
    R,seed=common_trials('testing')
    if st.button('Simulate studies',type='primary'):
        st.session_state.testing=(testing(p0,p1,n,R,alpha,seed),{'n':n,'alpha':alpha,'p0':p0,'p1':p1,'R':R,'seed':seed})
    if 'testing' in st.session_state:
        f,meta=st.session_state.testing
        shown=st.slider('Reveal completed studies',1,len(f),min(100,len(f))) if len(f)>1 else 1;g=f.iloc[:shown]
        fig=px.scatter(g,x='trial',y='estimate',color='reject',color_discrete_map={True:'#e87926',False:'#0d9488'})
        fig.add_hline(y=meta['p0'],line_dash='dash')
        st.plotly_chart(figure_layout(fig,'Sample proportions and test decisions'),use_container_width=True)
        rate=g.reject.mean();under_null=meta['p0']==meta['p1']
        st.metric('Estimated Type I error' if under_null else 'Estimated power',f'{rate:.1%}')
        if not under_null:st.metric('Estimated Type II error',f'{1-rate:.1%}')
        st.caption('Exact two-sided binomial tests can be conservative because counts are discrete. Each dot represents an independent study.')
        st.download_button('Download studies CSV',g.to_csv(index=False),'hypothesis_tests.csv','text/csv')
        st.download_button('Download test settings JSON',json.dumps(meta,indent=2),'test_settings.json','application/json')

elif page=='Resampling':
    st.subheader('Bootstrap and randomization')
    method=st.selectbox('Method',['Bootstrap','Permutation test'])
    atext=st.text_area('Group A observations (comma-separated)','8,9,10,10,11,12,13,14')
    two=st.checkbox('Two groups',method=='Permutation test',disabled=method=='Permutation test')
    btext=st.text_area('Group B observations (comma-separated)','5,6,7,8,8,9,10,11') if two or method=='Permutation test' else None
    R,seed=common_trials('resample')
    if st.button('Generate resamples',type='primary'):
        try:
            aa=np.array([float(v) for v in atext.split(',')]);bb=None if btext is None else np.array([float(v) for v in btext.split(',')])
            if len(aa)<2 or not np.all(np.isfinite(aa)) or (bb is not None and (len(bb)<2 or not np.all(np.isfinite(bb)))):raise ValueError('Enter at least two finite numbers per group.')
            f=resample(aa,R,seed,bb,method)
            start_run(f,dict(section=page,method=method,seed=seed,R=R,observed=float(aa.mean()-(bb.mean() if bb is not None else 0)),original_a=aa.tolist(),original_b=None if bb is None else bb.tolist()))
        except ValueError as exc:st.error(str(exc))
    st.caption('Bootstrap samples with replacement within groups. Permutation pools observations and shuffles labels while preserving group sizes; its null model requires exchangeability.')

elif page=='Sampling bias':
    st.subheader('Large samples can still be biased')
    fraction=prob('Fraction of population in group 1','bias_frac',.3)
    advantage=st.slider('Relative selection weight for group 1',.1,10.,4.,.1)
    n=number('Sample size',100,1,10000);R,seed=common_trials('bias')
    if st.button('Compare sampling methods',type='primary'):
        try:
            if n*R>1_000_000:raise ValueError('Reduce sample size or repetitions.')
            rng=np.random.default_rng(seed);N=5000
            group=(rng.random(N)<fraction).astype(int);values=rng.normal(50+30*group,10)
            weights=np.where(group==1,advantage,1.);weights=weights/weights.sum()
            rows=[]
            for label,p in [('Random sample',None),('Biased selection',weights)]:
                for i in range(R):
                    idx=rng.choice(N,n,replace=True,p=p);rows.append({'method':label,'trial':i+1,'mean':values[idx].mean(),'true_mean':values.mean()})
            st.session_state.bias=(pd.DataFrame(rows),pd.DataFrame({'value':values,'group':group}),dict(n=n,R=R,seed=seed,fraction=fraction,selection_weight=advantage))
        except ValueError as exc:st.error(str(exc))
    if 'bias' in st.session_state:
        f,pop,meta=st.session_state.bias
        st.plotly_chart(figure_layout(px.histogram(pop,x='value',color='group',nbins=50),'The heterogeneous population'),use_container_width=True)
        fig=px.histogram(f,x='mean',color='method',barmode='overlay',opacity=.55,histnorm='probability density');fig.add_vline(x=f.true_mean.iloc[0],line_dash='dash')
        st.plotly_chart(figure_layout(fig,'Repeated sample means · dashed line is the population mean'),use_container_width=True)
        st.dataframe(f.groupby('method').agg(mean_estimate=('mean','mean'),sd=('mean','std')))
        st.caption('Sampling is with replacement from the same finite population. Selection weights alter who is observed; a larger n reduces spread without fixing this selection bias.')
        st.download_button('Download sampling-bias results CSV',f.to_csv(index=False),'sampling_bias.csv','text/csv')
        st.download_button('Download finite population CSV',pop.to_csv(index=False),'sampling_population.csv','text/csv')
        st.download_button('Download sampling settings JSON',json.dumps(meta,indent=2),'sampling_settings.json','application/json')

elif page=='Regression':
    st.subheader('Correlation, noise, and regression')
    a,b=st.columns(2)
    with a:slope=number('True slope',1.,-10.,10.);intercept=number('True intercept',0.,-100.,100.)
    with b:noise=number('Noise SD',2.,.01,100.);n=number('Number of paired observations',100,3,10000)
    outlier=st.checkbox('Add an influential outlier');seed=number('Random seed',42,0,2**32-1)
    if st.button('Generate paired observations',type='primary'):
        rng=np.random.default_rng(seed);x=rng.uniform(0,10,n);y=intercept+slope*x+rng.normal(0,noise,n)
        if outlier:x=np.r_[x,20];y=np.r_[y,intercept-20*slope-20]
        st.session_state.regression=(pd.DataFrame({'x':x,'y':y}),dict(slope=slope,intercept=intercept,noise=noise,n=n,outlier=outlier,seed=seed))
    if 'regression' in st.session_state:
        f,meta=st.session_state.regression
        shown=st.slider('Reveal observations',3,len(f),min(30,len(f)));g=f.iloc[:shown]
        fit=stats.linregress(g.x,g.y);xx=np.linspace(g.x.min(),g.x.max(),100)
        fig=px.scatter(g,x='x',y='y');fig.add_trace(go.Scatter(x=xx,y=fit.intercept+fit.slope*xx,name='Fitted line'))
        fig.add_trace(go.Scatter(x=xx,y=meta['intercept']+meta['slope']*xx,name='Generating line',line_dash='dash'))
        st.plotly_chart(figure_layout(fig,'Paired observations and fitted relationship',400),use_container_width=True)
        st.write(f'Slope: {fit.slope:.4g} · correlation r: {fit.rvalue:.4g} · R²: {fit.rvalue**2:.4g}')
        residuals=g.y-fit.intercept-fit.slope*g.x
        st.plotly_chart(figure_layout(go.Figure(go.Scatter(x=g.x,y=residuals,mode='markers')),'Residuals versus x',250),use_container_width=True)
        st.download_button('Download paired observations CSV',g.assign(residual=residuals).to_csv(index=False),'regression.csv','text/csv')
        st.download_button('Download regression settings JSON',json.dumps(meta,indent=2),'regression_settings.json','application/json')

if page in ['Experiments','Distribution explorer','CLT & sampling','Resampling']:playback()

with st.expander('Saved runs in this session'):
    archive=st.session_state.get('archive',[])
    if archive:
        options=list(range(len(archive)))
        ix=st.selectbox('Run',options,format_func=lambda i:f'{i+1}. {archive[i]["meta"].get("kind",archive[i]["meta"]["section"])} · {archive[i]["meta"]["created_utc"][:19]}')
        saved=archive[ix]
        if st.button('Load saved run'):
            st.session_state.run={'data':saved['data'],'meta':saved['meta'],'shown':0,'playing':False,'phase':0,'clock_fraction':0.}
            st.rerun()
        st.caption('The latest five generated workbench runs are retained in this session. Download your data before closing the app.')
    else:st.caption('No generated runs yet.')
