"""Reusable X-versus-X-bar teaching panel for recorded experiments."""
from __future__ import annotations
import io
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from practical_clt import WITH_REPLACEMENT,DISJOINT,empirical_sample_means,long_samples

@st.cache_data(max_entries=8,ttl=600,show_spinner=False)
def cached_means(values,n,R,seed,method):
    return empirical_sample_means(values,n,R,seed,method)


def layout(fig,title,height=290):
    fig.update_layout(title=title,height=height,margin=dict(l=20,r=20,t=50,b=30),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',font_color='#12263a',
                      legend=dict(orientation='h',y=1.15))
    return fig


def render_practical_clt(frame,prefix='practical',label='Recorded outcome',colors=None,default_enabled=False):
    """Input columns: population, trial, value. No unrevealed outcomes allowed."""
    st.markdown('### Practical CLT · from X to X̄')
    enabled=st.checkbox('Show outcome and sample-mean distributions',default_enabled,key=prefix+'_enabled')
    if not enabled:return
    st.caption(f'X = {label}. Completed recorded outcomes supply the data for this panel; unfinished trials/games are excluded. As completed outcomes accumulate, the empirical source and its sample-mean distribution update.')
    if frame.empty or frame.groupby('population').size().min()<2:
        st.info('Complete at least two trials/games per selected population to begin. More completed outcomes give a better picture of the original distribution.')
        return
    colors=colors or {'A':'#0d9488','B':'#e87926'}
    method=st.selectbox('How to form repeated samples',[WITH_REPLACEMENT,DISJOINT],key=prefix+'_method')
    c1,c2,c3=st.columns(3)
    with c1:n=st.number_input('Observations per sample n',1,1000,10,key=prefix+'_n')
    with c2:R=st.number_input('Repeated samples R',1,10000,1000,key=prefix+'_R',disabled=method==DISJOINT)
    with c3:seed=st.number_input('Sample-mean random seed',0,2**32-1,123,key=prefix+'_seed',disabled=method==DISJOINT)
    overlay=st.checkbox('Overlay normal reference for X̄',True,key=prefix+'_normal')
    if method==WITH_REPLACEMENT:
        st.info('Each sample draws n outcomes WITH replacement from all completed recorded outcomes. This estimates a sampling distribution conditional on the empirical population (a bootstrap-style CLT demonstration). It does not change the original games or generate new games.')
    else:
        st.info('Consecutive, disjoint groups of n recorded outcomes each contribute one mean. No outcome is reused. R = floor(N/n); the incomplete last group is excluded. With independent game/trial outcomes, these groups represent separate samples from the experiment.')
    groups=list(frame.groupby('population',sort=False));computed=[]
    # Calculate once for each current dataset/settings combination, not on every animation tick.
    for pop,g in groups:
        g=g.sort_values('trial');x=g.value.to_numpy(dtype=float)
        try:table,samples,indices=cached_means(x,int(n),int(R),int(seed),method)
        except ValueError as exc:st.warning(f'{pop}: {exc}');continue
        computed.append((pop,g,x,table,samples,indices))
    if not computed:return
    count=min(len(item[3]) for item in computed)
    reveal=st.slider('Reveal repeated sample means',1,count,count,key=prefix+'_reveal') if count>1 else 1
    st.caption('One point in the sampling distribution represents one complete sample of n observations. Moving this slider reveals the selected samples; it does not add original observations.')
    all_tables=[];export_data=[]
    for j,(pop,g,x,table,samples,indices) in enumerate(computed):
        selected=table.iloc[:reveal];sample=samples[reveal-1];mu=x.mean();sigma=x.std(ddof=0);se=sigma/np.sqrt(n)
        color=colors.get(pop,['#0d9488','#e87926'][j%2]);st.markdown(f'**{pop} · {label}**')
        metrics=st.columns(4)
        metrics[0].metric('Completed outcomes N',len(x));metrics[1].metric('Source mean',f'{mu:.3f}')
        metrics[2].metric('Source SD',f'{sigma:.3f}');metrics[3].metric('Reference SD of X̄',f'{se:.3f}')
        left,middle,right=st.columns([1.15,1,1.25])
        with left:
            fig=go.Figure(go.Histogram(x=x,histnorm='probability density',marker_color=color,opacity=.7,name='Recorded X'))
            fig.add_vline(x=mu,line_dash='dash');fig.update_xaxes(title='Individual X');fig.update_yaxes(title='Density')
            st.plotly_chart(layout(fig,'1 · Distribution of X'),use_container_width=True,key=prefix+'_x_'+str(j))
        with middle:
            fig=go.Figure(go.Scatter(x=sample,y=np.zeros(len(sample)),mode='markers',marker=dict(color=color,size=9,opacity=.65),name='Sample observations'))
            fig.add_vline(x=sample.mean(),line_color='#ed174c',annotation_text='X̄')
            fig.update_yaxes(visible=False);fig.update_xaxes(title='Sample values')
            st.plotly_chart(layout(fig,f'2 · Sample {reveal}: n={n} → X̄={sample.mean():.3f}'),use_container_width=True,key=prefix+'_sample_'+str(j))
        with right:
            fig=go.Figure(go.Histogram(x=selected.sample_mean,histnorm='probability density',marker_color=color,opacity=.65,name='Sample means'))
            if overlay and se>0:
                xx=np.linspace(mu-4*se,mu+4*se,250)
                fig.add_trace(go.Scatter(x=xx,y=stats.norm.pdf(xx,mu,se),name='Normal reference',line_color='#ed174c'))
            fig.update_xaxes(title='Sample mean X̄');fig.update_yaxes(title='Density')
            st.plotly_chart(layout(fig,f'3 · Sampling distribution: {len(selected)} means'),use_container_width=True,key=prefix+'_xbar_'+str(j))
        st.caption(f'Observed mean of X̄: {selected.sample_mean.mean():.4g}; observed SD of X̄: {selected.sample_mean.std(ddof=0):.4g}. Normal reference uses the recorded population mean and SD/√n. Density axes are scaled independently; X and X̄ have the same measurement units.')
        if len(selected)<30:st.caption(f'Only {len(selected)} sample means are shown. Increase repeated resamples, or collect more original outcomes for disjoint grouping, to see a stable histogram.')
        if len(x)<30:st.warning('The recorded source population is small. Resampling more often smooths the conditional picture but does not replace collecting more original observations.')
        if sigma==0:st.warning('All recorded values are identical. Sample means are constant, so there is no nondegenerate normal approximation.')
        selected=selected.assign(population=str(pop),variable=label)
        all_tables.append(selected);export_data.append((pop,g,samples[:reveal],indices[:reveal]))
    st.latex(r'E[\bar X]=\mu,\qquad \operatorname{SD}(\bar X)=\sigma/\sqrt n')
    st.caption('The usual CLT concerns means of iid observations with finite, nonzero variance. It gives approximate normality as n increases. The original distribution of X does not become normal. Increasing n changes the distribution of means; increasing R improves the Monte Carlo picture. A running average over one sequence is a different object from a distribution of repeated sample means.')
    if st.checkbox('Prepare practical CLT CSV downloads',False,key=prefix+'_export'):
        result=pd.concat(all_tables,ignore_index=True)
        st.download_button('Download sample means · CSV',result.to_csv(index=False).encode(),'practical_sample_means.csv','text/csv',key=prefix+'_means_csv')
        raw=[];source=[]
        for pop,g,samples,indices in export_data:
            raw.append(long_samples(samples,indices,g.trial.to_numpy()).assign(population=str(pop),variable=label))
            source.append(g.assign(variable=label))
        settings={'variable':label,'method':method,'n':int(n),'requested_R':int(R),'revealed_R':int(reveal),'seed':int(seed),'source':'completed recorded outcomes only'}
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('sample_means.csv',result.to_csv(index=False))
            z.writestr('sample_draws.csv',pd.concat(raw,ignore_index=True).to_csv(index=False))
            z.writestr('source_outcomes.csv',pd.concat(source,ignore_index=True).to_csv(index=False))
            z.writestr('sampling_settings.json',json.dumps(settings,indent=2))
        st.download_button('Download means + draws + source outcomes · ZIP',b.getvalue(),'practical_clt_results.zip','application/zip',key=prefix+'_all_zip')
