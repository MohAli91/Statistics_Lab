"""Sampling Methods section with five real-data finite-population scenarios."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sampling_engine import (METHODS,SRS,STRAT_PROP,STRAT_EQUAL,SYSTEMATIC,CLUSTER,TWO_STAGE,CONVENIENCE,
                             source_catalog,load_population,collect_sample,repeat_samples,group_variation)
from sampling_export import sampling_workbook

@st.cache_data(show_spinner=False)
def cached_population(case):return load_population(case)


@st.cache_data(max_entries=4,ttl=600,show_spinner=False)
def cached_workbook(pop,meta,config,sample=None,summary=None,repeated=None,draws=None,parameters=None):
    return sampling_workbook(pop,meta,config,sample,summary,repeated,draws,parameters)


def layout(fig,title,height=380):
    fig.update_layout(title=title,height=height,margin=dict(l=30,r=20,t=45,b=35),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',font_color='#12263a',
                      legend=dict(orientation='h',y=1.12))
    return fig


def population_plot(pop,meta,sample=None):
    selected=pop.iloc[:0] if sample is None else sample
    if meta['visual']=='map':
        f=go.Figure(go.Scattergl(x=pop.x,y=pop.y,mode='markers',
            customdata=pop[['unit_id','value','stratum','cluster']],
            hovertemplate='%{customdata[0]}<br>Value: %{customdata[1]:.2f}<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>',
            marker=dict(color=pop.value,colorscale='Viridis',size=7,opacity=.8,colorbar=dict(title=meta['units'])),name='Population'))
        if len(selected):f.add_trace(go.Scattergl(x=selected.x,y=selected.y,mode='markers',name='Collected sample',
            marker=dict(size=12,color='#f43f5e',symbol='circle-open',line=dict(width=2)),text=selected.unit_id))
    else:
        grid=pop.pivot(index='y',columns='x',values='value').sort_index()
        f=go.Figure(go.Heatmap(x=grid.columns,y=grid.index,z=grid.values,colorscale='Viridis',
            colorbar=dict(title=meta['units']),hovertemplate='Slot %{x}<br>Row %{y}<br>Value %{z:.2f}<extra></extra>',hoverongaps=False))
        if len(selected):f.add_trace(go.Scatter(x=selected.x,y=selected.y,mode='markers',name='Collected sample',
            marker=dict(size=10 if meta['visual']=='calendar' else 24,color='#f43f5e',symbol='square-open',line=dict(width=2)),
            customdata=selected[['unit_id','value']],hovertemplate='%{customdata[0]}<br>%{customdata[1]:.2f}<extra></extra>'))
        ys=pop[['y','y_label']].drop_duplicates().sort_values('y')
        step=max(1,int(np.ceil(len(ys)/16)));show=ys.iloc[::step]
        f.update_yaxes(tickvals=show.y,ticktext=show.y_label,autorange='reversed')
        xs=pop[['x','x_label']].drop_duplicates().sort_values('x')
        step=max(1,int(np.ceil(len(xs)/16)));show=xs.iloc[::step]
        f.update_xaxes(tickvals=show.x,ticktext=show.x_label,tickangle=-40 if meta['visual']=='field' else 0)
    f.update_xaxes(title=meta['x_title']);f.update_yaxes(title=meta['y_title'])
    return layout(f,'Population values and collected units',460)


def inference_controls(prefix,truth):
    a,b,c=st.columns(3)
    with a:level=st.select_slider('Confidence level',[.80,.90,.95,.99],value=.95,key=prefix+'_level')
    with b:alpha=st.select_slider('Test significance α',[.01,.05,.10],value=.05,key=prefix+'_alpha')
    with c:alternative=st.selectbox('Alternative hypothesis',['Two-sided','Greater','Less'],key=prefix+'_alternative')
    null_truth=st.checkbox('Set H₀ mean equal to the known population mean (study Type I error)',True,key=prefix+'_null_true')
    mu0=st.number_input('Null population mean μ₀',value=float(truth),disabled=null_truth,key=prefix+('_mu0_truth' if null_truth else '_mu0_manual'),format='%.4f')
    if null_truth:mu0=float(truth)
    st.caption('Confidence intervals are two-sided. Hypothesis-test tails follow the selected alternative; confidence level and significance level are separate settings.')
    return dict(confidence=level,mu0=mu0,alpha=alpha,alternative=alternative)


def show_single_summary(run):
    r=run['summary'];valid=bool(r.design_inference_available)
    st.markdown('**Completed sample**')
    columns=st.columns(4)
    columns[0].metric('Collected units',int(r.n));columns[1].metric('Unweighted sample mean X̄',f'{r.sample_mean:.3f}')
    columns[2].metric('Design estimate of population mean',f'{r.estimate:.3f}')
    columns[3].metric('Known finite-population mean',f'{r.true_mean:.3f}')
    if valid:
        st.write(f'Approximate {r.confidence:.0%} design-aware interval: **[{r.lower:.3f}, {r.upper:.3f}]**. Standard error: **{r.standard_error:.3f}**.')
        st.write(f'H₀: μ = {r.null_mean:.3f}; {r.alternative.lower()} p-value = **{r.p_value:.4g}** → '+('reject H₀' if r.reject else 'do not reject H₀')+f' at α={r.alpha:.2f}.')
        if r.n==r.population_size:st.caption('This is a census of the finite frame: sampling error is zero.')
        if run['config']['method'] in [STRAT_PROP,STRAT_EQUAL,CLUSTER,TWO_STAGE]:st.caption(r.analysis_note)
    else:
        st.warning(r.analysis_note)
        if st.checkbox('Show a naive SRS working interval/test for comparison',False,key='sampling_single_naive'):
            st.write(f'Working interval: [{r.working_lower:.3f}, {r.working_upper:.3f}]; p={r.working_p_value:.4g}. These calculations assume SRS and do not establish valid inference for this design.')
    if run['config']['method'] in [CLUSTER,TWO_STAGE]:
        st.caption('The design estimate uses estimated population total / known N. A sample-weighted ratio mean is a different estimator when sampled cluster sizes differ.')
    st.dataframe(run['sample'],hide_index=True,use_container_width=True)


@st.fragment(run_every=.45)
def population_workbench(pop,meta,case,signature):
    run=st.session_state.get('sampling_single')
    if run is not None and run['signature']!=signature:run=None
    selected=None
    if run is not None:
        a,b,c,d,e=st.columns(5)
        if a.button('▶ Animate collection',key='sampling_play'):run['playing']=True
        if b.button('⏸ Pause collection',key='sampling_pause'):run['playing']=False
        if c.button('One selected unit',key='sampling_step'):run['shown']=min(len(run['sample']),run['shown']+1);run['playing']=False
        if d.button('Reveal whole sample',key='sampling_finish'):run['shown']=len(run['sample']);run['playing']=False
        if e.button('Replay collection',key='sampling_replay'):run['shown']=0;run['playing']=False
        speed=st.select_slider('Units revealed per tick',[1,2,5,10,25,50,100],value=5,key='sampling_speed')
        if run['playing']:run['shown']=min(len(run['sample']),run['shown']+speed)
        if run['shown']==len(run['sample']):run['playing']=False
        st.progress(run['shown']/len(run['sample']),text=f'{run["shown"]} / {len(run["sample"])} selected units collected')
        selected=run['sample'].iloc[:run['shown']]
    st.plotly_chart(population_plot(pop,meta,selected),use_container_width=True,key='sampling_population_plot')
    st.caption(meta['layout']+' Color represents recorded values; similar colors indicate similar values. Red outlines mark collected units.')
    if run is not None:
        if run['shown']==len(run['sample']):
            show_single_summary(run)
            if st.checkbox('Prepare sample Excel workbook',False,key='sampling_prepare_single_excel'):
                payload=cached_workbook(pop,meta,run['config'],sample=run['sample'],summary=run['summary'],parameters=run['parameters'])
                st.download_button('Download sample and population · Excel',payload,'sampling_sample.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key='sampling_single_xlsx')
        else:st.caption('Inference and Excel export become available after the entire planned sample is collected. Animation reveals a preselected sample; it does not choose convenient units as it runs.')


def repeated_view(pop,meta,signature):
    run=st.session_state.get('sampling_repeated')
    if run is None or run['signature']!=signature:return
    results=run['results'];st.subheader('Repeated samples, confidence intervals, and tests')
    shown=st.slider('Completed samples to inspect',1,len(results),len(results),key='sampling_repeat_reveal') if len(results)>1 else 1
    visible=results.iloc[:shown]
    statistic=st.radio('Sampling-distribution statistic',['Design estimate of population mean','Unweighted sample mean X̄'],horizontal=True,key='sampling_statistic')
    xcol='estimate' if statistic.startswith('Design') else 'sample_mean'
    valid=bool(visible.design_inference_available.iloc[0])
    choices=['Design-aware approximate intervals/tests','Naive SRS working intervals/tests'] if valid else ['Naive SRS working intervals/tests']
    analysis=st.selectbox('Analysis of repeated studies',choices,key='sampling_analysis')
    working=analysis.startswith('Naive');prefix='working_' if working else ''
    if working:st.warning('Working SRS intervals/tests ignore the actual design. Their observed coverage and rejection rates are diagnostics; normal-looking sample means do not establish valid population inference.')
    if run['config']['method'] in [CLUSTER,TWO_STAGE]:st.caption('The number of independently selected clusters matters for normal/t approximations. Many measured units within a few clusters do not provide the same information as many independent clusters.')
    low=prefix+'lower';high=prefix+'upper';covered=prefix+'covers';pvalue=prefix+'p_value';reject=prefix+'reject'
    truth=float(pop.value.mean());coverage=visible[covered].dropna().astype(bool).mean()
    rejects=visible[reject].dropna().astype(bool).mean()
    stats_columns=st.columns(5)
    stats_columns[0].metric('Repeated samples shown',len(visible))
    stats_columns[1].metric('Mean estimate',f'{visible[xcol].mean():.3f}')
    stats_columns[2].metric('Bias versus known mean',f'{visible[xcol].mean()-truth:+.3f}')
    stats_columns[3].metric('Observed CI coverage',f'{coverage:.1%}')
    stats_columns[4].metric('Rejection rate',f'{rejects:.1%}')
    if np.isclose(run['parameters']['mu0'],truth):st.caption('Here H₀ equals the known finite-population mean, so the rejection rate estimates Type I error.')
    else:st.caption('H₀ differs from the known finite-population mean. The rejection rate reflects detection under this scenario; for a one-sided test, also check the direction of the alternative.')
    a,b=st.columns(2)
    with a:
        fig=go.Figure(go.Histogram(x=pop.value,histnorm='probability density',name='Population X',marker_color='#0d9488',opacity=.7))
        fig.add_vline(x=truth,line_dash='dash');fig.update_xaxes(title=meta['variable']+' ('+meta['units']+')')
        st.plotly_chart(layout(fig,'Original finite-population distribution',310),use_container_width=True,key='sampling_population_hist')
    with b:
        values=visible[xcol];fig=go.Figure(go.Histogram(x=values,histnorm='probability density',name='Sample estimates',marker_color='#2563eb',opacity=.6))
        sd=values.std(ddof=0)
        if len(values)>=20 and sd>0:
            xx=np.linspace(values.mean()-4*sd,values.mean()+4*sd,250)
            fig.add_trace(go.Scatter(x=xx,y=stats.norm.pdf(xx,values.mean(),sd),name='Moment-matched normal reference',line_color='#ef4444'))
        fig.add_vline(x=truth,line_dash='dash');fig.update_xaxes(title=statistic+' ('+meta['units']+')')
        st.plotly_chart(layout(fig,'Sampling distribution across independent samples',310),use_container_width=True,key='sampling_means_hist')
    st.caption('The normal reference is descriptive and uses the repeated estimates’ mean and SD. It is not proof of normality or correct coverage. Each repetition redraws from the same complete finite frame, rather than resampling the first collected sample.')
    fig=go.Figure();tail=visible.dropna(subset=[low,high]).tail(200)
    for cover,color in [(True,'#0d9488'),(False,'#ef4444')]:
        h=tail[tail[covered].astype(bool)==cover];xx=[];yy=[]
        for row in h.to_dict('records'):xx.extend([row[low],row[high],None]);yy.extend([row['replicate'],row['replicate'],None])
        fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',name='Covers true mean' if cover else 'Misses true mean',line_color=color))
    fig.add_vline(x=truth,line_dash='dash');fig.update_xaxes(title='Confidence interval for population mean');fig.update_yaxes(title='Independent sample')
    st.plotly_chart(layout(fig,'Confidence-interval coverage · latest 200 shown',440),use_container_width=True,key='sampling_cis')
    a,b=st.columns(2)
    with a:
        fig=go.Figure(go.Scatter(x=visible.replicate,y=visible[covered].astype(float).expanding().mean(),mode='lines',name='Coverage'))
        fig.add_hline(y=run['parameters']['confidence'],line_dash='dash');fig.update_yaxes(range=[0,1],title='Cumulative coverage')
        st.plotly_chart(layout(fig,'Does coverage approach the requested level?',260),use_container_width=True,key='sampling_running_coverage')
    with b:
        fig=go.Figure(go.Histogram(x=visible[pvalue],xbins=dict(start=0,end=1,size=.05),marker_color='#7c3aed'))
        fig.add_vline(x=run['parameters']['alpha'],line_dash='dash');fig.update_xaxes(title='p-value',range=[0,1])
        st.plotly_chart(layout(fig,'Hypothesis-test p-values',260),use_container_width=True,key='sampling_pvalues')
    inspector=st.slider('Inspect a repeated sample',1,shown,shown,key='sampling_inspector') if shown>1 else 1
    selected=run['draws'][run['draws'].replicate==inspector]
    selected_pop=pop[pop.unit_id.isin(selected.unit_id)]
    with st.expander('Inspect this sample on the population and in the data table'):
        st.plotly_chart(population_plot(pop,meta,selected_pop),use_container_width=True,key='sampling_replicate_map')
        st.dataframe(selected,hide_index=True,use_container_width=True)
        st.dataframe(visible[visible.replicate==inspector],hide_index=True,use_container_width=True)
    with st.expander('Repeated-study results and downloads'):
        st.dataframe(visible.tail(1000),hide_index=True,use_container_width=True)
        st.caption('The table shows up to 1,000 recent revealed studies. Downloads contain every revealed study and its sampled units.')
        if st.checkbox('Prepare repeated-study Excel workbook',False,key='sampling_prepare_repeated_excel'):
            draws=run['draws'][run['draws'].replicate<=shown]
            payload=cached_workbook(pop,meta,run['config'],repeated=visible,draws=draws,parameters=run['parameters'])
            st.download_button('Download repeated samples, intervals and tests · Excel',payload,'sampling_repeated_studies.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key='sampling_repeated_xlsx')
        st.download_button('Download repeated-study summary · CSV',visible.to_csv(index=False).encode(),'sampling_repeated_summary.csv','text/csv',key='sampling_repeat_csv')


def render_sampling_lab():
    st.subheader('Sampling Methods · five real-data scenarios')
    catalog=source_catalog();case=st.selectbox('Sampling scenario',list(catalog),format_func=lambda k:catalog[k]['title'],key='sampling_case')
    pop,meta=cached_population(case);N=len(pop);H=pop.stratum.nunique();M=pop.cluster.nunique()
    st.markdown('**Case question:** '+meta['question']);st.caption(meta['population'])
    st.info('Suggested design: '+meta['recommended']+'. Try alternatives and compare precision, bias, coverage, and the number of units actually measured.')
    st.markdown(f'**Recorded variable:** {meta["variable"]} ({meta["units"]}). **Population frame:** {N:,} units, {H} strata, {M} clusters.')
    with st.expander('Sampling design and inference settings',expanded=True):
        method=st.selectbox('Sampling method',METHODS,key='sampling_method')
        config={'method':method}
        if method in [CLUSTER,TWO_STAGE]:
            config['clusters']=st.number_input('Number of groups to select m',2,M,min(4,M),key='sampling_m_'+case)
            st.caption('Whole-cluster sampling measures every unit in the selected groups, so actual sample size depends on which groups are selected.')
            if method==TWO_STAGE:config['per_cluster']=st.number_input('Units per selected group',2,int(pop.groupby('cluster').size().max()),min(5,int(pop.groupby('cluster').size().max())),key='sampling_within_'+case)
        else:
            minimum=int(np.minimum(pop.groupby('stratum').size(),2).sum()) if method in [STRAT_PROP,STRAT_EQUAL] else 2
            config['n']=st.number_input('Sample size n (individual units)',minimum,N,min(max(30,minimum),N),key='sampling_n_'+case+'_'+method)
        if method==SYSTEMATIC:
            config['order']=st.selectbox('Systematic frame order',['Source order','Shuffled frame','Grouped by stratum','Sorted by value'],key='sampling_order')
            config['order_seed']=902
            st.caption('A single random start uses interval N/n. Frame order can create aliasing. Sorting by value is an artificial teaching comparison, not a field protocol.')
        if method==CONVENIENCE:st.warning('This teaching design restricts access to the first 35% of the ordered records (expanded if n is larger), then randomly samples within that pool. It represents undercoverage, not a documented access mechanism in the source data.')
        inference=inference_controls('sampling_inference_'+case,pop.value.mean())
        a,b=st.columns(2)
        with a:seed=st.number_input('Sampling random seed',0,2**32-1,42,key='sampling_seed')
        with b:R=st.number_input('Number of independent samples R',1,3000,300,key='sampling_R')
        signature=hashlib.sha256(json.dumps([case,meta['csv_sha256'],config,inference],sort_keys=True).encode()).hexdigest()
        parameters={**inference,'seed':int(seed),'repetitions':int(R),'created_utc':datetime.now(timezone.utc).isoformat()}
        a,b=st.columns(2)
        if a.button('Collect one sample',type='primary',key='sampling_collect'):
            try:
                sample,summary=collect_sample(pop,config,int(seed),**inference)
                st.session_state.sampling_single=dict(case=case,signature=signature,config=config.copy(),sample=sample,summary=summary,parameters=parameters,shown=0,playing=False)
            except ValueError as exc:st.error(str(exc))
        if b.button('Run repeated sampling studies',key='sampling_repeat'):
            try:
                with st.spinner('Drawing independent samples and calculating design-specific inference…'):
                    results,draws=repeat_samples(pop,config,int(R),int(seed),**inference)
                st.session_state.sampling_repeated=dict(case=case,signature=signature,config=config.copy(),results=results,draws=draws,parameters=parameters)
            except ValueError as exc:st.error(str(exc))
        st.caption('Changed design/inference settings require a new draw. Existing results retain the settings used to create them. Sampling is without replacement inside a sample; independent repetitions may select the same unit again.')
    with st.expander('Estimator and uncertainty for this design'):
        if method==SRS:
            st.latex(r'\widehat{\mu}=\bar y,\qquad \widehat{\operatorname{Var}}(\bar y)=(1-n/N)s^2/n')
            st.caption('Simple random sampling without replacement uses the finite-population correction. As n approaches N, sampling uncertainty approaches zero.')
        elif method in [STRAT_PROP,STRAT_EQUAL]:
            st.latex(r'\widehat{\mu}=\sum_h W_h\bar y_h,\qquad \widehat{\operatorname{Var}}(\widehat{\mu})=\sum_h W_h^2(1-n_h/N_h)s_h^2/n_h')
            st.caption('W_h=N_h/N. Every stratum contributes; allocations and known population shares determine weights. The t approximation uses Satterthwaite degrees of freedom.')
        elif method==CLUSTER:
            st.latex(r'\widehat{\mu}=\frac{M}{mN}\sum_{g\in s}T_g,\qquad \widehat{\operatorname{Var}}(\widehat{\mu})=\frac{M^2(1-m/M)s_T^2}{mN^2}')
            st.caption('T_g is the full total in selected cluster g, and M is the population number of clusters. The approximate t interval uses m−1 degrees of freedom.')
        elif method==TWO_STAGE:
            st.latex(r'\widehat{\mu}=\frac{M}{mN}\sum_{g\in s}N_g\bar y_g')
            st.caption('The variance estimate includes between-cluster estimated-total variation and within-cluster SRS uncertainty, with both finite-population corrections. The approximate t interval uses m−1 degrees of freedom.')
        elif method==SYSTEMATIC:
            st.caption('The random start gives each unit equal first-order inclusion probability n/N. The sample mean is an unbiased design estimator, but one systematic start does not supply a generally unbiased sample-only variance estimator. Repeated-start experiments reveal the effect of ordering.')
        else:
            st.caption('Undercoverage prevents a generally valid design-based estimate of the full population. A bell-shaped histogram of repeated estimates can still be centered on the wrong target.')
    population_workbench(pop,meta,case,signature)
    with st.expander('Variation within and between groups',expanded=True):
        by=st.radio('Compare groups by',['stratum','cluster'],format_func=lambda x:meta['stratum_label'] if x=='stratum' else meta['cluster_label'],horizontal=True,key='sampling_group_by')
        groups,share=group_variation(pop,by)
        st.metric('Between-group share of total squared variation',f'{share:.1%}')
        st.caption('A descriptive ANOVA decomposition. Larger shares indicate greater separation of group means. Small within-group SDs indicate more homogeneous groups; this number alone does not choose the best sampling design.')
        display=groups.head(35)
        fig=px.bar(display,x=by,y='mean',error_y='std',color_discrete_sequence=['#0d9488'])
        fig.update_yaxes(title=meta['variable']);fig.update_xaxes(title=meta['stratum_label'] if by=='stratum' else meta['cluster_label'])
        st.plotly_chart(layout(fig,'Group means with within-group SD (not confidence intervals)',320),use_container_width=True)
        if len(groups)>35:st.caption('Chart shows the first 35 groups. The table and Excel workbook include every group.')
        st.dataframe(groups,hide_index=True,use_container_width=True)
    repeated_view(pop,meta,signature)
    with st.expander('Source data, definitions, and teaching assumptions'):
        st.markdown(f'**Source:** {meta["source"]}\n\n[Dataset documentation]({meta["source_url"]})\n\n**Preprocessing:** {meta["preprocessing"]}\n\n**Source terms:** {meta["license"]}')
        st.caption('All displayed measurements are retained from the cited data. Strata/cluster definitions, bounded teaching frames, and schematic layouts are documented above. Inference targets the bundled finite frame, not future observations, all households, or causal treatment effects.')
        st.dataframe(pop.head(200),hide_index=True,use_container_width=True)
        if st.checkbox('Prepare full population Excel workbook',False,key='sampling_prepare_population'):
            payload=cached_workbook(pop,meta,config,parameters=parameters)
            st.download_button('Download population and definitions · Excel',payload,'sampling_population.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key='sampling_population_xlsx')
