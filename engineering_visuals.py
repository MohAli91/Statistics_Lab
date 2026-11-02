"""Domain-specific population views for generated engineering cases."""
from __future__ import annotations
import html
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st
import streamlit.components.v1 as components


def _color(value,pop):
    lo=float(pop.value.min());hi=float(pop.value.max())
    return sample_colorscale('Viridis',[float((value-lo)/(hi-lo)) if hi>lo else .5])[0]


def scene_html(pop,meta,sample=None):
    case=meta['model_parameters']['case'];chosen=set() if sample is None else set(sample.unit_id)
    latest=None if sample is None or sample.empty else sample.iloc[-1]
    caption='Schematic scene · colors match this generated population'
    items=[]
    if case=='shafts':
        anchor=0 if latest is None else max(0,int(latest.order)-6)
        items=['<rect x="24" y="90" width="852" height="32" rx="14" fill="#334155"/>']
        for x in range(45,870,55):items.append(f'<g class="roller"><circle cx="{x}" cy="120" r="12" fill="#94a3b8" stroke="#334155" stroke-width="3"/></g>')
        for j in range(12):
            row=pop.iloc[(anchor+j)%len(pop)];x=45+j*69;stroke='#f43f5e' if row.unit_id in chosen else '#334155'
            items.append(f'<g><rect x="{x}" y="61" width="48" height="28" rx="9" fill="{_color(row.value,pop)}" stroke="{stroke}" stroke-width="3"/><ellipse cx="{x+7}" cy="75" rx="7" ry="14" fill="#ffffff55"/><text x="{x+24}" y="52" text-anchor="middle" font-size="11">#{int(row.shaft_number)}</text></g>')
        items.append('<rect x="421" y="14" width="90" height="26" rx="6" fill="#0f766e"/><text x="466" y="31" text-anchor="middle" fill="white" font-size="11">DIAMETER</text><path class="beam" d="M466 40 V90" stroke="#ef4444" stroke-width="3" stroke-dasharray="4 3"/>')
        caption='Conveyor inspection · each shaft is one sampling unit'
    elif case=='production_lines':
        for i,(line,g) in enumerate(pop.groupby('stratum',sort=False)):
            y=26+i*27;items.append(f'<text x="15" y="{y+7}" font-size="12">{html.escape(line)} · N={len(g)}</text><rect x="140" y="{y+5}" width="735" height="8" rx="4" fill="#64748b"/>')
            for j in range(15):
                row=g.iloc[int(j*(len(g)-1)/14)];x=150+j*47
                items.append(f'<rect x="{x}" y="{y-8}" width="23" height="14" rx="4" fill="{_color(row.value,pop)}" stroke="#334155"/>')
        caption='Five production lines · unequal volumes and different process distributions'
    elif case=='pallets':
        for pallet,g in pop.groupby('cluster',sort=False):
            row=g.iloc[0];x=24+int(row.rack_column)*145;y=13+int(row.rack_row)*24
            count=sum(uid in chosen for uid in g.unit_id)
            items.append(f'<rect x="{x}" y="{y}" width="126" height="19" rx="3" fill="{_color(g.value.mean(),pop)}" stroke="{"#f43f5e" if count else "#64748b"}" stroke-width="{3 if count else 1}"/><text x="{x+63}" y="{y+13}" text-anchor="middle" font-size="10" fill="white">{html.escape(pallet)} · {count}/{len(g)}</text>')
        caption='Warehouse racks · pallet labels show collected components / pallet size'
    elif case=='asphalt':
        items=['<rect x="20" y="47" width="860" height="85" fill="#334155"/><path d="M20 89.5 H880" stroke="white" stroke-width="2" stroke-dasharray="20 14"/>']
        for j in range(6):
            x=20+j*860/6;items.append(f'<path d="M{x} 45 V135" stroke="#e2e8f0" stroke-width="1"/><text x="{x+70}" y="30" font-size="12" text-anchor="middle">Sublot {j+1}</text>')
        if sample is not None:
            for row in sample.itertuples():
                x=20+row.station_m/1200*860;y=47+row.offset_m/7.2*85
                items.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#fb7185" stroke="white"/>')
        px=40 if latest is None else 20+latest.station_m/1200*800
        items.append(f'<g transform="translate({px},55)"><rect width="38" height="20" rx="3" fill="#eab308"/><rect x="10" y="-13" width="17" height="14" fill="#94a3b8" stroke="#334155"/><circle cx="8" cy="23" r="6" fill="#111827"/><circle cx="30" cy="23" r="6" fill="#111827"/></g>')
        caption='Road plan · six sublots, two lanes, and selected core locations'
    else:
        anchor=0 if latest is None else max(0,int(latest.order)-3)
        for j in range(8):
            row=pop.iloc[(anchor+j)%len(pop)];x=20+j*110;color=_color(row.value,pop)
            items.append(f'<g><rect x="{x+53}" y="59" width="29" height="24" rx="4" fill="#f59e0b"/><rect x="{x+56}" y="61" width="15" height="10" fill="#dbeafe"/><ellipse cx="{x+31}" cy="64" rx="27" ry="20" fill="{color}" stroke="{"#f43f5e" if row.unit_id in chosen else "#64748b"}" stroke-width="3"/><path d="M{x+12} 47 L{x+45} 79 M{x+20} 45 L{x+53} 72" stroke="#ffffffaa" stroke-width="3"/><circle cx="{x+18}" cy="87" r="8" fill="#334155"/><circle cx="{x+65}" cy="87" r="8" fill="#334155"/><text x="{x+41}" y="33" text-anchor="middle" font-size="11">Load {int(row.delivery_load)}</text>')
            for k in range(3):items.append(f'<rect x="{x+22+k*14}" y="110" width="10" height="22" rx="3" fill="#94a3b8" stroke="#334155"/>')
            items.append('</g>')
        caption='Delivery loads → placement zones · three specimens belong to each load'
    text='Ready to collect a sample' if latest is None else f'Latest collected: {latest.unit_id} · {latest.value:.3f} {meta["units"]}'
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:#f1f5f9;font-family:Arial,sans-serif;color:#12263a;border-radius:12px}}.caption{{padding:9px 14px;font-size:13px;font-weight:bold}}svg{{display:block;width:100%;max-height:160px}}.status{{padding:7px 14px;font-size:12px}}.beam{{animation:pulse 1s alternate infinite}}@keyframes pulse{{to{{opacity:.25}}}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important}}}}</style></head><body><div class="caption">{html.escape(caption)}</div><svg viewBox="0 0 900 155" role="img" aria-label="{html.escape(caption)}">{''.join(items)}</svg><div class="status">{html.escape(text)}</div></body></html>'''


def render_scene(pop,meta,sample=None):
    components.html(scene_html(pop,meta,sample),height=225,scrolling=False)


def engineering_population_plot(pop,meta,sample=None):
    case=meta['model_parameters']['case'];sample=pop.iloc[:0] if sample is None else sample
    if case=='pallets':
        g=pop.groupby('cluster',sort=False).agg(mean=('value','mean'),size=('value','size'),rack_column=('rack_column','first'),rack_row=('rack_row','first')).reset_index()
        z=np.full((6,6),np.nan);labels=np.empty((6,6),object)
        for row in g.itertuples():z[int(row.rack_row),int(row.rack_column)]=row.mean;labels[int(row.rack_row),int(row.rack_column)]=row.cluster
        fig=go.Figure(go.Heatmap(z=z,x=list(range(6)),y=list(range(6)),text=labels,colorscale='Viridis',zmin=float(pop.value.min()),zmax=float(pop.value.max()),
            colorbar=dict(title='Pallet mean (ohm)'),hovertemplate='%{text}<br>Population pallet mean: %{z:.2f} ohm<extra></extra>'))
        if len(sample):
            selected=sample.groupby('cluster').agg(count=('value','size'),x=('rack_column','first'),y=('rack_row','first')).reset_index()
            fig.add_trace(go.Scatter(x=selected.x,y=selected.y,mode='markers+text',text=selected['count'],textfont_color='white',
                name='Collected components',marker=dict(symbol='square-open',size=39,color='#f43f5e',line_width=3)))
        fig.update_xaxes(title='Pallet column',tickvals=list(range(6)),ticktext=list(range(1,7)))
        fig.update_yaxes(title='Warehouse rack row',tickvals=list(range(6)),ticktext=list(range(1,7)),autorange='reversed')
        title='Warehouse pallet means · outlined numbers are collected component counts'
    else:
        grid=pop.pivot(index='y',columns='x',values='value').sort_index()
        fig=go.Figure(go.Heatmap(x=grid.columns,y=grid.index,z=grid.values,colorscale='Viridis',colorbar=dict(title=meta['units']),
            hovertemplate='Position %{x}<br>Row/offset %{y}<br>Value %{z:.4f}<extra></extra>',hoverongaps=False))
        if len(sample):fig.add_trace(go.Scatter(x=sample.x,y=sample.y,mode='markers',name='Collected units',
            marker=dict(symbol='circle-open' if case=='asphalt' else 'square-open',size=12 if case in ['shafts','production_lines'] else 18,color='#f43f5e',line_width=2),
            customdata=sample[['unit_id','value']],hovertemplate='%{customdata[0]}<br>%{customdata[1]:.4f}<extra></extra>'))
        title={'shafts':'Shaft diameters in wrapped production order','production_lines':'Individual part diameters along five lines','asphalt':'Pavement thickness plan · dashed lines bound paving sublots','concrete':'Slab placement zones colored by delivery outcome'}[case]
        if case=='asphalt':
            fig.add_hline(y=3.6,line_color='white',line_width=3)
            for boundary in [200,400,600,800,1000]:fig.add_vline(x=boundary,line_dash='dash',line_color='white',line_width=1)
            fig.update_yaxes(range=[0,7.2])
        else:
            yy=pop[['y','y_label']].drop_duplicates().sort_values('y');step=max(1,len(yy)//12)
            fig.update_yaxes(tickvals=yy.y.iloc[::step],ticktext=yy.y_label.iloc[::step],autorange='reversed')
        fig.update_xaxes(title=meta['x_title']);fig.update_yaxes(title=meta['y_title'])
    fig.update_layout(title=title,height=400,margin=dict(l=35,r=25,t=50,b=45),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',font_color='#12263a')
    fig.update_xaxes(automargin=True);fig.update_yaxes(automargin=True)
    return fig


def engineering_detail_plot(pop,meta,sample=None,group=None):
    case=meta['model_parameters']['case'];sample=pop.iloc[:0] if sample is None else sample
    if case=='pallets':
        p=pop[pop.cluster==group];s=sample[sample.cluster==group]
        fig=go.Figure(go.Scatter(x=p.component_column,y=p.component_row,mode='markers+text',text=p.component_in_pallet,
            marker=dict(symbol='square',size=38,color=p.value,colorscale='Viridis',cmin=pop.value.min(),cmax=pop.value.max(),colorbar=dict(title='ohm')),
            textfont=dict(color='white',size=10),customdata=p[['unit_id','value']],hovertemplate='%{customdata[0]}<br>%{customdata[1]:.2f} ohm<extra></extra>'))
        if len(s):fig.add_trace(go.Scatter(x=s.component_column,y=s.component_row,mode='markers',name='Collected components',marker=dict(symbol='square-open',size=43,color='#f43f5e',line_width=3)))
        fig.update_yaxes(autorange='reversed',title='Component row');fig.update_xaxes(title='Component column')
        title=f'{group}: all components and those collected'
    else:
        if case=='asphalt':
            g=pop.groupby('station_m',sort=True).value.mean()
            fig=go.Figure(go.Scatter(x=g.index,y=g.values,mode='lines+markers',name='Mean across 12 offsets'))
            fig.update_xaxes(title='Distance along pavement (m)');title='Mean thickness at each station (averaged across road width)'
        else:
            fig=go.Figure(go.Scatter(x=pop.order+1,y=pop.value,mode='lines',name='Population outcomes',line=dict(width=1,color='#64748b')))
            if len(sample):fig.add_trace(go.Scatter(x=sample.order+1,y=sample.value,mode='markers',name='Collected',marker=dict(color='#f43f5e',size=7)))
            fig.update_xaxes(title='Production order' if case!='concrete' else 'Delivery load');title='Outcome variation along the production/delivery sequence'
        for limit in [meta.get('lower_limit'),meta.get('upper_limit')]:
            if limit is not None:fig.add_hline(y=limit,line_dash='dash',line_color='#ef4444')
        fig.update_yaxes(title=meta['variable']+' ('+meta['units']+')')
    fig.update_layout(title=title,height=300,margin=dict(l=25,r=25,t=45,b=40),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',font_color='#12263a')
    return fig


def render_details(pop,meta,sample=None):
    case=meta['model_parameters']['case'];sample=pop.iloc[:0] if sample is None else sample
    with st.expander('Engineering detail view',expanded=case in ['pallets','concrete']):
        if case=='pallets':
            group=st.selectbox('Pallet to inspect visually',list(pop.cluster.unique()),key='engineering_pallet_view')
            st.caption('This viewer does not choose the sampled pallets. The sampling design still selects them randomly.')
            st.plotly_chart(engineering_detail_plot(pop,meta,sample,group),use_container_width=True,key='engineering_detail_plot')
        else:st.plotly_chart(engineering_detail_plot(pop,meta,sample),use_container_width=True,key='engineering_detail_plot')
        if case=='concrete':
            default_load=1 if sample.empty else int(sample.iloc[-1].delivery_load)
            load=st.number_input('Delivery load to inspect specimens',1,len(pop),default_load,key='engineering_specimen_load')
            row=pop[pop.delivery_load==load].iloc[0]
            fig=go.Figure(go.Bar(x=['Specimen 1','Specimen 2','Specimen 3'],y=[row.specimen_1_MPa,row.specimen_2_MPa,row.specimen_3_MPa],marker_color='#0d9488'))
            fig.add_hline(y=row.load_mean_strength_MPa,line_dash='dash',annotation_text='Load-average strength')
            fig.update_layout(height=260,title=f'Load {load}: three nested specimen results',yaxis_title='28-day specimen strength (MPa)',margin=dict(l=25,r=25,t=50,b=35))
            st.plotly_chart(fig,use_container_width=True,key='engineering_specimens')
            st.caption('These three specimens share one delivery load. They do not turn that delivery into three independent sampled loads. Strengths are generated cylinder-test results, not measurements of every point in the placed slab.')
        st.caption('Dashed red limits, where shown, are illustrative teaching benchmarks. Estimating a satisfactory mean does not establish that all individual units or locations meet a limit.')
