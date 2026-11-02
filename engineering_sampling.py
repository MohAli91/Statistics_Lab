"""Reproducible engineering teaching populations. All values are synthetic.

Default CSVs are bundled for offline use. The UI can regenerate populations
from an independent seed and variation controls. No illustrative limit here
is a construction-code or product-acceptance requirement.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ENGINEERING_CASES=['shafts','production_lines','pallets','asphalt','concrete']
DEFAULT_GENERATION={'seed':2026,'between_scale':1.,'noise_scale':1.,'periodic':False,'thin_patch':True,'concrete_measure':'Strength'}
MODELS={
 'shafts':dict(title='Machined shafts on a conveyor (generated)',variable='Shaft diameter',units='mm',
  question='Estimate mean diameter across the production run. Compare a random-start systematic inspection with random sampling, then turn on a periodic machining problem.',
  population='1,200 generated shafts in production order. A tool is replaced every 200 shafts; a small wear trend restarts after replacement. Each unit is one shaft.',
  layout='A wrapped production-order heatmap: 50 consecutive shafts per displayed row, not 24 separate physical lines. Tool runs are real groups within this generated model; cartons contain 50 shafts.',
  stratum_label='Tool run',cluster_label='Carton',visual='conveyor',default_method='Systematic (one random start)',
  recommended='Systematic random-start inspection for routine coverage; compare with SRS when periodic faults are present',
  source_url='https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm',
  x_title='Position within displayed 50-shaft segment',y_title='Consecutive production segment',
  lower_limit=19.95,upper_limit=20.05,nominal=20.,
  mechanism='Diameter = 20 mm + centered linear wear within each 200-shaft tool run + small run offset + independent normal measurement/process noise. Optional period-20 oscillation illustrates systematic aliasing.'),
 'production_lines':dict(title='Parts from five production lines (generated)',variable='Part diameter',units='mm',
  question='Estimate the plant-wide mean across unequal production volumes. Compare proportional and equal allocation, and explain why equal allocation needs population weights.',
  population='1,380 generated parts: line volumes 480, 360, 240, 180 and 120. Each unit is one part; line means and process SDs differ.',
  layout='Five horizontal production lines. Colored cells are individual parts in within-line production order. Empty cells beyond shorter line lengths are not units. Lots contain up to 30 parts.',
  stratum_label='Production line',cluster_label='Production lot',visual='production_lines',default_method='Stratified (proportional allocation)',
  recommended='Stratified random sampling by production line, with proportional allocation and known line-volume weights',
  source_url='https://www150.statcan.gc.ca/n1/edu/power-pouvoir/ch13/prob/5214899-eng.htm',
  x_title='Part position along line',y_title='Production line',lower_limit=19.95,upper_limit=20.05,nominal=20.,
  mechanism='Five normal line processes have mean offsets 0, +0.025, −0.020, +0.040 and −0.035 mm, SDs 0.010, 0.020, 0.015, 0.030 and 0.012 mm, and a small within-line drift.'),
 'pallets':dict(title='Component inspection in warehouse pallets (generated)',variable='Component resistance',units='ohm',
  question='Estimate mean component resistance. Compare inspecting more pallets with inspecting more components per pallet at a similar total measurement cost.',
  population='36 generated pallets from three suppliers. Pallet sizes cycle through 24, 30, 36 and 42 components; there are 1,188 components in total. Each unit is one component.',
  layout='Warehouse rack view with a 6×6 arrangement of pallets. Pallet detail shows components in schematic carton positions. Shared pallet offsets create similarity within a pallet.',
  stratum_label='Supplier',cluster_label='Pallet',visual='pallets',default_method='Two-stage (groups then units)',
  recommended='Two-stage sampling: randomly select pallets, then randomly select components inside each pallet',
  source_url='https://www150.statcan.gc.ca/n1/edu/power-pouvoir/ch13/prob/5214899-eng.htm',
  x_title='Warehouse position',y_title='Warehouse rack row',lower_limit=95.,upper_limit=105.,nominal=100.,
  mechanism='Resistance = 100 ohm + supplier offset (−2, 0, +2) + a shared normal pallet offset (SD 2.5 ohm) + independent component noise (SD 1.2 ohm). Unequal pallet sizes require the design weights.'),
 'asphalt':dict(title='Asphalt pavement thickness (generated)',variable='Pavement thickness',units='mm',
  question='Estimate mean pavement thickness while spreading cores across the paving work. Compare mean estimation with the chance of detecting a localized thin patch.',
  population='720 generated potential core locations on a 1,200 m × 7.2 m two-lane pavement. Six 200 m sublots are strata; twelve 100 m segments are clusters.',
  layout='Plan-view road heatmap using distance and transverse offset. Lane markings and sublot boundaries show placement groups. Values are hypothetical thickness at candidate locations, not real cores.',
  stratum_label='Paving sublot',cluster_label='100 m segment',visual='asphalt',default_method='Stratified (proportional allocation)',
  recommended='Stratified random core locations within every paving sublot',
  source_url='https://www.fhwa.dot.gov/pavement/materials/hif24071.pdf',
  x_title='Distance along pavement (m)',y_title='Transverse offset (m)',lower_limit=75.,upper_limit=None,nominal=80.,
  mechanism='Thickness = 80 mm + a smooth longitudinal wave + lane offsets ±1.5 mm + independent noise (SD 2 mm). An optional localized Gaussian-shaped thin patch is centered near station 850 m and offset 5.4 m.'),
 'concrete':dict(title='Concrete delivery loads and slab placement (generated)',variable='28-day load-average specimen strength',units='MPa',
  question='Estimate average strength across delivery loads. Distinguish sampling more truckloads from making more specimens from the same truckload.',
  population='80 generated equal-volume concrete deliveries, each linked to one schematic slab zone. Every delivery has three correlated 28-day specimen results; the load average is the sampling outcome, so N=80 loads, not 240 specimens.',
  layout='Concrete truck sequence linked to an 8×10 slab-placement map. Adjacent delivery groups share a batching effect. Three cylinders are nested within each delivery; slab positions are schematic.',
  stratum_label='Concrete supplier',cluster_label='Four-load batch',visual='concrete',default_method='Simple random (without replacement)',
  recommended='Simple random sampling of delivery loads; specimens from one load remain nested observations',
  source_url='https://www.fhwa.dot.gov/construction/reviews/revconc2.cfm',
  x_title='Slab placement column',y_title='Slab placement row',lower_limit=30.,upper_limit=None,nominal=35.,
  mechanism='Load strength = 36 MPa + supplier offset (0, −2, +2) + shared four-load batch noise (SD 1.2 MPa) + load noise (SD 1.6 MPa) + a small delivery-order effect. Each of three specimens adds SD 0.75 MPa noise. Slump is generated separately and is not used as a strength predictor.')}


def generate_engineering_population(case,config=None):
    if case not in ENGINEERING_CASES:raise ValueError('Unknown generated engineering case.')
    c={**DEFAULT_GENERATION,**(config or {})};seed=int(c['seed']);b=float(c['between_scale']);w=float(c['noise_scale'])
    if not 0<=b<=2 or not .25<=w<=2:raise ValueError('Use between-group scale 0–2 and within-group scale 0.25–2.')
    rng=np.random.default_rng(seed);rows=[];meta=dict(MODELS[case])
    def add(value,stratum,cluster,x,y,x_label,y_label,**extra):
        i=len(rows);rows.append(dict(unit_id=f'{case}-{i+1:04d}',source_row=i+1,value=float(value),stratum=stratum,cluster=cluster,
                                    x=x,y=y,x_label=str(x_label),y_label=str(y_label),order=i,**extra))
    if case=='shafts':
        run_offsets=rng.normal(0,.003,6)
        for i in range(1200):
            run=i//200;wear=.02*((i%200)/199-.5)
            periodic=.035*np.sin(2*np.pi*i/20) if c['periodic'] else 0.
            value=20+b*(wear+run_offsets[run]+periodic)+rng.normal(0,.012*w)
            add(value,f'Tool run {run+1:02d}',f'Carton {i//50+1:02d}',i%50,i//50,i%50+1,f'Segment {i//50+1:02d}',
                shaft_number=i+1,tool_run=run+1,tool_age_shafts=i%200,periodic_component_mm=b*periodic)
    elif case=='production_lines':
        for line,(N,offset,sd) in enumerate(zip([480,360,240,180,120],[0,.025,-.020,.040,-.035],[.010,.020,.015,.030,.012]),1):
            for i in range(N):
                value=20+b*(offset+.006*(i/max(1,N-1)-.5))+rng.normal(0,sd*w)
                add(value,f'Line {line}',f'Line {line} lot {i//30+1:02d}',i,line-1,i+1,f'Line {line}',
                    line=line,part_number_in_line=i+1,line_population=N,nominal_line_mean_mm=20+b*offset)
    elif case=='pallets':
        for pallet in range(36):
            supplier=pallet%3;N=[24,30,36,42][pallet%4];shared=b*([-2.,0.,2.][supplier]+rng.normal(0,2.5))
            for j in range(N):
                value=100+shared+rng.normal(0,1.2*w)
                x=(pallet%6)*8+j%6;y=(pallet//6)*9+j//6
                add(value,f'Supplier {"ABC"[supplier]}',f'Pallet {pallet+1:02d}',x,y,x,y,
                    pallet_number=pallet+1,component_in_pallet=j+1,pallet_size=N,rack_column=pallet%6,rack_row=pallet//6,
                    component_column=j%6,component_row=j//6)
    elif case=='asphalt':
        for station in range(60):
            x=10+20*station
            for cross in range(12):
                y=.3+.6*cross;lane=1 if y<3.6 else 2
                patch=-12*np.exp(-.5*((x-850)/55)**2-.5*((y-5.4)/.65)**2) if c['thin_patch'] else 0.
                value=80+b*(4*np.sin(2*np.pi*x/1200)+(1.5 if lane==1 else -1.5)+patch)+rng.normal(0,2*w)
                add(value,f'Sublot {int(x//200)+1}',f'Segment {int(x//100)+1:02d}',x,round(y,2),x,round(y,2),
                    station_m=x,offset_m=round(y,2),lane=lane,thin_patch_component_mm=b*patch)
    else:
        suppliers=np.repeat(['A','B','C'],[40,24,16]);rng.shuffle(suppliers)
        batch_effects=rng.normal(0,1.2,20)
        for i,supplier in enumerate(suppliers):
            offset={'A':0.,'B':-2.,'C':2.}[supplier]
            latent=36+b*(offset+batch_effects[i//4]-.01*i)+rng.normal(0,1.6*w)
            cylinders=np.maximum(5.,latent+rng.normal(0,.75*w,3))
            slump=max(20.,110+b*{'A':0.,'B':10.,'C':-10.}[supplier]+rng.normal(0,12*w))
            value=float(cylinders.mean()) if c['concrete_measure']=='Strength' else float(slump)
            add(value,f'Supplier {supplier}',f'Batch {i//4+1:02d}',i%10,i//10,i%10+1,f'Placement row {i//10+1}',
                delivery_load=i+1,volume_m3=8.,specimen_1_MPa=float(cylinders[0]),specimen_2_MPa=float(cylinders[1]),
                specimen_3_MPa=float(cylinders[2]),load_mean_strength_MPa=float(cylinders.mean()),fresh_slump_mm=float(slump))
        if c['concrete_measure']=='Slump':
            meta.update(variable='Fresh-concrete slump per delivery',units='mm',lower_limit=75.,upper_limit=150.,nominal=110.,
                        question='Estimate mean fresh slump across delivery loads. The strength specimens remain nested auxiliary data, not additional delivery sampling units.')
    f=pd.DataFrame(rows);limits=(meta['lower_limit'],meta['upper_limit'])
    outside=np.zeros(len(f),bool)
    if limits[0] is not None:outside|=f.value<limits[0]
    if limits[1] is not None:outside|=f.value>limits[1]
    f['outside_illustrative_limits']=outside
    recipe={'case':case,'generator_version':1,'numpy_version':np.__version__,**c}
    csv=f.to_csv(index=False)
    meta.update(data_kind='generated',file=f'{case}.csv',population_size=len(f),collected='Generated teaching population',
                source='Generated engineering teaching data; context reference only, not measured field data.',
                download_url=meta['source_url'],license='Values are generated by this app; no third-party measurements are reproduced.',
                preprocessing=meta['mechanism']+' All numeric benchmarks and tolerances are illustrative, not acceptance specifications.',
                model_parameters=recipe,raw_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest(),
                csv_sha256=hashlib.sha256(csv.encode()).hexdigest())
    return f,meta


def write_default_populations(directory=None):
    root=Path(directory) if directory is not None else Path(__file__).parent/'sampling_data'
    catalog=json.loads((root/'sources.json').read_text())
    for case in ENGINEERING_CASES:
        f,meta=generate_engineering_population(case);f.to_csv(root/meta['file'],index=False);catalog[case]=meta
    (root/'sources.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False))
    return catalog


if __name__=='__main__':
    result=write_default_populations()
    print({k:result[k]['population_size'] for k in ENGINEERING_CASES})
