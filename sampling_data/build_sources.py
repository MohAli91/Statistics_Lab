"""Rebuild offline teaching frames from documented public sources.
Run from project root: python sampling_data/build_sources.py --raw-dir /path/to/cache
Network is needed only when refreshing, never during normal app use.
"""
from pathlib import Path
import argparse,io,zipfile,tarfile,json,hashlib,urllib.request
import numpy as np
import pandas as pd

URLS={
 'housing.tgz':'https://ndownloader.figshare.com/files/5976036',
 'steel.zip':'https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip',
 'bike.zip':'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
 'air.zip':'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
 'oats.csv':'https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oats.csv'}

def main(raw_dir):
 root=Path(__file__).resolve().parent;raw_dir=Path(raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
 for name,url in URLS.items():
  p=raw_dir/name
  if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=60).read())
 meta={}
 def save(key,f,**info):
  f=f.reset_index(drop=True);f['order']=np.arange(len(f));f['unit_id']=[f'{key}-{v}' for v in f.source_row]
  needed=['unit_id','source_row','value','stratum','cluster','x','y','x_label','y_label','order']
  f=f[needed+[c for c in f.columns if c not in needed]]
  assert f.unit_id.is_unique and np.isfinite(f.value).all()
  f.to_csv(root/f'{key}.csv',index=False)
  meta[key]={**info,'file':f'{key}.csv','population_size':len(f),'collected':'2026-11-02',
             'csv_sha256':hashlib.sha256((root/f'{key}.csv').read_bytes()).hexdigest()}
 # CaliforniaHousing source column order follows scikit-learn's official loader.
 with tarfile.open(raw_dir/'housing.tgz') as t:
  a=np.loadtxt(t.extractfile('CaliforniaHousing/cal_housing.data'),delimiter=',')
 f=pd.DataFrame(a,columns=['longitude','latitude','housing_age','rooms','bedrooms','residents','households','income','house_value'])
 f['source_row']=np.arange(1,len(f)+1)
 f=f[f.latitude.between(33.6,34.6)&f.longitude.between(-118.8,-117.6)].copy()
 f=f.sort_values(['latitude','longitude','source_row']).iloc[::max(1,int(np.ceil(len(f)/2200)))].copy()
 f['value']=f.house_value/1000
 f['stratum']=pd.cut(f.latitude,[33.59,33.85,34.1,34.35,34.61],labels=['South','South central','North central','North']).astype(str)
 f['cluster']=['Tile '+str(int(np.floor((x+118.8)/.2)))+'-'+str(int(np.floor((y-33.6)/.2))) for x,y in zip(f.longitude,f.latitude)]
 f['x']=f.longitude;f['y']=f.latitude;f['x_label']=f.longitude;f['y_label']=f.latitude
 save('housing',f,title='Neighborhood housing survey',variable='Block-group median house value',units='thousand 1990 US dollars',
      question='Estimate the mean of block-group median house values in this fixed LA-area teaching frame. Compare geographically spread sampling with visiting selected tiles.',
      population='1990 California census block groups within latitude 33.6–34.6 and longitude −118.8–−117.6, deterministically thinned to at most 2,200 units. The target is this bundled frame, not all California households.',
      layout='Actual latitude/longitude. Geographic tiles and four latitude-band strata are analytical groupings, not county boundaries. The observation is a block-group median, not an individual house price.',
      stratum_label='Latitude band',cluster_label='Geographic tile',visual='map',recommended='Stratified proportional or two-stage geographic sampling',
      source='Pace & Barry (1997), California Housing; archive used by scikit-learn',
      source_url='https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset',
      download_url=URLS['housing.tgz'],license='Original research dataset; retain Pace & Barry attribution and source links.',
      preprocessing='Filter the stated bounding rectangle, sort latitude/longitude/source row, retain every ceil(N/2200)-th row, divide housing value by 1000. Original values are capped at 500001 in the source.',
      x_title='Longitude',y_title='Latitude')
 # Plant: first 28 observed days, 15-minute readings, real load classifications.
 with zipfile.ZipFile(raw_dir/'steel.zip') as z:f=pd.read_csv(z.open('Steel_industry_data.csv'))
 f['source_row']=np.arange(1,len(f)+1);dt=pd.to_datetime(f['date'],dayfirst=True)
 f['timestamp']=dt;f['day']=dt.dt.strftime('%Y-%m-%d');days=sorted(f.day.unique())[:28];f=f[f.day.isin(days)].copy()
 f['value']=f.Usage_kWh;f['stratum']=f.Load_Type.str.replace('_',' ');f['cluster']=f.day
 f['x']=f.timestamp.dt.hour*4+f.timestamp.dt.minute//15;f['y']=f.day.map({d:i for i,d in enumerate(days)})
 f['x_label']=f.timestamp.dt.strftime('%H:%M');f['y_label']=f.day
 f=f[['source_row','value','stratum','cluster','x','y','x_label','y_label','date','Day_of_week']]
 save('steel',f,title='Steel-plant energy audit',variable='Energy per 15-minute interval',units='kWh',
      question='Estimate average interval energy use across the recorded 28-day plant audit. Compare stratifying operating loads with inspecting whole days or taking periodic meter readings.',
      population='The first 28 recorded dates of the UCI steel-industry dataset from a South Korean plant. Each unit is one 15-minute meter record.',
      layout='Calendar heatmap: rows are actual dates and columns are 15-minute clock slots. Load-type strata come from the source; clusters are dates. No production-line measurements are invented.',
      stratum_label='Operating load type',cluster_label='Audit day',visual='calendar',recommended='Stratified proportional by load type; examine systematic aliasing',
      source='Sathishkumar V E, Shin & Cho (2021), UCI Steel Industry Energy Consumption, DOI 10.24432/C52G8C',
      source_url='https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption',download_url=URLS['steel.zip'],license='CC BY 4.0',
      preprocessing='Take the first 28 distinct recorded dates. Retain Usage_kWh without changing measurements.',x_title='15-minute clock slot',y_title='Recorded date')
 # Bikes: first 90 recorded dates, real hourly system demand, no invented stations.
 with zipfile.ZipFile(raw_dir/'bike.zip') as z:f=pd.read_csv(z.open('hour.csv'))
 f['source_row']=f.instant;days=sorted(f.dteday.unique())[:90];f=f[f.dteday.isin(days)].copy()
 f['value']=f.cnt;f['stratum']=pd.cut(f.hr,[-1,5,9,15,19,23],labels=['Overnight','Morning commute','Midday','Evening commute','Late evening']).astype(str)
 f['cluster']=f.dteday;f['x']=f.hr;f['y']=f.dteday.map({d:i for i,d in enumerate(days)})
 f['x_label']=f.hr.astype(str).str.zfill(2)+':00';f['y_label']=f.dteday
 f=f[['source_row','value','stratum','cluster','x','y','x_label','y_label','dteday','hr','workingday','weathersit']]
 save('bikes',f,title='Bike-share demand study',variable='System-wide rentals per hour',units='rentals/hour',
      question='Estimate mean hourly rentals for this fixed winter-period frame. Compare covering each time-of-day band with surveying only convenient early records or a few whole days.',
      population='Hourly Capital Bikeshare rental counts for the first 90 recorded dates in the UCI 2011–2012 dataset. These are system totals, not station-level counts.',
      layout='Date × hour demand heatmap. Five time-of-day strata and day clusters are derived from real timestamps. Blank cells were not recorded and are not sampling units.',
      stratum_label='Time-of-day band',cluster_label='Calendar day',visual='calendar',recommended='Stratified sampling across time of day',
      source='Fanaee-T (2013), UCI Bike Sharing, DOI 10.24432/C5W894',source_url='https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
      download_url=URLS['bike.zip'],license='CC BY 4.0',preprocessing='Take first 90 dates and retain cnt as recorded; derive five time-of-day bands.',x_title='Hour of day',y_title='Recorded date')
 # Air: missing -200 values excluded, first 28 calendar dates retained.
 with zipfile.ZipFile(raw_dir/'air.zip') as z:f=pd.read_csv(z.open('AirQualityUCI.csv'),sep=';',decimal=',')
 f['source_row']=np.arange(1,len(f)+1)
 dt=pd.to_datetime(f.Date.astype(str)+' '+f.Time.astype(str),format='%d/%m/%Y %H.%M.%S',errors='coerce')
 f['timestamp']=dt;f=f[dt.notna()].copy();f['day']=f.timestamp.dt.strftime('%Y-%m-%d');days=sorted(f.day.unique())[:28]
 f=f[f.day.isin(days)&(f['NO2(GT)']!=-200)&f['NO2(GT)'].notna()].copy()
 f['value']=f['NO2(GT)'];f['stratum']=pd.cut(f.timestamp.dt.hour,[-1,5,11,17,23],labels=['Night','Morning','Afternoon','Evening']).astype(str)
 f['cluster']=f.day;f['x']=f.timestamp.dt.hour;f['y']=f.day.map({d:i for i,d in enumerate(days)})
 f['x_label']=f.timestamp.dt.strftime('%H:00');f['y_label']=f.day
 f=f[['source_row','value','stratum','cluster','x','y','x_label','y_label','Date','Time']]
 save('air',f,title='Roadside air-quality monitoring',variable='Hourly reference-analyzer NO₂',units='µg/m³',
      question='Estimate the mean of the available hourly NO₂ readings. Compare spreading measurements across time bands with visiting only selected days.',
      population='Available NO₂ reference-analyzer readings during the first 28 calendar dates of the UCI Italian-city Air Quality record. Missing -200 values are excluded, not set to zero.',
      layout='Date × hour heatmap from one roadside monitoring site, not a network of invented neighborhoods. Four time-of-day strata; days are unequal-size clusters after missing readings are removed.',
      stratum_label='Time-of-day band',cluster_label='Monitoring day',visual='calendar',recommended='Stratified time-of-day monitoring; unequal-size day clusters',
      source='De Vito et al. (2008), UCI Air Quality, DOI 10.24432/C59K5F',source_url='https://archive.ics.uci.edu/dataset/360/air+quality',
      download_url=URLS['air.zip'],license='UCI lists CC BY 4.0 and retains an author research-only note. This bundled frame is for academic teaching/research.',
      preprocessing='Retain the first 28 calendar dates; remove missing NO2(GT)=-200 and NA. Do not impute excluded hours.',x_title='Hour of day',y_title='Recorded date')
 # Actual agronomy experiment; arrangement schematic, not invented field coordinates.
 f=pd.read_csv(raw_dir/'oats.csv');f['source_row']=f.iloc[:,0]
 blocks=sorted(f.Block.unique());varieties=sorted(f.Variety.unique());levels=sorted(f.nitro.unique())
 f['value']=f['yield'];f['stratum']='N='+f.nitro.astype(str);f['cluster']='Block '+f.Block.astype(str)
 f['x']=[varieties.index(v)*len(levels)+levels.index(n) for v,n in zip(f.Variety,f.nitro)]
 f['y']=f.Block.map({b:i for i,b in enumerate(blocks)})
 f['x_label']=f.Variety+' / N='+f.nitro.astype(str);f['y_label']=f.cluster
 f=f.sort_values(['y','x'])
 save('oats',f,title='Oat field-trial yield survey',variable='Subplot oat yield',units='Yates yield units (¼ lb per subplot)',
      question='Estimate mean yield across the 72 recorded subplots. Compare covering all nitrogen treatment groups with selecting entire field blocks.',
      population='All 72 subplots in the nlme Oats split-plot trial: six blocks, three oat varieties, and four nitrogen concentrations. The target is the finite recorded trial, not a causal treatment-effect estimate.',
      layout='Schematic block × variety/nitrogen subplot grid. Blocks and treatments are real; displayed within-block order groups treatments for teaching and is not claimed to be the original field position.',
      stratum_label='Nitrogen treatment',cluster_label='Field block',visual='field',recommended='Stratify by assigned treatment; compare whole-block sampling',
      source='Yates (1935), via Pinheiro & Bates nlme Oats dataset; Rdatasets mirror',
      source_url='https://stat.ethz.ch/R-manual/R-devel/RHOME/library/nlme/html/Oats.html',download_url=URLS['oats.csv'],
      units_reference='https://search.r-project.org/CRAN/refmans/agridat/html/yates.oats.html',license='Source nlme package: GPL (>=2); retain dataset and publication attribution.',
      preprocessing='Retain all 72 yields. Derive nitrogen strata and field-block clusters. Reorder the display by block, variety and nitrogen.',x_title='Variety and nitrogen subplot',y_title='Field block')
 for item in meta.values():
  raw_name=next(k for k,v in URLS.items() if v==item['download_url']);item['raw_sha256']=hashlib.sha256((raw_dir/raw_name).read_bytes()).hexdigest()
 (root/'sources.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
 print({key:(v['population_size'],pd.read_csv(root/v['file']).value.mean()) for key,v in meta.items()})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--raw-dir',default='sampling_raw');args=parser.parse_args();main(args.raw_dir)
