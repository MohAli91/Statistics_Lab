"""Excel downloads for sampling labs; openpyxl is the only added dependency."""
from __future__ import annotations
import io,json
import numpy as np
import pandas as pd
from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter


def sampling_workbook(population,metadata,config,sample=None,summary=None,repeated=None,draws=None,parameters=None):
    b=io.BytesIO()
    params=[('Scenario',metadata['title']),('Variable',metadata['variable']),('Units',metadata['units']),
            ('Population definition',metadata['population']),('Layout',metadata['layout']),
            ('Sampling method',config['method']),('Design settings',json.dumps(config)),
            ('Source attribution',metadata['source']),('Source URL',metadata['source_url']),
            ('Download URL',metadata['download_url']),('Source license/notes',metadata['license']),
            ('Preprocessing',metadata['preprocessing']),('Source file SHA256',metadata['raw_sha256']),
            ('Frame CSV SHA256',metadata['csv_sha256']),
            ('Inference','Approximate design-aware t intervals use finite-population corrections. Systematic single-start and convenience inference is available only as explicitly labeled working-SRS analysis.'),
            ('Repeat interpretation','Each replicate redraws a sample independently from the SAME fixed frame. This is not bootstrap sampling from the first sample.')]
    params.extend((str(k),json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in (parameters or {}).items())
    frames={'Parameters':pd.DataFrame(params,columns=['Setting','Value']),'Population':population}
    if sample is not None:frames['Sample']=sample
    if summary is not None:frames['Single_summary']=pd.DataFrame({'Metric':summary.index,'Value':summary.values})
    if repeated is not None:frames['Repeated_results']=repeated
    if draws is not None:frames['Repeated_draws']=draws
    frames['Strata']=population.groupby('stratum').value.agg(['count','mean','std','min','max']).reset_index()
    frames['Clusters']=population.groupby('cluster').value.agg(['count','mean','std','min','max']).reset_index()
    with pd.ExcelWriter(b,engine='openpyxl') as writer:
        for name,frame in frames.items():
            clean=frame.replace([np.inf,-np.inf],np.nan)
            clean.to_excel(writer,sheet_name=name,index=False)
            ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            ws.row_dimensions[1].height=32
            for cell in ws[1]:
                cell.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF')
                cell.fill=PatternFill('solid',fgColor='12465A');cell.alignment=Alignment(wrap_text=True,vertical='center')
                cell.border=Border(bottom=Side(style='thin',color='0D9488'))
            for j,column in enumerate(clean.columns,1):
                values=clean[column].head(150).astype(str)
                width=min(42,max(14,len(str(column))+2,max((len(v) for v in values),default=0)+2))
                ws.column_dimensions[get_column_letter(j)].width=width
                for cells in ws.iter_cols(min_col=j,max_col=j,min_row=2):
                    for cell in cells:
                        if isinstance(cell.value,(float,np.floating)):cell.number_format='#,##0.000'
                        elif isinstance(cell.value,(int,np.integer)) and not isinstance(cell.value,bool):cell.number_format='#,##0'
            if name in ['Parameters','Single_summary']:
                ws.column_dimensions['A'].width=29;ws.column_dimensions['B'].width=105
                for row in ws.iter_rows(min_row=2):
                    for c in row:c.alignment=Alignment(wrap_text=True,vertical='top')
                    ws.row_dimensions[row[0].row].height=max(22,15*(1+len(str(row[1].value or ''))//100))
                    if str(row[1].value).startswith('https://'):row[1].hyperlink=row[1].value;row[1].font=Font(color='0563C1',underline='single')
            if name=='Population' and len(clean)>0:
                j=list(clean.columns).index('value')+1;letter=get_column_letter(j)
                ws.conditional_formatting.add(f'{letter}2:{letter}{len(clean)+1}',ColorScaleRule(start_type='min',start_color='E0F2FE',end_type='max',end_color='006B8F'))
            if name=='Repeated_results':
                for heading in ['covers','working_covers']:
                    if heading in clean:
                        j=list(clean.columns).index(heading)+1
                        for row in ws.iter_rows(min_col=j,max_col=j,min_row=2):
                            cell=row[0]
                            if cell.value is not None:cell.fill=PatternFill('solid',fgColor='D1FAE5' if bool(cell.value) else 'FEE2E2')
    return b.getvalue()
