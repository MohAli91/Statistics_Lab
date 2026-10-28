"""Streamlit UI for the head-to-head basketball experiment."""
from __future__ import annotations
import base64
import html
import io
import json
import time
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats
from basketball_game import (TEAMS,TEAM_COLORS,default_lineups,simulate_games,player_summary,standings,STARTERS,JERSEYS,LINEUP_AS_OF)

ASSETS=Path(__file__).parent/'assets'

@st.cache_data
def logo_uri(team):
    name='sixers' if team=='76ers' else 'knicks'
    return 'data:image/svg+xml;base64,'+base64.b64encode((ASSETS/f'{name}.svg').read_bytes()).decode()


def header():
    a,b=st.columns(2)
    for team,col in zip(TEAMS,[a,b]):
        with col:
            st.markdown(f'<div style="display:flex;gap:18px;align-items:center;background:white;padding:12px;border-radius:14px;border:1px solid #dce5ed"><img src="{logo_uri(team)}" width="80" height="80" alt="{team} logo"><div><b style="font-size:24px;color:{TEAM_COLORS[team]}">{"Philadelphia 76ers" if team=="76ers" else "New York Knicks"}</b><br>Five starters · repeated simulated games</div></div>',unsafe_allow_html=True)


def charts_layout(fig,title,height=310):
    fig.update_layout(title=title,height=height,margin=dict(l=20,r=20,t=45,b=30),paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='white',font_color='#12263a',legend=dict(orientation='h',y=1.12))
    return fig


def clock_label(elapsed,settings,end):
    T=settings['duration_min'];q=T/4
    if elapsed>=end-1e-7:return 'FINAL'
    if elapsed<T:
        period=min(4,int(elapsed/q)+1);remaining=period*q-elapsed;label=f'Q{period}'
    else:
        ot=int((elapsed-T)/5)+1;remaining=T+ot*5-elapsed;label=f'OT{ot}'
    seconds=max(0,int(np.ceil(remaining*60)));return f'{label} · {seconds//60:02d}:{seconds%60:02d}'


def victory_markup(game,seen,celebrating=True):
    """Locally rendered Sixers win overlay: logo, MVP, spotlights, confetti."""
    points=seen[seen.team=='76ers'].groupby('player',sort=False).points.sum()
    leader=html.escape(str(points.idxmax())) if len(points) else 'Philadelphia'
    best=int(points.max()) if len(points) else 0
    particles=[]
    if celebrating:
        rng=np.random.default_rng(int(game.game)+7600)
        phase=time.time()%3.6
        for i in range(110):
            left=float(rng.uniform(0,100));width=int(rng.integers(5,12));color=['#006bb6','#ed174c','#ffffff','#ffd166'][i%4]
            delay=-((phase+float(rng.uniform(0,3.6)))%3.6)
            particles.append(f'<i class="confetti-piece" style="left:{left:.2f}%;width:{width}px;height:{width*1.6}px;background:{color};animation-delay:{delay:.3f}s"></i>')
    markup=f'<div class="sixers-party" aria-label="76ers victory celebration"><div class="spotlight left-light"></div><div class="spotlight right-light"></div>{"".join(particles)}<div class="win-card"><img src="{logo_uri("76ers")}" alt="76ers logo"><div class="win-eyebrow">PHILADELPHIA TAKES THE W</div><div class="win-title">SIXERS WIN!</div><div class="win-score">{int(game.sixers_score)} – {int(game.knicks_score)}</div><div class="win-mvp">SCORING LEADER · {leader} · {best} PTS</div></div></div>'
    css="""
    .sixers-party{position:absolute;inset:125px 0 0;overflow:hidden;pointer-events:none;display:grid;place-items:center;background:radial-gradient(ellipse at center,#001b4899,transparent 75%);z-index:10}
    .win-card{position:relative;z-index:12;text-align:center;padding:14px 32px;border:2px solid #ffffffaa;border-radius:20px;background:linear-gradient(135deg,#003b78ee,#001d46ee);box-shadow:0 0 40px #006bb6cc;color:white;animation:win-pop .55s ease-out}
    .win-card img{width:67px;height:67px;background:white;border-radius:50%;padding:7px;animation:logo-bounce 1.2s ease-in-out infinite}
    .win-eyebrow{font-size:10px;letter-spacing:3px;margin-top:7px;color:#bbdefb}.win-title{font-size:clamp(30px,6vw,53px);font-weight:1000;letter-spacing:3px;text-shadow:3px 3px #ed174c;animation:win-glow 1s alternate infinite}
    .win-score{font:bold 30px monospace;color:#ffd166;margin:5px}.win-mvp{font-size:12px;max-width:440px;letter-spacing:1px}
    .confetti-piece{position:absolute;top:-30px;z-index:15;animation:confetti-fall 3.6s linear infinite;opacity:.95}
    .spotlight{position:absolute;width:80px;height:600px;background:linear-gradient(#ffffff55,transparent);transform-origin:top center;filter:blur(12px);z-index:9;top:-30px}
    .left-light{left:8%;animation:sweep-left 2s ease-in-out infinite alternate}.right-light{right:8%;animation:sweep-right 2s ease-in-out infinite alternate}
    @keyframes confetti-fall{0%{transform:translate(0,-40px) rotate(0deg)}100%{transform:translate(55px,560px) rotate(650deg)}}
    @keyframes win-pop{0%{transform:scale(.65);opacity:0}100%{transform:scale(1);opacity:1}}
    @keyframes logo-bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
    @keyframes win-glow{from{filter:brightness(1)}to{filter:brightness(1.25)}}
    @keyframes sweep-left{from{transform:rotate(-20deg)}to{transform:rotate(-55deg)}}
    @keyframes sweep-right{from{transform:rotate(20deg)}to{transform:rotate(55deg)}}
    @media(prefers-reduced-motion:reduce){.sixers-party *{animation:none!important}.confetti-piece{display:none}}
    """
    return markup,css


def migrate_starter_names(run):
    """Upgrade only old position placeholders, preserving custom edited names."""
    for team in TEAMS:
        roster=run['settings']['lineups'][team]
        for i,entry in enumerate(roster):
            if entry['Player']==f'{team} {entry["Position"]}':
                entry['Player']=STARTERS[team][i];entry['Jersey']=JERSEYS[team][i]
                for frame in [run['players'],run['events']]:
                    frame.loc[(frame.team==team)&(frame.player_id==i),'player']=STARTERS[team][i]
    run['settings']['lineup_as_of']=LINEUP_AS_OF


def court_animation(game,events,elapsed,settings,completed,celebrating=False,celebrations_enabled=True):
    seen=events[events.elapsed_min<=elapsed+1e-9]
    last=None if seen.empty else seen.iloc[-1]
    phi=0 if last is None else int(last.sixers_score);nyk=0 if last is None else int(last.knicks_score)
    label=clock_label(elapsed,settings,float(game.duration_min))
    if last is None:call='Tip-off · both teams ready'
    else:call=f'{last.player} · +{int(last.points)} '+('point' if last.points==1 else 'points')
    player_positions={'76ers':[(600,80),(680,290),(560,180),(740,230),(690,120)],
                      'Knicks':[(300,280),(220,70),(340,180),(160,130),(210,240)]}
    players=[]
    for team in TEAMS:
        for i,(x,y) in enumerate(player_positions[team]):
            active=last is not None and last.team==team and int(last.player_id)==i
            color=TEAM_COLORS[team];name=html.escape(settings['lineups'][team][i]['Player'])
            jersey=html.escape(str(settings['lineups'][team][i].get('Jersey',i+1)))
            surname=html.escape(settings['lineups'][team][i]['Player'].split()[-1])
            players.append(f'<g><title>{name}</title><circle cx="{x}" cy="{y}" r="{22 if active else 17}" fill="{color}" stroke="white" stroke-width="{4 if active else 2}"/><text x="{x}" y="{y+5}" fill="white" font-size="14" text-anchor="middle" font-weight="bold">{jersey}</text><text x="{x}" y="{y+33}" fill="#152039" font-size="11" text-anchor="middle" font-weight="bold">{surname}</text></g>')
    ball=''
    if last is not None:
        x,y=player_positions[last.team][int(last.player_id)];hx=850 if last.team=='76ers' else 50
        # A visible curved shot path and a moving ball show the latest scoring play.
        path=f'M {x} {y} Q {(x+hx)/2} {max(15,min(y,180)-100)} {hx} 180'
        ball=f'<path d="{path}" fill="none" stroke="#ffffffaa" stroke-width="2" stroke-dasharray="5 6"/><g><animateMotion dur="0.65s" fill="freeze" path="{path}"/><circle r="10" fill="#ed8b23" stroke="#713f12" stroke-width="2"/><path d="M -10 0 L 10 0 M 0 -10 L 0 10" stroke="#713f12" stroke-width="1.5"/></g>'
    ribbon=''
    if label=='FINAL':
        ribbon=f'<div class="winner">{html.escape(str(game.winner))} WIN · {phi}–{nyk}'+(' · OVERTIME' if game.overtime_periods else '')+'</div>'
    victory,win_css= victory_markup(game,seen,celebrating) if label=='FINAL' and game.winner=='76ers' and celebrations_enabled else ('','')
    doc=f'''<!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}body{{margin:0;font-family:Arial,sans-serif;background:#0b1425;color:#fff;border-radius:16px;overflow:hidden;position:relative}}
    .scoreboard{{display:flex;align-items:center;justify-content:space-around;padding:14px;background:linear-gradient(90deg,#00529b,#152039,#a54d16)}}
    .team{{display:flex;align-items:center;gap:12px}}.team img{{width:58px;height:58px;background:white;border-radius:12px;padding:4px}}
    .score{{font:bold 48px monospace;min-width:66px;text-align:center}}.clock{{text-align:center;font:700 20px monospace;color:#ffe08a}}.small{{font:12px Arial;color:#dbeafe;margin-bottom:7px}}
    .call{{text-align:center;padding:9px;font-size:15px;background:#152039;min-height:36px}}svg{{display:block;width:100%;max-height:340px}}
    .winner{{background:#0d9488;text-align:center;padding:10px;font-weight:bold;animation:celebrate .8s ease-out}}
    @keyframes celebrate{{0%{{letter-spacing:10px;opacity:.2}}100%{{letter-spacing:1px;opacity:1}}}}
    {win_css}
    </style></head><body><div class="scoreboard"><div class="team"><img src="{logo_uri('76ers')}" alt="76ers"><div><div class="small">PHILADELPHIA</div><b>76ERS</b></div><span class="score">{phi}</span></div>
    <div class="clock"><div class="small">GAME {int(game.game)} · {completed} COMPLETED</div>{label}</div>
    <div class="team"><span class="score">{nyk}</span><div><div class="small">NEW YORK</div><b>KNICKS</b></div><img src="{logo_uri('Knicks')}" alt="Knicks"></div></div>
    <div class="call">{html.escape(call)}</div><svg viewBox="0 0 900 360" role="img" aria-label="Animated basketball court with five players per team">
    <defs><pattern id="wood" width="90" height="25" patternUnits="userSpaceOnUse"><rect width="90" height="25" fill="#c89354"/><path d="M0 24 H90 M89 0 V25" stroke="#b27e42" stroke-width="1"/></pattern></defs>
    <rect x="0" y="0" width="900" height="360" fill="url(#wood)"/>
    <rect x="28" y="18" width="844" height="324" fill="none" stroke="white" stroke-width="2"/>
    <path d="M450 18 V342" stroke="white" stroke-width="2"/><circle cx="450" cy="180" r="47" fill="none" stroke="white" stroke-width="2"/>
    <rect x="28" y="115" width="140" height="130" fill="#006bb655" stroke="white" stroke-width="2"/><rect x="732" y="115" width="140" height="130" fill="#f5842655" stroke="white" stroke-width="2"/>
    <circle cx="168" cy="180" r="47" fill="none" stroke="white" stroke-width="2"/><circle cx="732" cy="180" r="47" fill="none" stroke="white" stroke-width="2"/>
    <path d="M28 35 C300 35 300 325 28 325 M872 35 C600 35 600 325 872 325" fill="none" stroke="white" stroke-width="2"/>
    <path d="M38 155 V205 M862 155 V205" stroke="white" stroke-width="5"/><circle cx="50" cy="180" r="9" stroke="#ed6b1b" stroke-width="4" fill="none"/><circle cx="850" cy="180" r="9" stroke="#ed6b1b" stroke-width="4" fill="none"/>
    <image href="{logo_uri('76ers')}" x="418" y="146" width="64" height="64" opacity=".8"/>{''.join(players)}{ball}
    </svg>{ribbon}{victory}</body></html>'''
    components.html(doc,height=520 if label=='FINAL' else 485,scrolling=False)
    return seen


def team_game_table(players,games,team,settings):
    roster=settings['lineups'][team];names=[r['Player'] for r in roster]
    p=players[players.team==team]
    if p.empty:return pd.DataFrame(columns=['Game',*names,'Team total'])
    table=p.pivot(index='game',columns='player',values='points').reindex(columns=names).fillna(0).astype(int)
    table['Team total']=table.sum(axis=1);table.index=[f'Game {i}' for i in table.index]
    table.loc['TOTAL']=table.sum();table.loc['AVERAGE / GAME']=table.iloc[:-1].mean()
    table.index.name='Game'
    return table.reset_index()


def exports(run,completed_games,completed_players,current_events):
    if not st.checkbox('Prepare basketball CSV downloads',False,key='bb_export'):return
    # Only completed games are exported in the season tables; live play log is separate.
    settings=run['settings'];summary=player_summary(completed_players);standing=standings(completed_games)
    all_events=run['events'][run['events'].game<=run['completed']]
    files={'game_results.csv':completed_games,'player_points_by_game.csv':completed_players,
           'player_totals_and_averages.csv':summary,'team_standings.csv':standing,'scoring_events.csv':all_events,
           'current_game_revealed_events.csv':current_events}
    for filename,label in [('game_results.csv','Download final game scores · CSV'),
                           ('player_points_by_game.csv','Download player points by game · CSV'),
                           ('player_totals_and_averages.csv','Download player totals and averages · CSV')]:
        st.download_button(label,files[filename].to_csv(index=False).encode(),filename,'text/csv',key='bb_'+filename)
    if not completed_games.empty:
        for team in TEAMS:
            table=team_game_table(completed_players,completed_games,team,settings)
            st.download_button(f'Download {team} score table · CSV',table.to_csv(index=False).encode(),f'{team.lower()}_score_table.csv','text/csv',key='bb_pivot_'+team)
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,f in files.items():z.writestr(name,f.to_csv(index=False))
        z.writestr('settings.json',json.dumps(settings,indent=2))
    st.download_button('Download all basketball results + settings · ZIP',out.getvalue(),'knicks_76ers_results.zip','application/zip',key='bb_zip')
    st.caption('Season exports include completed games only. The ZIP also contains the revealed live-game event log. Finish all games to export the full simulated season.')


def begin_next_if_needed(run):
    G=len(run['games'])
    if run['completed']>=run['display_game'] and run['completed']<G:
        run['display_game']=run['completed']+1;run['elapsed']=0.


def complete_current(run):
    run['completed']=max(run['completed'],run['display_game'])
    run['elapsed']=float(run['games'].iloc[run['display_game']-1].duration_min)
    if run['completed']==len(run['games']):run['playing']=False


@st.fragment(run_every=.4)
def game_workbench():
    run=st.session_state.get('bb_run')
    if run is None:return
    games=run['games'];players=run['players'];events=run['events'];settings=run['settings'];G=len(games)
    st.divider();st.subheader('Courtside simulation')
    cols=st.columns(6)
    if cols[0].button('▶ Play games',key='bb_play') and run['completed']<G:run['playing']=True
    if cols[1].button('⏸ Pause games',key='bb_pause'):run['playing']=False
    if cols[2].button('🏀 Next scoring play',key='bb_step') and run['completed']<G:
        run['playing']=False;begin_next_if_needed(run)
        future=events[(events.game==run['display_game'])&(events.elapsed_min>run['elapsed']+1e-9)]
        if future.empty:complete_current(run)
        else:run['elapsed']=float(future.iloc[0].elapsed_min)
    if cols[3].button('🏁 Finish this game',key='bb_finish_game') and run['completed']<G:
        run['playing']=False;begin_next_if_needed(run);complete_current(run)
    if cols[4].button('⏩ Finish all games',key='bb_finish_all'):
        run['display_game']=G;run['completed']=G;run['elapsed']=float(games.iloc[-1].duration_min);run['playing']=False
    if cols[5].button('↺ Replay games',key='bb_replay'):
        run.update(completed=0,display_game=1,elapsed=0.,playing=False,celebrated_game=None,celebration_until=0.)
    celebrations=st.checkbox('Celebrate 76ers wins',True,key='bb_celebrations')
    left,right=st.columns(2)
    with left:mode=st.radio('Basketball playback mode',['Live game clock','Games in batches'],horizontal=True,key='bb_mode')
    with right:
        if mode=='Live game clock':speed=st.select_slider('Simulated minutes per tick',[.1,.25,.5,1.,2.,4.,8.,16.],value=1.,key='bb_clock_speed')
        else:speed=st.select_slider('Games per tick',[1,2,5,10,25,50,100],value=5,key='bb_batch_speed')
    with st.expander('Jump to a game / rewind the season'):
        jump=st.number_input('Game to replay',1,G,1,key='bb_jump_number')
        if st.button('Jump to game',key='bb_jump'):
            run.update(completed=jump-1,display_game=jump,elapsed=0.,playing=False,celebrated_game=None,celebration_until=0.)
    if run['playing'] and (not celebrations or time.monotonic()>=run.get('celebration_until',0.)):
        if mode=='Games in batches':
            run['completed']=min(G,run['completed']+int(speed));run['display_game']=run['completed'];complete_current(run)
        else:
            begin_next_if_needed(run);end=float(games.iloc[run['display_game']-1].duration_min)
            run['elapsed']=min(end,run['elapsed']+float(speed))
            if run['elapsed']>=end:complete_current(run)
    if run['completed']>=G:run['playing']=False
    st.progress(run['completed']/G,text=f'{run["completed"]:,} / {G:,} completed games · '+('playing' if run['playing'] else 'paused'))
    game=games.iloc[run['display_game']-1];gevents=events[events.game==run['display_game']]
    won=run['completed']>=run['display_game'] and game.winner=='76ers'
    if won and celebrations and run.get('celebrated_game')!=int(game.game):
        run['celebrated_game']=int(game.game);run['celebration_until']=time.monotonic()+3.6
    celebrating=won and celebrations and time.monotonic()<run.get('celebration_until',0.)
    seen=court_animation(game,gevents,run['elapsed'],settings,run['completed'],celebrating,celebrations)
    if won and celebrations:
        if st.button('🎉 Replay Sixers celebration',key='bb_party_again'):
            run['celebration_until']=time.monotonic()+3.6
        if celebrating:st.caption('SIXERS WIN! Confetti, team spotlights, and the scoring leader take center court. Autoplay holds briefly for the celebration.')
    if mode=='Games in batches':st.caption('Batch mode shows the last completed game; every game and scoring play is recorded. Use Live game clock or Next scoring play for play-by-play.')
    done=games[games.game<=run['completed']];pdone=players[players.game<=run['completed']]
    standing=standings(done)
    cards=st.columns(4)
    for j,team in enumerate(TEAMS):
        row=standing[standing.team==team].iloc[0]
        cards[2*j].metric(f'{team} wins / losses',f'{int(row.wins)} / {int(row.losses)}')
        cards[2*j+1].metric(f'{team} win relative frequency',f'{row.win_frequency:.1%}' if len(done) else '—')
    st.caption('Wins, totals, averages, and relative frequencies use completed games. Live points below reflect the current game clock.')
    columns=st.columns(2)
    summary=player_summary(pdone)
    for team,col in zip(TEAMS,columns):
        with col:
            st.markdown(f'### {team} · five-starter box score')
            current=seen[seen.team==team].groupby('player_id').points.sum()
            roster=pd.DataFrame(settings['lineups'][team]);roster['player_id']=np.arange(5)
            roster['Points this game']=[int(current.get(i,0)) for i in range(5)]
            totals=summary[summary.team==team].set_index('player_id') if not summary.empty else pd.DataFrame()
            roster['Total points (finished games)']=[int(totals.loc[i,'total_points']) if len(totals) else 0 for i in range(5)]
            roster['Average / finished game']=[float(totals.loc[i,'average_points']) if len(totals) else np.nan for i in range(5)]
            st.dataframe(roster[['Player','Position','Points this game','Total points (finished games)','Average / finished game']],hide_index=True,use_container_width=True)
            st.metric(f'{team} current game points',int(roster['Points this game'].sum()))
            st.markdown('**Points in every completed game**')
            table=team_game_table(pdone,done,team,settings)
            if len(table)>202:
                st.caption('Showing the most recent 200 games plus total and average rows. CSVs include every game.')
                display=pd.concat([table.iloc[:-2].tail(200),table.tail(2)])
            else:display=table
            st.dataframe(display,hide_index=True,use_container_width=True)
    st.markdown('### Final scores and standings')
    st.dataframe(standing,hide_index=True,use_container_width=True)
    st.dataframe(done.tail(1000),hide_index=True,use_container_width=True)
    if len(done)>1000:st.caption('Final-score table displays the latest 1,000 games; export includes all completed games.')
    if not done.empty:
        plotcols=st.columns(2)
        with plotcols[0]:
            f=go.Figure()
            for team in TEAMS:
                frequency=(done.winner==team).cumsum()/np.arange(1,len(done)+1)
                f.add_trace(go.Scatter(x=done.game,y=frequency,name=team,line_color=TEAM_COLORS[team]))
            f.update_yaxes(range=[0,1],title='Cumulative win relative frequency');f.update_xaxes(title='Completed games')
            st.plotly_chart(charts_layout(f,'How often does each team win?'),use_container_width=True,key='bb_frequency')
        with plotcols[1]:
            scores=done.melt(id_vars='game',value_vars=['sixers_score','knicks_score'],var_name='team',value_name='points')
            scores.team=scores.team.map({'sixers_score':'76ers','knicks_score':'Knicks'})
            lo=float(scores.points.min())-.5;hi=float(scores.points.max())+.5;width=max(1,int(np.ceil((hi-lo)/45)))
            f=go.Figure()
            for team in TEAMS:f.add_trace(go.Histogram(x=scores[scores.team==team].points,name=team,histnorm='probability',opacity=.6,marker_color=TEAM_COLORS[team],xbins=dict(start=lo,end=hi,size=width)))
            f.update_layout(barmode='overlay');f.update_yaxes(title='Relative frequency');f.update_xaxes(title='Final team points')
            st.plotly_chart(charts_layout(f,'Final-score distributions'),use_container_width=True,key='bb_scores')
        f=px.histogram(done,x='sixers_margin',histnorm='probability',nbins=35,color_discrete_sequence=['#006bb6'])
        f.add_vline(x=0,line_dash='dash');f.update_xaxes(title='76ers points − Knicks points');f.update_yaxes(title='Relative frequency')
        st.plotly_chart(charts_layout(f,'Winning margins · negative values are Knicks wins',260),use_container_width=True,key='bb_margins')
        averages=summary.copy()
        if not averages.empty:
            f=px.bar(averages,x='player',y='average_points',color='team',color_discrete_map=TEAM_COLORS)
            st.plotly_chart(charts_layout(f,'Observed average points per player',280),use_container_width=True,key='bb_player_avg')
    else:st.info('Complete a game to begin the season statistics and relative-frequency plots.')
    with st.expander('Current game scoring timeline and play-by-play'):
        if seen.empty:st.caption('No scoring plays revealed yet.')
        else:
            f=go.Figure()
            for team,col in [('76ers','sixers_score'),('Knicks','knicks_score')]:
                f.add_trace(go.Scatter(x=np.r_[0,seen.elapsed_min],y=np.r_[0,seen[col]],name=team,line=dict(color=TEAM_COLORS[team],shape='hv')))
            f.update_xaxes(title='Elapsed simulated minutes');f.update_yaxes(title='Points')
            st.plotly_chart(charts_layout(f,'Live scoring progression'),use_container_width=True,key='bb_timeline')
            st.dataframe(seen.tail(100),hide_index=True,use_container_width=True)
    with st.expander('Download basketball results',expanded=False):exports(run,done,pdone,seen)


def render_basketball_experiment():
    st.subheader('🏀 Knicks vs. 76ers · head-to-head games')
    header()
    st.caption('Real starter names, verified October 28, 2026; names remain editable. Scoring parameters are classroom settings, not fitted player averages. The default scenario deliberately favors the 76ers.')
    with st.expander('Game settings · teams, players, and scoring rates',expanded=True):
        preset=st.selectbox('Matchup preset',['Philadelphia favored','Balanced matchup','Knicks favored'],key='bb_preset')
        defaults=default_lineups('Balanced matchup' if preset=='Balanced matchup' else 'Philadelphia favored')
        strengths_default={'Philadelphia favored':(1.,.75),'Balanced matchup':(1.,1.),'Knicks favored':(.65,1.25)}[preset]
        cols=st.columns(2);lineups={};strengths={}
        for j,(team,col) in enumerate(zip(TEAMS,cols)):
            with col:
                st.markdown(f'**{team} starters**')
                strengths[team]=st.slider(f'{team} scoring strength',.0,2.,strengths_default[j],.05,key=f'bb_strength_{team}_{preset}')
                lineups[team]=st.data_editor(defaults[team],num_rows='fixed',hide_index=True,use_container_width=True,
                    column_config={'Player':st.column_config.TextColumn('Player (editable name)',required=True),
                                   'Position':st.column_config.TextColumn('Position'),
                                   'Jersey':st.column_config.NumberColumn('Jersey',min_value=0,max_value=99,step=1),
                                   'Expected points / 48 min':st.column_config.NumberColumn('Expected points / 48 min',min_value=0.,max_value=100.,step=1.,required=True)},
                    key=f'bb_lineup_real_v2_{team}_{preset}')
        a,b,c=st.columns(3)
        with a:G=st.number_input('Number of games',1,5000,100,key='bb_games')
        with b:T=st.number_input('Regulation game length T (minutes)',1.,96.,48.,1.,key='bb_T')
        with c:seed=st.number_input('Basketball random seed',0,2**32-1,42,key='bb_seed')
        weights_text=st.text_input('Weights for 1-, 2-, and 3-point scoring plays','12,60,28',key='bb_weights')
        try:
            weights=np.array([float(x.strip()) for x in weights_text.split(',')])
            if len(weights)!=3 or not np.all(np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:raise ValueError('Enter three nonnegative weights with a positive sum.')
            weights/=weights.sum();m=float(weights@np.arange(1,4));second=float(weights@(np.arange(1,4)**2));expect={}
            for team in TEAMS:
                means=np.asarray(lineups[team]['Expected points / 48 min'],float)
                expect[team]=float(means.sum()*strengths[team]*T/48)
            st.write(f'Expected regulation scores: **76ers {expect["76ers"]:.1f} — Knicks {expect["Knicks"]:.1f}**')
            variance=(expect['76ers']+expect['Knicks'])*second/m
            if variance>0:
                approx=stats.norm.cdf((expect['Knicks']-expect['76ers'])/np.sqrt(variance))
                st.caption(f'Rough normal approximation to a Knicks regulation win: {approx:.1%}. This is a model approximation, not a guarantee or a fitted NBA prediction; ties and overtime change the final-game probability.')
            if preset=='Philadelphia favored' and expect['Knicks']>=expect['76ers']:st.warning('Your edits removed Philadelphia’s expected scoring advantage. Lower Knicks strength or player means to make their wins rare again.')
            if st.button('Generate Knicks–76ers games',type='primary',key='bb_generate'):
                with st.spinner('Generating games and scoring plays…'):
                    g,p,e=simulate_games(lineups,strengths,T,int(G),int(seed),weights)
                settings={'experiment':'Knicks vs. 76ers','model':'independent compound-Poisson player scoring processes',
                          'duration_min':T,'games':int(G),'seed':int(seed),'scoring_weights':weights.tolist(),
                          'strengths':strengths,'lineups':{team:lineups[team].to_dict('records') for team in TEAMS},
                          'created_utc':datetime.now(timezone.utc).isoformat(),'schema_version':2,'lineup_as_of':LINEUP_AS_OF}
                st.session_state.bb_run=dict(games=g,players=p,events=e,settings=settings,completed=0,display_game=1,elapsed=0.,playing=False)
        except (ValueError,TypeError,OverflowError) as exc:st.error(str(exc))
        st.caption('Each scoring play adds 1, 2, or 3 points. Scoring-play rate λ = (player expected points per 48 minutes × team strength) / (48 × mean points per play). T changes expected points; playback speed changes only the animation. Ties trigger five-minute overtime periods.')
    if st.session_state.get('bb_run') is not None:migrate_starter_names(st.session_state.bb_run)
    game_workbench()
    with st.expander('Model assumptions and team-logo sources'):
        st.markdown('Only the five configured starters score. Their independent scoring rates are constant through regulation and overtime. The model omits defense interactions, substitutions, possessions, fatigue, and real player statistics. After 20 tied overtimes an explicitly flagged sudden-death extension resolves the game. Default names use verified starting-lineup sources as of October 28, 2026. Game-day starters can vary; names remain editable.')
        st.markdown('Actual team logos are bundled from the NBA official CDN: [76ers logo](https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg) · [Knicks logo](https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg). Team names and logos belong to their respective owners.')

    with st.expander('Starting-lineup references (October 28, 2026)'):
        st.markdown('76ers: [official team starting-five reference](https://www.nba.com/sixers/76ers-players/lebron-james). Knicks: [NBA starting-lineup reference](https://www.nba.com/news/how-the-knicks-won-the-2026-nba-championship) and [October 28 projected lineup](https://www.rotowire.com/betting/nba/game/bulls-vs-knicks-odds-2026-10-28-3018316).')
