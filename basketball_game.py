"""Knicks–76ers classroom game model, independent of Streamlit.

Each player's scoring plays form an independent homogeneous Poisson process.
A scoring play is marked with 1, 2, or 3 points. Player points are therefore
compound Poisson, not plain Poisson. Defaults intentionally favor Philadelphia.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TEAMS=('76ers','Knicks')
TEAM_COLORS={'76ers':'#006bb6','Knicks':'#f58426'}
POSITIONS=['PG','SG','SF','PF','C']
# Verified October 28, 2026; sources are documented in LINEUP_SOURCES.md.
STARTERS={
    '76ers':['Tyrese Maxey','VJ Edgecombe','Jaylen Brown','LeBron James','Joel Embiid'],
    'Knicks':['Jalen Brunson','Mikal Bridges','Josh Hart','OG Anunoby','Karl-Anthony Towns'],
}
JERSEYS={'76ers':[0,77,7,23,21],'Knicks':[11,25,3,8,32]}
LINEUP_AS_OF='2026-10-28'
EVENT_COLUMNS=['game','event','elapsed_min','period','team','player_id','player','points','sixers_score','knicks_score']


def default_lineups(preset='Philadelphia favored'):
    phi=[30.,27.,24.,21.,18.]
    nyk=[28.,25.,22.,20.,17.]
    if preset=='Balanced matchup':nyk=phi.copy()
    return {team:pd.DataFrame({'Player':STARTERS[team], 'Jersey':JERSEYS[team],
                              'Position':POSITIONS,'Expected points / 48 min':means})
            for team,means in [('76ers',phi),('Knicks',nyk)]}


def validate(lineups,strengths,duration,games,weights):
    if not 1<=games<=5000:raise ValueError('Use 1–5,000 games.')
    if not np.isfinite(duration) or not 1<=duration<=96:raise ValueError('Game duration must be 1–96 minutes.')
    p=np.asarray(weights,dtype=float)
    if p.shape!=(3,) or not np.all(np.isfinite(p)) or np.any(p<0) or p.sum()<=0:
        raise ValueError('Enter three nonnegative scoring-play weights with a positive sum.')
    p=p/p.sum();mark_mean=float(p@np.arange(1,4));expected=0.
    for team in TEAMS:
        f=lineups[team]
        if len(f)!=5:raise ValueError(f'{team} must have exactly five starters.')
        names=f.Player.astype(str).str.strip()
        if (names=='').any() or names.duplicated().any():raise ValueError(f'{team}: use five nonempty, unique player names.')
        means=np.asarray(f['Expected points / 48 min'],dtype=float)
        if not np.all(np.isfinite(means)) or np.any(means<0) or np.any(means>100):
            raise ValueError('Expected points must be finite and between 0 and 100 per player.')
        if not np.isfinite(strengths[team]) or strengths[team]<0 or strengths[team]>3:raise ValueError('Team strengths must be between 0 and 3.')
        expected+=means.sum()*strengths[team]*duration/48/mark_mean
    if expected==0:raise ValueError('At least one team must have a positive scoring rate.')
    if games*expected>500_000:raise ValueError('Reduce games, scoring strengths, or duration (500,000 expected scoring-play budget).')
    return p,mark_mean


def simulate_games(lineups,strengths,duration,games,seed,weights=(.12,.60,.28),form_shape=None):
    """Return games, player-game points, and complete scoring-event tables.

    Games use independent child seeds, so the first k games are reproducible
    when requesting more games. Tied games get five-minute overtime periods.
    An emergency sudden-death extension follows 20 tied OTs for termination
    under very low scoring rates; this simplified rule is explicitly recorded.
    """
    p,mark_mean=validate(lineups,strengths,duration,games,weights)
    if form_shape is not None and (not np.isfinite(form_shape) or not .1<=form_shape<=100):
        raise ValueError('Game-to-game form shape must be between 0.1 and 100.')
    base_rates={team:np.asarray(lineups[team]['Expected points / 48 min'],float)*strengths[team]/48/mark_mean for team in TEAMS}
    games_rows=[];player_rows=[];event_rows=[];actual_events=0
    for game,stream in enumerate(np.random.SeedSequence(seed).spawn(games),1):
        rng=np.random.default_rng(stream);points={t:np.zeros(5,dtype=int) for t in TEAMS};local=[]
        form={t:np.ones(5) if form_shape is None else rng.gamma(form_shape,1/form_shape,5) for t in TEAMS}
        rates={t:base_rates[t]*form[t] for t in TEAMS}
        def period_events(start,length,ot=0):
            nonlocal actual_events
            batch=[]
            for team in TEAMS:
                for player,rate in enumerate(rates[team]):
                    k=int(rng.poisson(rate*length));actual_events+=k
                    if actual_events>650_000:raise ValueError('Realized event budget exceeded; reduce the run size.')
                    ts=start+rng.uniform(0,length,k);marks=rng.choice([1,2,3],k,p=p)
                    for t,v in zip(ts,marks):
                        period=f'OT{ot}' if ot else f'Q{min(4,int(t/(duration/4))+1)}'
                        batch.append((float(t),team,player,int(v),period))
            batch.sort(key=lambda x:x[0]);return batch
        def apply(batch):
            for t,team,player,v,period in batch:
                points[team][player]+=v
                local.append({'game':game,'event':len(local)+1,'elapsed_min':t,'period':period,
                              'team':team,'player_id':player,'player':str(lineups[team].iloc[player].Player),
                              'points':v,'sixers_score':int(points['76ers'].sum()),'knicks_score':int(points['Knicks'].sum())})
        apply(period_events(0,duration));reg_phi=int(points['76ers'].sum());reg_nyk=int(points['Knicks'].sum())
        ot=0;end=duration;sudden=False
        while points['76ers'].sum()==points['Knicks'].sum() and ot<20:
            ot+=1;apply(period_events(end,5.,ot));end+=5.
        if points['76ers'].sum()==points['Knicks'].sum():
            sudden=True;all_rates=np.r_[rates['76ers'],rates['Knicks']];total_rate=all_rates.sum()
            idx=int(rng.choice(10,p=all_rates/total_rate));team=TEAMS[idx//5];player=idx%5
            end+=float(rng.exponential(1/total_rate));v=int(rng.choice([1,2,3],p=p))
            apply([(end,team,player,v,'Sudden death')]);actual_events+=1
        phi=int(points['76ers'].sum());nyk=int(points['Knicks'].sum());winner='76ers' if phi>nyk else 'Knicks'
        games_rows.append({'game':game,'sixers_score':phi,'knicks_score':nyk,'winner':winner,'sixers_margin':phi-nyk,
                           'regulation_sixers':reg_phi,'regulation_knicks':reg_nyk,'overtime_periods':ot,
                           'sudden_death':sudden,'duration_min':end})
        for team in TEAMS:
            for i,value in enumerate(points[team]):
                player_rows.append({'game':game,'team':team,'player_id':i,'player':str(lineups[team].iloc[i].Player),
                                    'position':str(lineups[team].iloc[i].Position),'points':int(value),
                                    'form_multiplier':float(form[team][i])})
        event_rows.extend(local)
    return (pd.DataFrame(games_rows),pd.DataFrame(player_rows),pd.DataFrame(event_rows,columns=EVENT_COLUMNS))


def player_summary(players):
    if players.empty:return pd.DataFrame(columns=['team','player_id','player','position','games','total_points','average_points','min_points','max_points'])
    return players.groupby(['team','player_id','player','position'],sort=False).points.agg(
        games='count',total_points='sum',average_points='mean',min_points='min',max_points='max').reset_index()


def standings(games):
    rows=[]
    for team,score,against in [('76ers','sixers_score','knicks_score'),('Knicks','knicks_score','sixers_score')]:
        wins=int((games.winner==team).sum());n=len(games)
        rows.append({'team':team,'games':n,'wins':wins,'losses':n-wins,'win_frequency':wins/n if n else 0.,
                     'total_points':int(games[score].sum()),'points_per_game':games[score].mean() if n else 0.,
                     'opponent_ppg':games[against].mean() if n else 0.})
    return pd.DataFrame(rows)
