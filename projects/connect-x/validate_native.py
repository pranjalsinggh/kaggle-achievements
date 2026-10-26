"""Run the original native-agent checks with explicit local artifacts on Linux."""
from pathlib import Path
import ctypes,time,json,sys,platform
from types import SimpleNamespace
root=Path(__file__).resolve().parent
output=root/'validation-output'
output.mkdir(exist_ok=True)
ns={}
sys.path.append(str(root))
exec(compile((root/'main.py').read_text(),str(root/'main.py'),'exec'),ns)
sys.path.pop()
config=SimpleNamespace(rows=6,columns=7,inarow=4,__raw_path__=str(root/'main.py'))
board=[0]*42
board[38]=1
start=time.perf_counter()
action=ns['agent'](SimpleNamespace(board=board,mark=2),config)
elapsed=time.perf_counter()-start
assert action in range(7) and board[action]==0
assert elapsed<1.95,elapsed
print('NATIVE_LIBRARY_PASS',{'move':action,'seconds':elapsed,'depth':ns['_LIB'].cx_depth(),'nodes':ns['_LIB'].cx_nodes()})

from kaggle_environments.envs.connectx.connectx import play,is_win,negamax_agent
import random
rng=random.Random(2026)
checks=0
for _ in range(500):
    b=[0]*42;mark=1;terminal=False
    for _ in range(rng.randrange(3,32)):
        legal=[c for c in range(7) if not b[c]]
        if not legal: terminal=True;break
        col=rng.choice(legal);play(b,col,mark,config)
        if is_win(b,col,mark,config): terminal=True;break
        mark=3-mark
    if terminal: continue
    for who in [1,2]:
        own=occupied=0
        for row in range(6):
            for col in range(7):
                bit=1<<(7*col+5-row)
                if b[row*7+col]: occupied|=bit
                if b[row*7+col]==who: own|=bit
        possible=(occupied+sum(1<<(7*c) for c in range(7)))&sum(63<<(7*c) for c in range(7))
        computed=ns['_LIB'].cx_winning(own,occupied)&possible
        got=[c for c in range(7) if computed&(63<<(7*c))]
        expected=[c for c in range(7) if not b[c] and is_win(b,c,who,config,False)]
        assert got==expected,(got,expected)
    checks+=1
print('TACTICAL_CHECKS_PASS',checks)
from kaggle_environments import make
# This uses Kaggle's actual Python-file loader with the submitted archive files.
env=make('connectx',debug=True)
start=time.perf_counter()
env.run([str(root/'main.py'),str(root/'main.py')])
assert all(s['status']=='DONE' for s in env.state),[(s['status'],s.get('action')) for s in env.state]
print('OFFICIAL_SELFPLAY_PASS',{'seconds':time.perf_counter()-start,'rewards':[s['reward'] for s in env.state],'steps':len(env.steps)})

import random,json
results={}
for opponent in ['random','negamax']:
    outcomes=[]
    for i in range(8):
        random.seed(2026+i);b=[0]*42;seat=1+i%2;mark=1
        play(b,i%7,mark,config);mark=2;outcome=0
        for ply in range(1,42):
            if mark==seat: col=ns['choose'](b,mark,budget=.045,configuration=config)
            elif opponent=='random': col=random.choice([c for c in range(7) if not b[c]])
            else: col=negamax_agent(SimpleNamespace(board=b[:],mark=mark),config)
            assert col in range(7) and b[col]==0
            play(b,col,mark,config)
            if is_win(b,col,mark,config):
                outcome=1 if mark==seat else -1;break
            mark=3-mark
        outcomes.append(outcome)
    results[opponent]={'wins':outcomes.count(1),'draws':outcomes.count(0),'losses':outcomes.count(-1)}
print('NATIVE_BENCHMARK_RESULTS',json.dumps(results))
(output/'runtime_validation.json').write_text(json.dumps({'native_first_call_seconds':elapsed,'tactical_boards':checks,'selfplay_status':[s['status'] for s in env.state],'selfplay_rewards':[s['reward'] for s in env.state],'benchmarks':results},indent=2))
