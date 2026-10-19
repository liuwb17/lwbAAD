"""Bounded exploration, coordinated edits and runtime feedback for v5."""
import json
import statistics
from .io import digest

END='// AAD_V5_COMPLETE'
CYCLE=('local','redesign','runtime','local','crossover','redesign','local','independent')

def program(text):
    code=text.strip()
    if code.startswith('```'):
        lines=code.splitlines()
        if lines[-1].strip()!='```':raise ValueError('Unclosed source fence')
        code='\n'.join(lines[1:-1]).strip()
    if not code.endswith(END):raise ValueError('Missing complete-program marker')
    if len(code.encode('utf-8'))>160000:raise ValueError('Source too large')
    if 'main(' not in ''.join(code.split()):raise ValueError('No main function')
    return code+'\n'

def edit(parent,text):
    payload=json.loads(text)
    if not isinstance(payload,dict):raise ValueError('Expected edit object')
    if payload.get('parent_sha256')!=digest(parent):raise ValueError('Parent hash differs')
    if not isinstance(payload.get('hypothesis'),str) or not 1<=len(payload['hypothesis'])<=3000:
        raise ValueError('Missing concise change hypothesis')
    edits=payload.get('edits')
    if not isinstance(edits,list) or not 1<=len(edits)<=6:raise ValueError('Expected 1..6 coordinated edits')
    spans=[]
    for item in edits:
        if not isinstance(item,dict):raise ValueError('Invalid edit item')
        old=item.get('old');new=item.get('new')
        if not isinstance(old,str) or not isinstance(new,str) or not old or old==new:raise ValueError('Invalid edit')
        if parent.count(old)!=1:raise ValueError('Each anchor must occur once in ORIGINAL parent')
        start=parent.index(old);spans.append((start,start+len(old),new))
    spans.sort()
    if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):raise ValueError('Overlapping edit anchors')
    code=parent
    for start,end,new in reversed(spans):code=code[:start]+new+code[end:]
    return program(code),payload['hypothesis']

def repeated_mean(record):
    repeats=record.get('repeats',[])
    if repeats and all(r['valid'] for r in repeats):return statistics.mean(r['mean'] for r in repeats)
    return record.get('full',{}).get('mean',-1)

def valid_for_archive(record):
    return record.get('full',{}).get('valid',False) and all(r['valid'] for r in record.get('repeats',[]))

def pool(records,limit=8):
    eligible=sorted((k for k,r in records.items() if valid_for_archive(r)),key=lambda k:repeated_mean(records[k]),reverse=True)
    selected=[];families={}
    for k in eligible:
        root=records[k]['lineage']
        if families.get(root,0)>=2:continue
        selected.append(k);families[root]=families.get(root,0)+1
        if len(selected)==limit:break
    return selected

def feedback_timing(record):
    full=record['full'];short=record.get('smoke',{})
    names={r['name'] for r in short.get('cases',[])}
    paired=[r['score']/1e9 for r in full['cases'] if r['name'] in names]
    elapsed=statistics.mean(r['seconds'] for r in full['cases'])
    return {'allocated_seconds':4.8,'mean_process_seconds':elapsed,'utilization':elapsed/4.8,
        'same_cases_score_gain_from_short':statistics.mean(paired)-short['mean'] if paired else None,
        'underutilized':elapsed<2.4 and full['mean']<.999,
        'instruction':'If quality remains low and extra time brings no gains, diagnose termination or reachable-state limitations. Propose useful additional search. Never add sleep, busy work, padding, or reward time consumption itself.'}

def choose_operator(events,slot):
    arm=CYCLE[slot%len(CYCLE)]
    # Hard recent quota, independent of historical rewards.
    if arm=='crossover' and any(e['arm']=='crossover' for e in events[-7:]):arm='redesign'
    if arm not in ('independent','redesign'):
        recent=[e for e in events[-16:] if e['arm']==arm][-3:]
        if len(recent)==3 and all(e.get('parent_gain',0)<=.00005 for e in recent):arm='redesign'
    return arm
