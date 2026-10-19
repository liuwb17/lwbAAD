"""Strict proposal contracts, exact edits, and deterministic paired selection."""
import json
import math
import random
import statistics
from .io import digest

END='// AAD_V4_COMPLETE'

def json_object(text):
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Expected JSON object')
    json.dumps(value,allow_nan=False)
    return value

def plan(text):
    value=json_object(text)
    for key in ('hypothesis','mechanism','expected_effect','risk','test'):
        if not isinstance(value.get(key),str) or not value[key].strip() or len(value[key])>2000:
            raise ValueError('Missing or oversized plan field: '+key)
    return value

def program(text):
    code=text.strip()
    if code.startswith('```'):
        lines=code.splitlines()
        if lines[-1].strip()!='```':raise ValueError('Unclosed code fence')
        code='\n'.join(lines[1:-1]).strip()
    if not code.endswith(END):raise ValueError('Missing program completion sentinel')
    if len(code.encode('utf-8'))>120000:raise ValueError('Program exceeds source limit')
    if 'main(' not in code.replace(' ',''):raise ValueError('Missing main')
    return code+'\n'

def apply_edit(parent,text):
    """One contiguous intervention, one hypothesis. Never fuzzy-match model edits."""
    proposal=json_object(text)
    if proposal.get('parent_sha256')!=digest(parent):raise ValueError('Wrong edit parent')
    edits=proposal.get('edits')
    if not isinstance(edits,list) or len(edits)!=1:raise ValueError('Exactly one atomic edit required')
    old=edits[0].get('old');new=edits[0].get('new')
    if not isinstance(old,str) or not isinstance(new,str) or not old or old==new:
        raise ValueError('Invalid or no-op edit')
    if len(old)+len(new)>32000:raise ValueError('Edit too large; propose a smaller intervention')
    if parent.count(old)!=1:raise ValueError('Edit anchor is not unique')
    return program(parent.replace(old,new,1))

def vectors(results):
    if not results or any(not r.get('valid') for r in results):raise ValueError('All repeats must be valid')
    rows=[{r['name']:r['score']/1e9 for r in result['cases']} for result in results]
    names=sorted(rows[0])
    if not names or any(set(row)!=set(names) for row in rows):raise ValueError('Different repeat case sets')
    return {name:statistics.mean(row[name] for row in rows) for name in names}

def paired_decision(challenger,incumbent,margin=.0001,alpha=.05,samples=2000):
    """Case-cluster bootstrap over repeat means; a conservative selection heuristic.

    Not a guarantee of population significance after adaptive repeated selection.
    Repeats average timing variation, NOT extra independent test cases.
    """
    if not math.isfinite(margin) or margin<0 or not 0<alpha<.5 or samples<100:raise ValueError('Invalid selection policy')
    a=vectors(challenger);b=vectors(incumbent)
    if set(a)!=set(b):raise ValueError('Unpaired evaluation')
    delta=[a[n]-b[n] for n in sorted(a)];rng=random.Random(44001)
    boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(samples))
    low=boot[int(alpha/2*samples)];high=boot[min(samples-1,int((1-alpha/2)*samples))]
    mean=statistics.mean(delta)
    return {'mean_gain':mean,'lower':low,'upper':high,'margin':margin,'alpha':alpha,
            'case_count':len(delta),'repeats':len(challenger),'promote':mean>margin and low>0,
            'interpretation':'paired case-bootstrap heuristic, not multiplicity-corrected evidence'}

def choose_arm(history,arms):
    """Token-cost-aware UCB, measured reward only; explicitly explore unused arms."""
    for arm in arms:
        if not any(e['arm']==arm for e in history):return arm
    total=max(2,len(history))
    def value(arm):
        events=[e for e in history if e['arm']==arm]
        reward=statistics.mean(e.get('reward',0)/max(1,e.get('tokens',0)/10000) for e in events)
        return reward+.15*math.sqrt(2*math.log(total)/len(events))
    return max(arms,key=value)

def parameter_influence(default_results,probe_results):
    """Fail closed: output changes are necessary, not proof of intended semantics."""
    if len(default_results)!=len(probe_results) or not default_results:raise ValueError('Unpaired probes')
    def outputs(group):
        if not group['valid']:raise ValueError('Invalid influence probe')
        return {r['name']:digest(r['stdout']) for r in group['cases']}
    default=[outputs(g) for g in default_results];probe=[outputs(g) for g in probe_results]
    if any(set(a)!=set(b) for a,b in zip(default,probe)):raise ValueError('Different probe cases')
    unstable=any(d!=default[0] for d in default[1:])
    return {'default_output_unstable':unstable,'changed':any(a!=b for a,b in zip(default,probe)),
            'accept':not unstable and any(a!=b for a,b in zip(default,probe)),
            'note':'Inconclusive if defaults vary with wall-clock timing; do not admit unverified knobs'}
