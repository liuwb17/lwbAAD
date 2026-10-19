"""Mean-oriented exploration with runtime intervention and repeat-failure repairs."""
import json
import statistics
from pathlib import Path
from .io import digest,read_json,save_json
from .provider import BudgetExhausted
from .base_engine import V4Engine
from .search_protocol import vectors,paired_decision
from .layout_diagnostics import layout_diagnostics
from .candidate_policy import program,edit,pool,repeated_mean,valid_for_archive,feedback_timing,choose_operator

class V5Engine(V4Engine):
    # Reuse only checkpoint/source/repeat/response-recovery helpers, not v4 scheduling.
    def __init__(self,root,provider,evaluate,train,validation,config):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.path=self.root/'state.json'
        self.provider=provider;self.evaluate=evaluate;self.train=train;self.validation=validation;self.config=config
        if len(train)!=40 or len(validation)!=10 or {c.fingerprint for c in train}&{c.fingerprint for c in validation}:
            raise ValueError('Expected disjoint 40/10 public split')
        protocol={'version':5,'config':config,'inputs':{c.name:c.fingerprint for c in [*train,*validation]},
                  'train':[c.name for c in train],'validation':[c.name for c in validation]}
        signature=digest(json.dumps(protocol,sort_keys=True))
        if self.path.exists():
            self.s=read_json(self.path)
            if self.s['signature']!=signature:raise ValueError('Protocol changed')
        else:
            self.s={'signature':signature,'records':{},'events':[],'root_attempts':0,'schedule_slot':0,
                'incumbent':None,'pending':None,'repair':None,'validation_started':False,'frozen':False}
        save_json(self.root/'protocol.json',protocol);self.save()

    def feedback(self,ident):
        record=self.s['records'][ident];by_name={c.name:c for c in self.train}
        result=record.get('full',record['smoke'])
        # Do not lose failures that occur only on confirmation repeats.
        failed=next((r for r in record.get('repeats',[]) if not r['valid']),None)
        if failed:result=failed
        rows=sorted(result['cases'],key=lambda r:(r['status']=='AC',r['score']))[:2]
        examples=[]
        for row in rows:
            case=by_name[row['name']]
            item={'name':row['name'],'score':row['score'],'status':row['status'],'input':case.text,
                'output':row.get('stdout',''),'message':row.get('message','')[-6000:]}
            if row['status']=='AC':item['measured_geometry']=layout_diagnostics(case.text,row['stdout'])
            examples.append(item)
        return {'train_mean':result['mean'],'examples':examples,'failure_on_repeat':failed is not None,
            'timing':feedback_timing(record) if record.get('full',{}).get('valid') else None,
            'target':'Improve generalizable true score; validity is mandatory; runtime itself earns no reward.'}

    def task(self):
        records=self.s['records'];events=self.s['events'];archive=pool(records)
        if self.s['repair']:
            parent=self.s['repair'];self.s['repair']=None
            return {'arm':'repair','parent':parent,'instruction':'Repair the demonstrated error including any repeat-only failure. Keep the intended search quality. Return the full corrected program.'}
        roots={r['lineage'] for r in records.values() if valid_for_archive(r)}
        if not archive or (len(roots)<4 and self.s['root_attempts']<20):
            if self.s['root_attempts']>=20 and not archive:raise RuntimeError('No legal cold start after 20 attempts')
            self.s['root_attempts']+=1
            return {'arm':'independent','parent':None,'instruction':f'Independent design {self.s["root_attempts"]}. Invent a complete optimizer from the specification, without seed code. Aim for high quality near .992. Design both construction and a search that can exploit available time; do not add idle work. Do not prioritize short source over algorithm quality.'}
        slot=self.s['schedule_slot'];self.s['schedule_slot']+=1
        arm=choose_operator(events,slot)
        if arm=='independent':
            return {'arm':arm,'parent':None,'instruction':'Invent a fresh optimizer from the specification only. Prior source is deliberately not supplied. Design useful exploration beyond a single greedy local optimum.'}
        parent=archive[0] if slot%4!=3 else archive[(slot//4)%len(archive)]
        timing=feedback_timing(records[parent])
        recent_runtime=[i for i,e in enumerate(events) if e['arm']=='runtime' and e.get('parent')==parent]
        if timing['underutilized'] and (not recent_runtime or len(events)-recent_runtime[-1]>=8):arm='runtime'
        directions={
            'local':'Address one measured weakness. Up to six coordinated edits across helpers and call sites are allowed. Preserve unrelated working mechanisms.',
            'redesign':'Escape representation and reachable-state limitations. Make a substantive structural redesign while retaining useful parent mechanisms. Return a complete program.',
            'runtime':'Measured runtime leaves useful budget unspent or scales poorly. Diagnose early termination and limits of the reachable states. Design score-improving continuation, not merely larger fixed iteration counts, repeated identical work, padding or sleeps. It is fine to finish early if no useful search exists, but explain this in source comments. Return the complete program.',
            'crossover':'Integrate a genuinely complementary mechanism from the donor. Related parts of the program may change together. Do not just switch whole solvers, average outputs, or route known cases.'}
        task={'arm':arm,'parent':parent,'instruction':directions[arm]}
        if arm=='crossover':
            others=[k for k in archive if k!=parent and records[k]['lineage']!=records[parent]['lineage']]
            if others:
                a={r['name']:r['score'] for r in records[parent]['full']['cases']}
                donor=max(others,key=lambda k:sum(max(0,r['score']-a[r['name']]) for r in records[k]['full']['cases']))
                if any(r['score']>a[r['name']] for r in records[donor]['full']['cases']):task['donor']=donor
            if 'donor' not in task:task.update(arm='redesign',instruction=directions['redesign'])
        return task

    def qualify(self,ident,p):
        record=self.s['records'][ident]
        if not self.measure(ident):
            p['status']='invalid_candidate'
            if p['arm']!='repair':self.s['repair']=ident
            return
        incumbent=self.s['incumbent']
        threshold=self.mean(incumbent) if incumbent else -1
        # Mean-based exploration first, statistical gating only for final holdout selection.
        if record['full']['mean']>=threshold-.0005 or p['parent'] is None:
            if not self.repeat(ident,self.train,'train','repeats'):
                p['status']='invalid_repeat'
                if p['arm']!='repair':self.s['repair']=ident
                return
            if incumbent is None:
                self.s['incumbent']=ident;p['status']='incumbent_promoted';return
            old=self.s['records'][incumbent]
            gains=[a['mean']-b['mean'] for a,b in zip(record['repeats'],old['repeats'])]
            p['confirmation']={'mean_gain':statistics.mean(gains),'repeat_gains':gains,
                               'policy':'mean and repeat-direction gate; no across-case significance required during exploration'}
            if statistics.mean(gains)>.00005 and sum(g>0 for g in gains)>=2:
                self.s['incumbent']=ident;p['status']='incumbent_promoted';return
        p['status']='archived'

    def parse_response(self,kind,parent,content):
        if kind=='edit':code,hypothesis=edit(self.source(parent),content)
        else:code=program(content);hypothesis='\n'.join(code.splitlines()[:8])[:2000]
        return code,hypothesis,{}

    def augment_task(self,pending,task):return task

    def note_response(self,pending,content):pass

    def record_attempt(self,pending):pass

    def run(self):
        if self.s['frozen'] or self.s['validation_started'] or self.s.get('development_closed'):raise ValueError('Development is closed')
        while self.s['pending'] or self.provider.requests<self.provider.max_calls:
            if (self.root/'STOP_AFTER_CURRENT').exists():return
            if not self.s['pending']:
                task=self.task();self.s['pending']={**task,'start_call':self.provider.requests};self.save()
            p=self.s['pending'];parent=p['parent']
            before=self.mean(self.s['incumbent']) if self.s['incumbent'] else 0
            try:
                task={'operation':p['arm'],'instruction':p['instruction']}
                if parent:
                    task.update(parent_source=self.source(parent),parent_sha256=parent,feedback=self.feedback(parent),
                        recent_experiments=[{k:e.get(k) for k in ('arm','status','parent_gain','error')} for e in self.s['events'][-8:]])
                if p.get('donor'):task['donor_source']=self.source(p['donor'])
                task=self.augment_task(p,task)
                kind='edit' if p['arm']=='local' else 'program'
                if 'candidate' not in p:
                    response=self.invoke(kind,task,'implementation_call')
                    self.note_response(p,response['content'])
                    code,hypothesis,metadata=self.parse_response(kind,parent,response['content'])
                    ident=digest(code)
                    if ident in self.s['records'] and any(self.s['records'][ident].get(k)!=v for k,v in metadata.items()):
                        raise ValueError('Conflicting metadata for identical candidate source')
                    p.update(candidate=ident,implementation_kind=kind,**metadata)
                    if ident not in self.s['records']:
                        folder=self.root/'candidates'/ident;folder.mkdir(parents=True,exist_ok=True)
                        (folder/'solution.cpp').write_text(code,encoding='utf-8')
                        self.s['records'][ident]={'parent':parent,'donor':p.get('donor'),
                            'lineage':self.s['records'][parent]['lineage'] if parent else ident,
                            'hypothesis':hypothesis,'implementation_call':response['call_id'],'implementation_kind':kind,**metadata}
                    else:p['duplicate']=True
                    self.save()
                ident=p['candidate']
                if p.get('duplicate'):p['status']='duplicate'
                else:self.qualify(ident,p)
            except BudgetExhausted:p['status']='budget_exhausted';self.save();return
            except (ValueError,RuntimeError) as error:
                p.update(status='failed',error=str(error)[:1500])
                if 'HTTP 401' in str(error) or 'HTTP 402' in str(error):self.save();raise
            record=self.s['records'].get(p.get('candidate'),{})
            p['parent_gain']=(repeated_mean(record)-repeated_mean(self.s['records'][parent])) if parent and valid_for_archive(record) else 0
            p['gain']=(self.mean(self.s['incumbent']) if self.s['incumbent'] else 0)-before
            p['tokens']=sum(r.get('usage',{}).get('total_tokens',r.get('reserved_tokens',0)) for r in self.provider.ledger[p['start_call']:])
            self.record_attempt(p)
            self.s['events'].append(dict(p));self.s['pending']=None;self.save()
            print({'call':self.provider.requests,'arm':p['arm'],'status':p['status'],
                   'incumbent_train_mean':self.mean(self.s['incumbent']) if self.s['incumbent'] else None},flush=True)

    def finalize(self):
        if self.s['frozen']:return self.s['champion']
        if self.s['pending']:
            if self.s['pending'].get('status')!='budget_exhausted':raise ValueError('Pending experiment')
            self.s['events'].append(dict(self.s['pending']));self.s['pending']=None
        if not self.s['incumbent']:raise ValueError('No valid repeated incumbent')
        if not self.s['validation_started']:
            self.s['development_closed']=True;self.save()
            finalists=[self.s['incumbent']]
            for ident in pool(self.s['records']):
                if ident in finalists:continue
                if self.repeat(ident,self.train,'train','repeats'):finalists.append(ident)
                if len(finalists)==3:break
            # Final candidates fixed before any validation result is opened.
            self.s.update(validation_started=True,finalists=finalists);self.save()
        finalists=self.s['finalists'];incumbent=finalists[0]
        valid={k:self.repeat(k,self.validation,'validation','validation_repeats') for k in finalists}
        if not valid[incumbent]:raise RuntimeError('Incumbent invalid on validation; stop without private evaluation')
        decisions={};winner=incumbent
        for candidate in finalists[1:]:
            if not valid[candidate]:continue
            d=paired_decision(self.s['records'][candidate]['validation_repeats'],self.s['records'][incumbent]['validation_repeats'],
                self.config['validation_margin'],alpha=.05/max(1,len(finalists)-1))
            decisions[candidate]=d
            if d['promote'] and (winner==incumbent or d['mean_gain']>decisions[winner]['mean_gain']):winner=candidate
        self.s.update(frozen=True,champion=winner,validation_decisions=decisions);self.save()
        (self.root/'best.cpp').write_text(self.source(winner),encoding='utf-8')
        save_json(self.root/'report.json',{'version':5,'champion':winner,'requests':self.provider.requests,
            'train_mean':self.mean(winner),'validation_mean':statistics.mean(vectors(self.s['records'][winner]['validation_repeats']).values()),
            'private_evaluated':False,'selection':'training repeat-direction gate; conservative paired public validation'})
        return winner
