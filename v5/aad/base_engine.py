"""Evidence-gated cold-start islands. No private data, no reference solver imports."""
import json
import statistics
from pathlib import Path
from .io import digest,read_json,save_json
from .provider import BudgetExhausted
from .search_protocol import plan,program,apply_edit,paired_decision,choose_arm,vectors

ARMS=('local','performance','crossover','redesign')

class V4Engine:
    def __init__(self,root,provider,evaluate,train,validation,config):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'state.json';self.provider=provider;self.evaluate=evaluate
        self.train=train;self.validation=validation;self.config=config
        if len(train)!=40 or len(validation)!=10 or {c.fingerprint for c in train}&{c.fingerprint for c in validation}:
            raise ValueError('Expected disjoint public 40/10 split')
        protocol={'version':4,'config':config,'inputs':{c.name:c.fingerprint for c in [*train,*validation]},
                  'train':[c.name for c in train],'validation':[c.name for c in validation]}
        signature=digest(json.dumps(protocol,sort_keys=True))
        if self.path.exists():
            self.s=read_json(self.path)
            if self.s['signature']!=signature:raise ValueError('Protocol or data changed')
        else:
            self.s={'signature':signature,'records':{},'events':[],'islands':{},'root_attempts':0,
                    'incumbent':None,'pending':None,'repair':None,'validation_started':False,'frozen':False}
        save_json(self.root/'protocol.json',protocol);self.save()

    def save(self):save_json(self.path,self.s)

    def source(self,ident):
        code=(self.root/'candidates'/ident/'solution.cpp').read_text(encoding='utf-8')
        if digest(code)!=ident:raise ValueError('Candidate source changed')
        return code

    def mean(self,ident):return statistics.mean(vectors(self.s['records'][ident]['repeats']).values())

    def feedback(self,ident):
        record=self.s['records'][ident];result=record.get('full',record['smoke'])
        by_name={c.name:c for c in self.train}
        examples=[]
        for row in sorted(result['cases'],key=lambda r:r['score'])[:2]:
            examples.append({'name':row['name'],'score':row['score'],'status':row['status'],
                'message':row.get('message','')[-3000:],'input':by_name[row['name']].text,
                'output':row.get('stdout','')})
        return {'training_mean':result['mean'],'examples':examples,
                'process_seconds_mean':statistics.mean(r.get('seconds',0) for r in result['cases'])}

    def task(self):
        records=self.s['records'];islands=self.s['islands'];events=self.s['events']
        repair=self.s.get('repair')
        if repair:
            self.s['repair']=None
            return {'arm':'repair','parent':repair,'island':records[repair]['island'],
                    'instruction':'Repair only the demonstrated failure. Preserve the algorithmic hypothesis.'}
        need_roots=len(islands)<self.config['initial_valid_roots']
        if not islands or (need_roots and self.s['root_attempts']<self.config['max_initial_attempts']):
            if self.s['root_attempts']>=self.config['max_initial_attempts'] and not islands:
                raise RuntimeError('No valid cold-start program within initial attempt cap')
            self.s['root_attempts']+=1
            return {'arm':'independent','parent':None,'island':None,
                'instruction':f'Independent cold-start attempt {self.s["root_attempts"]}: design the entire algorithm from the problem only. No seed code is supplied. Favor a compact, feasible implementation and explain the dominant bottleneck.'}
        if len(events)%12==0:
            return {'arm':'independent','parent':None,'island':None,
                    'instruction':'Independent restart from the problem specification only. No earlier programs are supplied.'}
        arm=choose_arm(events,ARMS)
        keys=sorted(islands);island=keys[len(events)%len(keys)];parent=islands[island]
        directions={'local':'One focused algorithmic intervention; keep unrelated code unchanged.',
            'performance':'One measured throughput improvement without changing the true objective or time budget.',
            'crossover':'Transfer ONE useful mechanism from the donor, preserving unrelated parent code.',
            'redesign':'Propose a genuinely different search representation or construction mechanism. A compact whole-program rewrite is allowed.'}
        result={'arm':arm,'parent':parent,'island':island,'instruction':directions[arm]}
        others=[v for k,v in islands.items() if k!=island]
        if arm=='crossover' and others:
            a=vectors(records[parent]['repeats'])
            result['donor']=max(others,key=lambda k:sum(max(0,v-a[n]) for n,v in vectors(records[k]['repeats']).items()))
        return result

    def invoke(self,kind,task,slot):
        """Save call ID BEFORE transport; recover a saved successful response on resume."""
        pending=self.s['pending']
        if slot not in pending:
            pending[slot]=self.provider.requests;self.save()
        call=pending[slot]
        if call<self.provider.requests:
            row=self.provider.ledger[call]
            if row['status']!='ok':raise ValueError('Previously reserved request did not complete successfully')
            payload=read_json(self.root/'llm'/f'{call:05d}.response.json')
            return {'call_id':call,'content':payload['choices'][0]['message']['content'],'usage':payload.get('usage',{})}
        return self.provider.request(kind,task)

    def measure(self,ident):
        record=self.s['records'][ident];code=self.source(ident)
        if 'smoke' not in record:
            record['smoke']=self.evaluate(code,self.train[::7],.6,'smoke');self.save()
        if not record['smoke']['valid']:return False
        if 'full' not in record:
            record['full']=self.evaluate(code,self.train,4.8,'train-r0');self.save()
        if not record['full']['valid']:return False
        record.setdefault('repeats',[record['full']]);self.save();return True

    def repeat(self,ident,cases,prefix,key):
        record=self.s['records'][ident];record.setdefault(key,[])
        while len(record[key])<self.config['repeats']:
            i=len(record[key]);record[key].append(self.evaluate(self.source(ident),cases,4.8,f'{prefix}-r{i}'));self.save()
        return all(r['valid'] for r in record[key])

    def run(self):
        if self.s['frozen'] or self.s['validation_started']:raise ValueError('Development is closed')
        while self.s['pending'] or self.provider.requests<self.provider.max_calls:
            if (self.root/'STOP_AFTER_CURRENT').exists():return
            if not self.s['pending']:
                task=self.task();self.s['pending']={**task,'start_call':self.provider.requests};self.save()
            p=self.s['pending'];parent=p['parent'];before=self.mean(self.s['incumbent']) if self.s['incumbent'] else 0
            try:
                context={'operation':p['arm'],'instruction':p['instruction']}
                if parent:
                    context.update(parent_source=self.source(parent),parent_sha256=parent,feedback=self.feedback(parent),
                        recent_outcomes=[{k:e.get(k) for k in ('arm','status','gain','error')} for e in self.s['events'][-6:]])
                if p.get('donor'):context['donor_source']=self.source(p['donor'])
                if 'plan' not in p:
                    response=self.invoke('plan',context,'plan_call');p['plan']=plan(response['content']);self.save()
                kind='program' if p['arm'] in ('independent','redesign') else 'edit'
                if 'candidate' not in p:
                    response=self.invoke(kind,{**context,'design':p['plan']},'implementation_call')
                    code=program(response['content']) if kind=='program' else apply_edit(self.source(parent),response['content'])
                    ident=digest(code);p.update(candidate=ident,implementation_kind=kind)
                    folder=self.root/'candidates'/ident;folder.mkdir(parents=True,exist_ok=True)
                    if ident not in self.s['records']:
                        (folder/'solution.cpp').write_text(code,encoding='utf-8')
                        self.s['records'][ident]={'plan':p['plan'],'island':p['island'] or ident,
                            'parent':parent,'donor':p.get('donor'),'plan_call':p['plan_call'],
                            'implementation_call':p['implementation_call'],'implementation_kind':kind}
                    else:p['duplicate']=True
                    self.save()
                ident=p['candidate'];record=self.s['records'][ident]
                if p.get('duplicate'):p['status']='duplicate'
                elif not self.measure(ident):
                    p['status']='invalid_candidate'
                    if p['arm']!='repair':self.s['repair']=ident
                else:
                    island=record['island'];old=self.s['islands'].get(island)
                    promising=not old or record['full']['mean']>=self.mean(old)+self.config['margin']/2 or len(self.s['events'])%5==0
                    p['status']='evaluated_not_promoted'
                    if promising and self.repeat(ident,self.train,'train','repeats'):
                        decision=None
                        if old:
                            self.repeat(old,self.train,'train','repeats')
                            decision=paired_decision(record['repeats'],self.s['records'][old]['repeats'],self.config['margin'])
                            p['island_decision']=decision
                        if not old or decision['promote']:
                            self.s['islands'][island]=ident;p['status']='island_promoted'
                        best=self.s['incumbent']
                        if best is None:self.s['incumbent']=ident;p['status']='incumbent_promoted'
                        elif best!=ident:
                            decision=paired_decision(record['repeats'],self.s['records'][best]['repeats'],self.config['margin'])
                            p['global_decision']=decision
                            if decision['promote']:self.s['incumbent']=ident;p['status']='incumbent_promoted'
            except BudgetExhausted:
                p['status']='budget_exhausted';self.save();return
            except (ValueError,RuntimeError) as error:
                p.update(status='failed',error=str(error)[:1000])
                if 'HTTP 401' in str(error) or 'HTTP 402' in str(error):self.save();raise
            after=self.mean(self.s['incumbent']) if self.s['incumbent'] else 0
            p['gain']=after-before;p['reward']=min(10,max(0,p['gain'])*1000)
            p['tokens']=sum(r.get('usage',{}).get('total_tokens',r.get('reserved_tokens',0)) for r in self.provider.ledger[p['start_call']:])
            self.s['events'].append(dict(p));self.s['pending']=None;self.save()
            print({'attempt':len(self.s['events']),'calls':self.provider.requests,'status':p['status'],
                   'islands':len(self.s['islands']),'incumbent_train_mean':after},flush=True)

    def finalize(self):
        if self.s['frozen']:return self.s['champion']
        if self.s['pending']:
            if self.s['pending'].get('status')!='budget_exhausted':raise ValueError('Finish pending experiment before validation')
            self.s['events'].append(dict(self.s['pending']));self.s['pending']=None
        if not self.s['incumbent']:raise ValueError('No qualified incumbent')
        if not self.s['validation_started']:
            incumbent=self.s['incumbent'];others=sorted(set(self.s['islands'].values())-{incumbent},key=self.mean,reverse=True)
            self.s.update(validation_started=True,finalists=[incumbent,*others[:2]]);self.save()
        finalists=self.s['finalists'];incumbent=finalists[0]
        valid={k:self.repeat(k,self.validation,'validation','validation_repeats') for k in finalists}
        if not valid[incumbent]:raise RuntimeError('Incumbent failed validation; do not silently substitute')
        decisions={};winner=incumbent
        for candidate in finalists[1:]:
            if not valid[candidate]:continue
            d=paired_decision(self.s['records'][candidate]['validation_repeats'],
                self.s['records'][incumbent]['validation_repeats'],self.config['validation_margin'],
                alpha=.05/max(1,len(finalists)-1))
            decisions[candidate]=d
            if d['promote'] and (winner==incumbent or d['mean_gain']>decisions[winner]['mean_gain']):winner=candidate
        self.s.update(frozen=True,champion=winner,validation_decisions=decisions);self.save()
        (self.root/'best.cpp').write_text(self.source(winner),encoding='utf-8')
        save_json(self.root/'report.json',{'version':4,'champion':winner,'requests':self.provider.requests,
            'train_mean':self.mean(winner),'validation_mean':statistics.mean(vectors(self.s['records'][winner]['validation_repeats']).values()),
            'private_evaluated':False,'selection':'paired repeated validation; incumbent retained for inconclusive differences'})
        return winner
