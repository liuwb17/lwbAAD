"""Split design/implementation transport; count every HTTP attempt before sending."""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from .io import read_json,save_json,digest
from .provider import environment,BudgetExhausted
from .layout_prompt import PROBLEM

class V4Provider:
    def __init__(self,root,model,max_calls,max_tokens_total,key=None):
        if max_calls<1 or max_tokens_total<0:raise ValueError('Invalid request/token limits')
        self.root=Path(root);self.folder=self.root/'llm';self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'ledger.json';self.ledger=read_json(self.path) if self.path.exists() else []
        for row in self.ledger:
            if row['status']=='started':row['status']='interrupted_usage_unknown'
        save_json(self.path,self.ledger)
        self.model=model;self.max_calls=max_calls;self.max_tokens_total=max_tokens_total
        self.key=key or environment('AAD_API_KEY')
        if not self.key:raise ValueError('AAD_API_KEY is required')

    @property
    def requests(self):return len(self.ledger)

    @property
    def charged_tokens(self):
        return sum(r.get('usage',{}).get('total_tokens',r['reserved_tokens']) for r in self.ledger)

    def request(self,kind,task):
        caps={'plan':65536,'program':24576,'edit':12288}
        rules={
            'plan':'Return only JSON with short strings hypothesis, mechanism, expected_effect, risk, test. Describe ONE falsifiable algorithmic change or independent design, not source code. Keep the entire plan under 1200 words.',
            'program':'Return only complete C++17 source, not JSON. End the source with the literal comment // AAD_V4_COMPLETE. Implement the supplied design succinctly. Never omit code or use ellipses.',
            'edit':'Return only JSON {"parent_sha256":"...","edits":[{"old":"exact unique source text","new":"replacement"}]}. Exactly ONE contiguous edit. Preserve all unrelated code. Do not return the whole program. Existing completion sentinel must remain.'}
        if kind not in caps:raise ValueError('Unknown request kind')
        system='You design algorithms from the task and current experiment only. Do not reconstruct named submissions. '+rules[kind]
        body={'model':self.model,'max_tokens':caps[kind],
              'thinking':{'type':'enabled' if kind=='plan' else 'disabled'},
              'messages':[{'role':'system','content':system},
                          {'role':'user','content':json.dumps({'problem':PROBLEM,**task},ensure_ascii=False)}]}
        if kind=='plan':body['reasoning_effort']='low'
        else:body['temperature']=.7
        if kind!='program':body['response_format']={'type':'json_object'}
        # UTF-8 bytes + framing is a conservative accounting reserve, not a tokenizer.
        reserve=len(json.dumps(body).encode('utf-8'))+1024+caps[kind]
        if self.requests>=self.max_calls or (self.max_tokens_total>0 and self.charged_tokens+reserve>self.max_tokens_total):
            raise BudgetExhausted('v4 request/token budget exhausted before HTTP')
        call=self.requests;save_json(self.folder/f'{call:05d}.request.json',body)
        row={'id':call,'kind':kind,'status':'started','reserved_tokens':reserve,'requested_model':self.model}
        self.ledger.append(row);save_json(self.path,self.ledger);start=time.monotonic()
        request=urllib.request.Request('https://api.deepseek.com/chat/completions',
            data=json.dumps(body).encode('utf-8'),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        try:
            with urllib.request.urlopen(request,timeout=600) as response:raw=response.read(2_000_001)
            if len(raw)>2_000_000:raise ValueError('Response too large')
            payload=json.loads(raw);save_json(self.folder/f'{call:05d}.response.json',payload)
            row.update(usage=payload.get('usage',{}),response_model=payload.get('model'),system_fingerprint=payload.get('system_fingerprint'))
            choice=payload['choices'][0];row['finish_reason']=choice.get('finish_reason')
            if choice.get('finish_reason')!='stop':raise ValueError('Incomplete response: '+str(choice.get('finish_reason')))
            content=choice['message'].get('content')
            if not isinstance(content,str) or not content.strip():raise ValueError('Empty final response')
            row.update(status='ok',content_sha256=digest(content))
            return {'call_id':call,'content':content,'usage':row['usage']}
        except urllib.error.HTTPError as error:
            row['status']=f'http_{error.code}';raise RuntimeError(f'HTTP {error.code}') from None
        except (urllib.error.URLError,TimeoutError):
            row['status']='network_error';raise RuntimeError('Network failure; billed usage unknown') from None
        except (ValueError,KeyError,IndexError,TypeError) as error:
            row['status']='invalid_response';raise ValueError(str(error)) from None
        finally:
            row['seconds']=time.monotonic()-start;save_json(self.path,self.ledger)
