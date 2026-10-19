"""One thinking-enabled API request per program/edit, with the existing strict ledger."""
import json
import time
import urllib.request
import urllib.error
from .base_provider import V4Provider
from .layout_prompt import PROBLEM
from .provider import BudgetExhausted
from .io import save_json,digest

class V5Provider(V4Provider):
    def request_body(self,kind,task):
        if kind not in ('program','edit'):raise ValueError('Unknown v5 request kind')
        rule=('Return only complete executable C++17 source, NOT JSON. Start with a concise comment describing the hypothesis and changed mechanism, at most eight lines. End with // AAD_V5_COMPLETE.' if kind=='program' else
              'Return only JSON with hypothesis (short string), parent_sha256, and edits (1..6 objects each containing old and new source strings). Anchors must be unique, disjoint substrings of the ORIGINAL parent. Related declarations, helpers and call sites may be changed together. Keep unrelated code unchanged. Preserve // AAD_V5_COMPLETE.')
        system=('You are an optimization algorithm researcher. Invent from the problem and this run only; never reconstruct named submissions. '
                'Reason about the actual search and feasible states before implementation. Do not sacrifice algorithm quality merely to minimize source size. '+rule)
        body={'model':self.model,'max_tokens':65536,'thinking':{'type':'enabled'},'reasoning_effort':'low',
              'messages':[{'role':'system','content':system},
                          {'role':'user','content':json.dumps({'problem':PROBLEM,**task},ensure_ascii=False)}]}
        if kind=='edit':body['response_format']={'type':'json_object'}
        return body

    def request(self,kind,task):
        body=self.request_body(kind,task)
        reserve=len(json.dumps(body).encode('utf-8'))+1024+65536
        if self.requests>=self.max_calls or (self.max_tokens_total>0 and self.charged_tokens+reserve>self.max_tokens_total):
            raise BudgetExhausted('v5 budget exhausted before HTTP')
        call=self.requests;save_json(self.folder/f'{call:05d}.request.json',body)
        row={'id':call,'kind':kind,'status':'started','reserved_tokens':reserve,'requested_model':self.model}
        self.ledger.append(row);save_json(self.path,self.ledger);start=time.monotonic()
        request=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        try:
            with urllib.request.urlopen(request,timeout=600) as response:raw=response.read(2_000_001)
            if len(raw)>2_000_000:raise ValueError('Response too large')
            payload=json.loads(raw);save_json(self.folder/f'{call:05d}.response.json',payload)
            row.update(usage=payload.get('usage',{}),response_model=payload.get('model'),system_fingerprint=payload.get('system_fingerprint'))
            choice=payload['choices'][0];row['finish_reason']=choice.get('finish_reason')
            if choice.get('finish_reason')!='stop':raise ValueError('Incomplete response: '+str(choice.get('finish_reason')))
            content=choice['message'].get('content')
            if not isinstance(content,str) or not content.strip():raise ValueError('Empty final response')
            row.update(status='ok',content_sha256=digest(content));return {'call_id':call,'content':content,'usage':row['usage']}
        except urllib.error.HTTPError as error:
            row['status']=f'http_{error.code}';raise RuntimeError(f'HTTP {error.code}') from None
        except (urllib.error.URLError,TimeoutError):
            row['status']='network_error';raise RuntimeError('Network failure; usage unknown') from None
        except (ValueError,TypeError,KeyError,IndexError) as error:
            row['status']='invalid_response';raise ValueError(str(error)) from None
        finally:
            row['seconds']=time.monotonic()-start;save_json(self.path,self.ledger)
