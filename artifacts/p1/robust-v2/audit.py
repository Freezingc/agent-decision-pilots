"""Read-only arithmetic audit of the frozen robust-v2 simulation artifacts.

Run from project root with PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B
artifacts/p1/robust-v2/audit.py. Prints evidence; never changes run files.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
from agent_pilots import recovery_robust as q

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
for name,item in json.loads((root/'manifest.json').read_text()).items():
    path=root/name
    if path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('manifest mismatch: '+name)
for path,digest in json.loads((root/'sources-before-run.json').read_text())['sources'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise ValueError('source changed: '+path)

def require(value,message):
    if not value: raise ValueError(message)

models_by_name=dict(q.parameter_sets());actuals={}
for name,models in models_by_name.items():
    for regime,p in (('nominal',models[0]),('long',models[1]),('outside',q.replace(models[0],delay=q.DELAY_OUTSIDE))):
        actuals[name+'_'+regime]=p
groups=collections.defaultdict(list);steps=0;gates=0;max_actual={}
with gzip.open(root/'test-trajectories.jsonl.gz','rt') as handle:
    for line in handle:
        row=json.loads(line);condition=row['condition'];method=row['method'];actual=actuals[condition]
        models=models_by_name[condition.rsplit('_',1)[0]]
        active=(models[0],) if method=='nominal_gated_h4' else models
        b=q.ModelBeliefs.initial(active);true_b=q.v1.initial_belief(actual);state=row['initial_state'];cost=0.;retries=0
        require(len(row['trajectory'])<=13,'horizon')
        require(state==list(q.v1._draw(list(q.v1.initial_belief(actual).items()),q.v1._uniform(row['seed'],-1,'initial'))),'initial draw')
        for step in row['trajectory']:
            a=step['action'];require(step['state_before']==state,'state chain')
            require(math.isclose(step['worst_executed'],b.risk('executed'),abs_tol=1e-10),'risk executed')
            require(math.isclose(step['worst_missing'],b.risk('missing'),abs_tol=1e-10),'risk missing')
            require(len(step['model_weights'])==len(b.weights),'weights length')
            require(all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(step['model_weights'],b.weights)),'model weights')
            guarded=method in ('robust_h4','nominal_gated_h4') or method.startswith('posterior_')
            if guarded and a=='retry':require(q.retry_allowed(b,retries),'retry risk gate');gates+=1
            if guarded and a=='accept':require(q.accept_allowed(b),'accept risk gate');gates+=1
            if method=='robust_h4':
                regime=condition.rsplit('_',1)[1]
                risk=max_actual.setdefault(regime,{'retry_executed':0.,'accept_missing':0.})
                if a=='retry':risk['retry_executed']=max(risk['retry_executed'],sum(p for s,p in true_b.items() if s[0]>0))
                if a=='accept':risk['accept_missing']=max(risk['accept_missing'],sum(p for s,p in true_b.items() if s[0]==0))
            expected=q.v1.action_cost(a,actual);require(math.isclose(step['cost'],expected,abs_tol=1e-10),'cost');cost+=expected
            if a not in q.v1.TERMINAL:
                sampled=q.v1._draw([((s,o),p) for s,o,p in q.v1.transition(tuple(state),a,actual)],q.v1._uniform(row['seed'],step['turn'],a))
                require((step['state_after'],step['observation'])==(list(sampled[0]),sampled[1]),'counterfactual draw')
                b=b.update(a,step['observation']);true_b,_=q.v1.update_belief(true_b,a,step['observation'],actual);steps+=1
                for post,saved in zip(b.beliefs,step['belief_after']):
                    saved={tuple(s):p for s,p in saved}
                    require(set(post)==set(saved) and all(math.isclose(p,saved[s],abs_tol=1e-10) for s,p in post.items()),'Bayes posterior')
                    require(math.isclose(sum(post.values()),1,abs_tol=1e-10),'posterior mass')
            else:require(step['state_after']==state,'terminal state changed')
            state=step['state_after'];retries+=int(a=='retry')
        require(row['terminal_action']==row['trajectory'][-1]['action'],'terminal action')
        require(row['terminal_state']==state,'terminal state')
        require(math.isclose(cost,row['cost'],abs_tol=1e-10),'total cost')
        require(all(row[k]==v for k,v in q.v1.outcome(tuple(state),row['terminal_action']).items()),'outcome')
        require(math.isclose(row['loss'],cost+q.v1.terminal_loss(tuple(state),row['terminal_action'],actual),abs_tol=1e-10),'loss')
        if method!='point_h4':require(retries<=1,'retry budget')
        groups[condition,method].append(row)
require(sum(map(len,groups.values()))==50400,'episode count')
for row in summary['summary']:
    records=groups[row['condition'],row['method']];require(len(records)==row['n']==600,'group count')
    for key in ('success','duplicate','false_completion','abstention','unconfirmed_single','cost','loss','checks','waits','retries'):
        require(math.isclose(sum(r[key] for r in records)/600,row[key],abs_tol=1e-10),'summary '+key)
    for key,value in row['counts'].items():require(sum(r[key] for r in records)==value,'counts')
for condition in actuals:
    sets=[rows for (c,m),rows in groups.items() if c==condition]
    for i in range(600):
        require(len({rows[i]['seed'] for rows in sets})==1,'paired seed')
        require(len({tuple(rows[i]['initial_state']) for rows in sets})==1,'paired initial state')
for regime in ('nominal','long'):
    require(max_actual[regime]['retry_executed']<=q.DUP_LIMIT+1e-10,'true in-set retry risk')
    require(max_actual[regime]['accept_missing']<=q.FALSE_LIMIT+1e-10,'true in-set acceptance risk')
print(json.dumps({'status':'matched','episodes':50400,'groups':len(groups),'Bayes_steps':steps,'gated_decisions':gates,'max_true_conditional_risks_by_regime':max_actual},indent=2))
