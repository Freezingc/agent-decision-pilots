"""Independent P1 robust-v2 synthetic screen; no algorithm-novelty claim.

Each declared model conditions on the identical visible history. Worst-case
risk gates retain every model with positive likelihood, regardless of its Bayes
model weight. The finite-tree planner uses rectangular max conditional values:
conservative for a fixed unknown model, not its exact nonrectangular optimum.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from . import recovery as v1

DUP_LIMIT=.01
FALSE_LIMIT=.05
HORIZON=12
DELAY_NOMINAL=((0,.25),(1,.25),(3,.5))
DELAY_LONG=((0,.1),(3,.2),(6,.7))
DELAY_OUTSIDE=((0,.05),(6,.15),(10,.8))


@dataclass
class ModelBeliefs:
    models: tuple
    beliefs: tuple
    weights: tuple

    @classmethod
    def initial(cls,models):
        return cls(tuple(models),tuple(v1.initial_belief(p) for p in models),tuple(1/len(models) for _ in models))

    def update(self,action,observation):
        posts=[]; unnormalized=[]
        for model,b,w in zip(self.models,self.beliefs,self.weights):
            joint=v1.predict_observations(b,action,model).get(observation,{})
            likelihood=sum(joint.values())
            posts.append({s:p/likelihood for s,p in joint.items()} if likelihood else {})
            unnormalized.append(w*likelihood)
        total=sum(unnormalized)
        if not total: raise ValueError('history impossible in every declared model')
        return ModelBeliefs(self.models,tuple(posts),tuple(w/total for w in unnormalized))

    def risk(self,event):
        pred={'executed':lambda s:s[0]>0,'missing':lambda s:s[0]==0,'duplicate':lambda s:s[0]>=2}[event]
        return max(sum(p for s,p in b.items() if pred(s)) for b in self.beliefs if b)

    def key(self): return tuple(v1._belief_key(b) for b in self.beliefs)


def retry_allowed(b,retries): return retries==0 and b.risk('executed')<=DUP_LIMIT+1e-12
def accept_allowed(b): return b.risk('missing')<=FALSE_LIMIT+1e-12


class RobustPlanner:
    def __init__(self,models,lookahead=4):
        self.models=tuple(models);self.lookahead=lookahead
        self._cached=lru_cache(maxsize=250000)(self._value)

    def _value(self,key,horizon,retries):
        b=ModelBeliefs(self.models,tuple(dict(k) for k in key),tuple(1/len(key) for _ in key))
        active=[i for i,x in enumerate(b.beliefs) if x]
        candidates=[(max(sum(prob*v1.terminal_loss(s,'abandon',self.models[i]) for s,prob in b.beliefs[i].items()) for i in active),'abandon')]
        if accept_allowed(b):
            candidates.append((max(sum(prob*v1.terminal_loss(s,'accept',self.models[i]) for s,prob in b.beliefs[i].items()) for i in active),'accept'))
        if horizon:
            for action in v1.NONTERMINAL:
                if action=='retry' and not retry_allowed(b,retries): continue
                distributions=[v1.predict_observations(b.beliefs[i],action,self.models[i]) for i in active]
                observations=set().union(*(set(x) for x in distributions))
                child_values={o:self._cached(b.update(action,o).key(),horizon-1,retries+int(action=='retry'))[0] for o in observations}
                value=max(v1.action_cost(action,self.models[i])+sum(sum(bucket.values())*child_values[o] for o,bucket in dist.items()) for i,dist in zip(active,distributions))
                candidates.append((value,action))
        order=('accept','abandon','wait','check','retry')
        return min(candidates,key=lambda x:(round(x[0],10),order.index(x[1])))

    def value(self,b,horizon,retries): return self._cached(b.key(),horizon,retries)
    def choose(self,b,history,remaining):
        return self.value(b,min(remaining,self.lookahead),sum(x['action']=='retry' for x in history))[1]


@dataclass(frozen=True)
class PosteriorRule:
    check_every:int=1
    abandon_after:int=12

    def choose(self,b,history,remaining):
        if accept_allowed(b): return 'accept'
        if not remaining or len(history)>=self.abandon_after: return 'abandon'
        if retry_allowed(b,sum(x['action']=='retry' for x in history)): return 'retry'
        return 'check' if len(history)%self.check_every==0 else 'wait'


@dataclass(frozen=True)
class FixedRule:
    wait:int=0
    check_every:int=1
    negatives:int=1
    direct_accept:bool=False
    abandon_after:int=12
    retry_limit:int=1
    initial_terminal:str=''

    def choose(self,history,remaining):
        if not history and self.initial_terminal: return self.initial_terminal
        if history and history[-1]['observation']=='present': return 'accept'
        if history and history[-1]['action']=='retry' and self.direct_accept: return 'accept'
        if not remaining or len(history)>=self.abandon_after: return 'abandon'
        retries=sum(x['action']=='retry' for x in history)
        index=max((i for i,x in enumerate(history) if x['action']=='retry'),default=-1)
        recent=history[index+1:]
        if len(recent)<self.wait: return 'wait'
        negatives=sum(x['action']=='check' and x['observation']=='absent' for x in recent)
        if negatives>=self.negatives and retries<self.retry_limit: return 'retry'
        return 'check' if (len(recent)-self.wait)%self.check_every==0 else 'wait'


def simulate(actual,policy,seed,models,retain_trace=True):
    state=v1._draw(list(v1.initial_belief(actual).items()),v1._uniform(seed,-1,'initial'))
    initial=list(state);b=ModelBeliefs.initial(models);history=[];trace=[];cost=0.
    track=retain_trace or not isinstance(policy,FixedRule)
    for turn in range(HORIZON+1):
        remaining=HORIZON-turn
        if isinstance(policy,RobustPlanner): action=policy.choose(b,history,remaining)
        elif isinstance(policy,PosteriorRule): action=policy.choose(b,history,remaining)
        elif isinstance(policy,v1.BeliefPlanner): action=policy.choose(b.beliefs[0],remaining)
        else: action=policy.choose(history,remaining)
        before=state
        item={'turn':turn,'action':action,'state_before':list(before)}
        if retain_trace:
            item.update(worst_executed=b.risk('executed'),worst_missing=b.risk('missing'),model_weights=list(b.weights))
        if action in v1.TERMINAL:
            if retain_trace: trace.append({**item,'observation':'terminal','state_after':list(state),'cost':0.})
            break
        if not remaining: raise ValueError('policy exceeds horizon')
        state,observation=v1._draw([((s,o),p) for s,o,p in v1.transition(state,action,actual)],v1._uniform(seed,turn,action))
        step=v1.action_cost(action,actual);cost+=step
        history.append({'action':action,'observation':observation})
        if track: b=b.update(action,observation)
        if retain_trace:
            trace.append({**item,'observation':observation,'state_after':list(state),'cost':step,
                          'belief_after':[[[list(s),p] for s,p in sorted(post.items())] for post in b.beliefs]})
    else: raise ValueError('unterminated')
    return {'seed':seed,'initial_state':initial,'terminal_state':list(state),'terminal_action':action,
            **v1.outcome(state,action),'cost':cost,'loss':cost+v1.terminal_loss(state,action,actual),
            'checks':sum(x['action']=='check' for x in history),'waits':sum(x['action']=='wait' for x in history),
            'retries':sum(x['action']=='retry' for x in history),'trajectory':trace}


def parameter_sets():
    for fault,pnot,stale in (('low',.1,.05),('medium',.35,.2)):
        for cost in (.1,1.):
            p=v1.Parameters(not_executed=pnot,stale_read=stale,check_cost=cost,delay=DELAY_NOMINAL)
            yield f'{fault}-fault_check-{cost:g}',(p,replace(p,delay=DELAY_LONG))


def paired(a,b):
    result={}
    for key in ('success','duplicate','false_completion','abstention','cost','loss'):
        values=[x[key]-y[key] for x,y in zip(a,b,strict=True)]
        mean=sum(values)/len(values);var=sum((v-mean)**2 for v in values)/(len(values)-1)
        half=1.96*math.sqrt(var/len(values))
        result[key]={'difference':mean,'normal_ci95':[mean-half,mean+half]}
    return result


def _select(results,target):
    key=lambda x:(x['metrics']['loss'],x['metrics']['cost'],json.dumps(x['rule'],sort_keys=True))
    best=min(results,key=key)
    qualified=[x for x in results if x['metrics']['success']>=target['success']-.01-1e-12
               and x['metrics']['duplicate']<=target['duplicate']+.01+1e-12
               and x['metrics']['false_completion']<=target['false_completion']+.01+1e-12]
    quality=min(qualified,key=lambda x:(x['metrics']['cost'],x['metrics']['loss'],json.dumps(x['rule'],sort_keys=True))) if qualified else None
    return {'loss_selected':best,'quality_selected':quality,'quality_fallback_to_loss':quality is None,'grid':results}


def run(output,dev_n=200,test_n=600,seed=2026101902):
    output=Path(output)
    protocol=json.loads((output/'frozen-protocol.json').read_text())
    if seed!=protocol['seed'] or dev_n!=protocol['dev_n_per_model'] or test_n!=protocol['test_n_per_condition']:
        raise ValueError('CLI disagrees with frozen protocol')
    if (output/'summary.json').exists(): raise ValueError('completed evidence cannot be overwritten')
    started=datetime.now(timezone.utc).isoformat();clock=time.monotonic()
    source_paths=[Path(__file__).resolve(),Path(v1.__file__).resolve(),Path('tests/test_recovery_robust.py').resolve()]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for p in source_paths: (output/('source-'+p.name)).write_bytes(p.read_bytes())
    (output/'sources-before-run.json').write_text(json.dumps({'utc':started,'sources':sources,'protocol_sha256':hashlib.sha256((output/'frozen-protocol.json').read_bytes()).hexdigest()},indent=2)+'\n')
    dev_seeds=[seed*1000000+i for i in range(dev_n)]
    test_seeds=[seed*1000000+100000+block*10000+i for block in range(3) for i in range(test_n//3)]
    if len(test_seeds)!=test_n: raise ValueError('test_n not divisible by 3')
    fixed=[FixedRule(w,e,n,d,a,r) for w,e,n,d,a,r in itertools.product((0,3,5),(1,2),(1,3,4),(False,True),(4,8,12),(0,1))]
    fixed.extend(FixedRule(initial_terminal=a) for a in v1.TERMINAL)
    thresholds=[PosteriorRule(e,a) for e,a in itertools.product((1,2,3),(4,8,12))]
    dev={};planners={}
    for name,models in parameter_sets():
        planner=RobustPlanner(models);planners[name]=planner
        def metrics(policy): return v1.aggregate([simulate(p,policy,s,models,False) for p in models for s in dev_seeds])
        target=metrics(planner)
        dev[name]={'robust':target,'fixed':_select([{'rule':asdict(r),'metrics':metrics(r)} for r in fixed],target),
                   'posterior':_select([{'rule':asdict(r),'metrics':metrics(r)} for r in thresholds],target)}
        print(json.dumps({'dev_complete':name,'robust':target,'fixed_loss':dev[name]['fixed']['loss_selected']['rule'],
                          'fixed_quality':None if dev[name]['fixed']['quality_selected'] is None else dev[name]['fixed']['quality_selected']['rule']}),flush=True)
    (output/'dev-selection.json').write_text(json.dumps(dev,indent=2)+'\n')
    dev_frozen=datetime.now(timezone.utc).isoformat()
    summaries=[];comparisons=[];blocks=[]
    with gzip.open(output/'test-trajectories.jsonl.gz','wt') as stream:
        for name,models in parameter_sets():
            policies={'point_h4':v1.BeliefPlanner(models[0],4),'robust_h4':planners[name]}
            for family,cls in (('fixed',FixedRule),('posterior',PosteriorRule)):
                selected=dev[name][family]
                policies[family+'_dev_loss']=cls(**selected['loss_selected']['rule'])
                policies[family+'_dev_quality_or_loss_fallback']=cls(**(selected['quality_selected'] or selected['loss_selected'])['rule'])
            for regime,actual in (('nominal',models[0]),('long',models[1]),('outside',replace(models[0],delay=DELAY_OUTSIDE))):
                condition=name+'_'+regime;records={}
                for method,policy in policies.items():
                    group=[]
                    for s in test_seeds:
                        row=simulate(actual,policy,s,models)
                        stream.write(json.dumps({'condition':condition,'method':method,**row},separators=(',',':'))+'\n')
                        group.append(row)
                    records[method]=group
                    summaries.append({'condition':condition,'method':method,'regime':regime,**v1.aggregate(group)})
                    for block in range(3):
                        part=group[block*(test_n//3):(block+1)*(test_n//3)]
                        blocks.append({'condition':condition,'method':method,'block':block,**v1.aggregate(part)})
                for method in policies:
                    if method!='robust_h4': comparisons.append({'condition':condition,'left':'robust_h4','right':method,'n':test_n,'differences':paired(records['robust_h4'],records[method])})
                print(json.dumps({'test_complete':condition,'robust':v1.aggregate(records['robust_h4']),'posterior':v1.aggregate(records['posterior_dev_loss'])}),flush=True)
    ended=datetime.now(timezone.utc).isoformat()
    report={'schema':'p1-robust-v2','status':'completed','evidence':'synthetic only','started_utc':started,'dev_frozen_utc':dev_frozen,'ended_utc':ended,
            'elapsed_seconds':time.monotonic()-clock,'seed':seed,'dev_n_per_model':dev_n,'test_n':test_n,'test_episodes':12*len(policies)*test_n,
            'command':sys.argv,'python':sys.version,'gpu_used':False,'paid_api_calls':0,'paid_api_cost':0,
            'dev_selection_sha256':hashlib.sha256((output/'dev-selection.json').read_bytes()).hexdigest(),
            'summary':summaries,'seed_blocks':blocks,'paired':comparisons,
            'limits':['conditional guarantees only when true fixed model is in declared set and all other parameters correct',
                      'rectangular robust planning is a conservative heuristic; not exact fixed-model minimax',
                      'model posterior weights diagnostic only; no model dropped except zero likelihood',
                      'normal CIs descriptive; no rare-risk certification; quality-matched dev selection is not test noninferiority',
                      'ordinary minimax and posterior thresholds are not algorithm innovation']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['condition','method','regime','n','success','duplicate','false_completion','abstention','unconfirmed_single','cost','loss','checks','waits','retries']
    with (output/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in summaries)
    evidence={}
    for p in output.iterdir():
        if p.is_file() and p.name not in ('manifest.json','run.log'): evidence[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/p1/robust-v2'))
    p.add_argument('--dev-n',type=int,default=200);p.add_argument('--test-n',type=int,default=600);p.add_argument('--seed',type=int,default=2026101902)
    a=p.parse_args();run(a.output,a.dev_n,a.test_n,a.seed)


if __name__=='__main__':main()
