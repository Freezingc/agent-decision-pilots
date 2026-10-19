"""Trusted reference computation with per-candidate official EvalPlus kernels.

Candidate sandboxes receive only one source and a single suite's input/output
values. They do not mount the dataset, reference source, result directory or
project files. The trusted caller alone writes persistent evaluation results.
"""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

from .execution import sandbox_command


def make_payload(source, entry_point, suite, atol):
    return {'source': source, 'entry_point': entry_point, 'inputs': suite['inputs'],
            'expected': suite['expected'], 'ref_time': suite['ref_time'], 'atol': atol}


def _kernel_limits():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (75, 76))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def evaluate_candidate(source, entry_point, suites, atol=0):
    base_prefix = str(Path(sys.base_prefix).resolve())
    venv_prefix = str(Path(sys.prefix).resolve())
    worker = str(Path(__file__).with_name('_evalplus_worker.py').resolve())
    interpreter = str(Path(sys.executable).absolute())
    cmd = sandbox_command(
        readonly={base_prefix: base_prefix, venv_prefix: venv_prefix, worker: '/work/worker.py'},
        python=interpreter, arguments=['-I', '/work/worker.py'],
        extra_env={'EVALPLUS_MAX_MEMORY_BYTES': str(4 * 1024**3)})
    result = {'solution': source, 'entry_point': entry_point}
    for name in ['base', 'plus']:
        started = time.monotonic()
        payload = make_payload(source, entry_point, suites[name], atol)
        try:
            proc = subprocess.run(cmd, input=pickle.dumps(payload, protocol=4), capture_output=True,
                                  timeout=80, preexec_fn=_kernel_limits)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'official evaluation sandbox infrastructure failure: {exc}') from exc
        if proc.returncode:
            raise RuntimeError(f'official evaluation sandbox exited {proc.returncode}: {proc.stderr.decode(errors="replace")[-2000:]}')
        try:
            outcome = json.loads(proc.stdout)
            if outcome['status'] not in {'pass', 'fail', 'timeout'}:
                raise ValueError('unrecognized official status')
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError('invalid official kernel receipt') from exc
        result[f'{name}_status'] = outcome['status']
        result[f'{name}_details'] = outcome['details']
        result[f'{name}_seconds'] = time.monotonic() - started
    return result


def reference_suites(problem):
    """Compute official oracle outputs only in the trusted evaluation stage."""
    from evalplus.gen.util import trusted_exec
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    source = problem['prompt'] + problem['canonical_solution']
    suites = {}
    for name in ['base', 'plus']:
        inputs = deepcopy(problem[f'{name}_input'])
        expected, ref_time = trusted_exec(
            source, inputs, problem['entry_point'], record_time=True,
            output_not_none=problem['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS)
        suites[name] = {'inputs': inputs, 'expected': expected, 'ref_time': ref_time}
    return suites
