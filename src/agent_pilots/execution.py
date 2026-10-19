"""Parse visible tests and execute generated code inside a Linux sandbox."""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import time


def extract_source(response: str) -> dict:
    matches = list(re.finditer(r'```(?:python|py)?[ \t]*\r?\n(.*?)```', response, re.S))
    source = matches[0].group(1) if matches else response
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    try:
        ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return {'source': source, 'source_hash': source_hash, 'status': 'syntax_error', 'error': str(exc)}
    return {'source': source, 'source_hash': source_hash, 'status': 'usable'}


def parse_tests(response: str, entry_point: str, batch_id: str) -> tuple[list, list]:
    """Accept JSON strings containing one literal equality assertion each.

    Validation only checks visible syntax and interface, never expected-answer
    validity. In particular wrong assertions remain part of the evidence pool.
    """
    fenced = re.search(r'```(?:json)?[ \t]*\r?\n(.*?)```', response, re.S)
    raw = fenced.group(1) if fenced else response
    try:
        assertions = json.loads(raw)
        if not isinstance(assertions, list):
            raise ValueError('expected a JSON array')
    except (ValueError, TypeError) as exc:
        return [], [{'generation_batch_id': batch_id, 'reason': 'json_format', 'detail': str(exc)}]
    tests, failures = [], []
    for index, assertion in enumerate(assertions):
        try:
            if not isinstance(assertion, str):
                raise ValueError('assertion must be a string')
            body = ast.parse(assertion).body
            if len(body) != 1 or not isinstance(body[0], ast.Assert) or body[0].msg is not None:
                raise ValueError('expected one assertion without message')
            expr = body[0].test
            if not (isinstance(expr, ast.Compare) and len(expr.ops) == 1
                    and isinstance(expr.ops[0], ast.Eq) and isinstance(expr.left, ast.Call)
                    and isinstance(expr.left.func, ast.Name) and expr.left.func.id == entry_point):
                raise ValueError('expected entry_point(literals) == literal')
            call = expr.left
            for arg in call.args:
                ast.literal_eval(arg)
            for kw in call.keywords:
                if kw.arg is None:
                    raise ValueError('keyword unpacking is unsupported')
                ast.literal_eval(kw.value)
            ast.literal_eval(expr.comparators[0])
            tests.append({'test_id': f'{batch_id}/t{index:02d}', 'generation_batch_id': batch_id,
                          'assertion': assertion})
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            failures.append({'generation_batch_id': batch_id, 'index': index,
                             'assertion': assertion, 'reason': 'interface_format', 'detail': str(exc)})
    return tests, failures


def sandbox_command(*, readonly: dict[str, str], writable: dict[str, str] | None = None,
                    binary: str = 'bwrap', python: str = '/usr/bin/python3',
                    arguments: list[str], extra_env: dict | None = None) -> list[str]:
    cmd = [binary, '--die-with-parent', '--unshare-all', '--new-session']
    for host in ['/usr', '/lib', '/lib64', '/bin']:
        if Path(host).exists():
            cmd += ['--ro-bind', host, host]
    cmd += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/work']
    for host, target in readonly.items():
        cmd += ['--ro-bind', str(host), target]
    for host, target in (writable or {}).items():
        cmd += ['--bind', str(host), target]
    cmd += ['--chdir', '/work', '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
            '--setenv', 'PYTHONHASHSEED', '0', '--setenv', 'OMP_NUM_THREADS', '1',
            '--setenv', 'OPENBLAS_NUM_THREADS', '1', '--setenv', 'HOME', '/tmp']
    for key, value in (extra_env or {}).items():
        cmd += ['--setenv', key, str(value)]
    return cmd + ['--', python, *arguments]


def _limits(seconds: float):
    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(seconds), math.ceil(seconds) + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    return apply


def execute_candidate(source: str, tests: list[dict], *, per_test_timeout: float = 0.5,
                      sandbox_binary: str = 'bwrap') -> dict:
    """Fresh process/state per test, isolated network/mount/pid namespaces.

    An abnormal sandbox supervisor exit is an infrastructure failure and raises;
    it must never silently become a row of program failures.
    """
    if per_test_timeout <= 0:
        raise ValueError('per_test_timeout must be positive')
    worker = Path(__file__).with_name('_sandbox_worker.py').resolve()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='tgselect-') as temp:
        payload = Path(temp) / 'payload.json'
        base_prefix = str(Path(sys.base_prefix).resolve())
        runtime = str(Path(base_prefix) / 'bin' / 'python')
        cmd = sandbox_command(readonly={str(payload): '/work/payload.json',
                                        str(worker): '/work/worker.py', base_prefix: base_prefix},
                              binary=sandbox_binary, python=runtime,
                              arguments=['-I', '-S', '/work/worker.py'])
        wall_limit = 15 + per_test_timeout
        result = {'results': []}
        # Each test gets a new mount and PID namespace. Exiting its namespace
        # removes descendants and writable state, including /tmp and /dev/shm.
        # Probe the sandbox even for a zero-column matrix.
        for test in tests or [None]:
            test_list = [] if test is None else [test]
            payload.write_text(json.dumps({'source': source, 'tests': test_list,
                                           'per_test_timeout': per_test_timeout}))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=wall_limit,
                                      preexec_fn=_limits(wall_limit))
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f'sandbox infrastructure failure: {exc}') from exc
            if proc.returncode:
                raise RuntimeError(f'sandbox exited {proc.returncode}: {proc.stderr[-2000:]}')
            try:
                one = json.loads(proc.stdout)
                if len(one['results']) != len(test_list):
                    raise ValueError('incomplete result count')
                result['results'].extend(one['results'])
                result['python_version'] = one['python_version']
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f'sandbox malformed output: {proc.stdout[-1000:]}') from exc
    result['wall_seconds'] = time.monotonic() - started
    result['execution_count'] = len(tests)
    return result
