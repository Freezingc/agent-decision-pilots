"""Standard-library-only supervisor; run exclusively inside bubblewrap."""

import contextlib
import io
import json
import multiprocessing as mp
import os
import resource
import signal
import sys
import time


def check(source, test, connection):
    started = time.monotonic()
    try:
        # Redirect file descriptors as well as Python streams: a generated print
        # must not corrupt the JSON supervisor channel.
        fd = os.open('/dev/null', os.O_RDWR)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if fd > 2:
            os.close(fd)
        scope = {'__name__': '__candidate__'}
        exec(compile(source, '<candidate>', 'exec'), scope)
        exec(compile(test['assertion'], '<generated-test>', 'exec'), scope)
        result = {'status': 'pass'}
    except AssertionError:
        result = {'status': 'assertion_fail'}
    except BaseException as exc:
        result = {'status': 'program_error', 'exception': type(exc).__name__, 'detail': str(exc)[:300]}
    result['seconds'] = time.monotonic() - started
    connection.send_bytes(json.dumps(result).encode())
    connection.close()


def main():
    # Apply after entering the new user namespace; before unshare this counts
    # host-namespace processes and can prevent bubblewrap itself from starting.
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    with open('/work/payload.json') as stream:
        payload = json.load(stream)
    if len(payload['tests']) > 1:
        raise ValueError('one writable/PID namespace must serve at most one test')
    context = mp.get_context('fork')
    results = []
    for test in payload['tests']:
        recv, send = context.Pipe(duplex=False)
        proc = context.Process(target=check, args=(payload['source'], test, send))
        proc.start()
        send.close()
        proc.join(payload['per_test_timeout'])
        if proc.is_alive():
            proc.kill()
            proc.join()
            result = {'status': 'timeout', 'seconds': payload['per_test_timeout']}
        elif recv.poll():
            try:
                result = json.loads(recv.recv_bytes(4096))
            except (EOFError, OSError, ValueError):
                result = {'status': 'program_error', 'exception': 'ChildProcessExit'}
        else:
            result = {'status': 'program_error', 'exception': 'ChildProcessExit', 'exit_code': proc.exitcode}
        recv.close()
        proc.close()
        results.append({'test_id': test['test_id'], **result})
    print(json.dumps({'results': results, 'python_version': sys.version.split()[0]}), flush=True)


if __name__ == '__main__':
    main()
