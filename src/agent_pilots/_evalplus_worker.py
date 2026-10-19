"""Official EvalPlus checking kernel, inside an ephemeral per-suite sandbox."""

import json
import pickle
import resource
import sys


def main():
    payload = pickle.load(sys.stdin.buffer)
    sys.stdin.close()
    # Limit after namespace creation, avoiding host UID process accounting.
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    from evalplus.eval import untrusted_check
    status, details = untrusted_check(
        'mbpp', payload['source'], payload['inputs'], payload['entry_point'],
        expected=payload['expected'], atol=payload['atol'], ref_time=payload['ref_time'],
        fast_check=True, min_time_limit=1.0, gt_time_limit_factor=4.0)
    print(json.dumps({'status': status, 'details': list(details)}), flush=True)


if __name__ == '__main__':
    main()
