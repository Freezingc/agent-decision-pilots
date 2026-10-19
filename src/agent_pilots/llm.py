"""Resumable local inference; prompts and receipts are hashed and logged."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

MODEL = '/mnt/drug_db/zhangyudong/cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class LocalRunner:
    def __init__(self, *, model_path=MODEL, temperature=0.7, batch_size=16, max_tokens=384):
        from .hf_backend import HFBackend
        self.backend = HFBackend(model_path, {'temperature': temperature, 'top_p': 0.95,
                                             'max_tokens': max_tokens, 'request_batch_size': batch_size,
                                             'max_model_len': 8192})
        self.batch_size = batch_size

    def run(self, requests, output, *, temperature=None):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if temperature is not None:
            self.backend.temperature = temperature
        existing = {row['id']: row for row in read_jsonl(output)}
        if len({row['id'] for row in requests}) != len(requests):
            raise ValueError('duplicate request id')
        todo = []
        for row in requests:
            request_hash = digest({**row, 'temperature': self.backend.temperature, 'top_p': self.backend.top_p})
            if row['id'] in existing:
                if existing[row['id']]['request_hash'] != request_hash:
                    raise ValueError('request/config changed during resume: ' + row['id'])
            else:
                todo.append((row, request_hash))
        with output.open('a') as stream:
            # Group max generation length so per-request limits remain exact.
            for limit in sorted({int(row.get('max_tokens', 384)) for row, _ in todo}):
                matching = [(row, rhash) for row, rhash in todo if int(row.get('max_tokens', 384)) == limit]
                self.backend.max_tokens = limit
                for start in range(0, len(matching), self.batch_size):
                    batch = matching[start:start + self.batch_size]
                    prompts = [self.backend.tokenizer.apply_chat_template(
                        [{'role': 'system', 'content': row['system']},
                         {'role': 'user', 'content': row['user']}],
                        tokenize=False, add_generation_prompt=True) for row, _ in batch]
                    seeds = [int(row.get('seed', int(digest(row['id'])[:15], 16))) for row, _ in batch]
                    began = time.monotonic()
                    results = self.backend.generate(prompts, seeds)
                    elapsed = time.monotonic() - began
                    for (row, request_hash), result, seed in zip(batch, results, seeds, strict=True):
                        receipt = {'id': row['id'], 'request_hash': request_hash, 'seed': seed,
                                   'temperature': self.backend.temperature,
                                   'max_tokens': limit, 'batch_wall_seconds': elapsed,
                                   'batch_size': len(batch), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                                   **result}
                        stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')
                        existing[row['id']] = receipt
                    stream.flush()
                    print(json.dumps({'event': 'batch', 'output': str(output), 'done': start + len(batch),
                                      'total': len(matching), 'max_tokens': limit,
                                      'output_tokens': sum(x['output_tokens'] for x in results),
                                      'seconds': round(elapsed, 2)}), flush=True)
        return {row['id']: existing[row['id']] for row in requests}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    runner = LocalRunner(temperature=args.temperature, batch_size=args.batch_size)
    runner.run(read_jsonl(args.requests), args.output)
    write_json(str(args.output) + '.metadata.json', runner.backend.metadata)


if __name__ == '__main__':
    main()
