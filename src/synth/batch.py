import json
import time
from pathlib import Path
from . import state
TERMINAL_OK = {'completed'}
TERMINAL_FAIL = {'failed', 'expired', 'cancelled', 'cancelling'}

class Backend:

    def upload(self, path: Path) -> str:
        ...

    def create(self, file_id: str, endpoint: str, window: str, metadata: dict) -> str:
        ...

    def status(self, batch_id: str) -> dict:
        ...

    def download(self, file_id: str) -> bytes:
        ...

class OpenAIBatchBackend(Backend):

    def __init__(self, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client

    def _retry(self, fn, tries=5):
        delay = 2.0
        for attempt in range(tries):
            try:
                return fn()
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def upload(self, path: Path) -> str:

        def _do():
            with open(path, 'rb') as f:
                return self.client.files.create(file=f, purpose='batch')
        return self._retry(_do).id

    def create(self, file_id: str, endpoint: str, window: str, metadata: dict) -> str:
        return self._retry(lambda: self.client.batches.create(input_file_id=file_id, endpoint=endpoint, completion_window=window, metadata=metadata)).id

    def status(self, batch_id: str) -> dict:
        b = self._retry(lambda: self.client.batches.retrieve(batch_id))
        return {'status': b.status, 'output_file_id': getattr(b, 'output_file_id', None), 'error_file_id': getattr(b, 'error_file_id', None)}

    def cancel(self, batch_id: str) -> str:
        return self._retry(lambda: self.client.batches.cancel(batch_id)).status

    def download(self, file_id: str) -> bytes:
        return self._retry(lambda: self.client.files.content(file_id)).content

def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'shards': {}}

def save_state(path: Path, st: dict) -> None:
    state.atomic_write_json(path, st)

def submit(backend: Backend, request_files: dict[str, Path], state_path: Path, *, endpoint: str, window: str, prompt_version: str, retry_failed: bool=False, created_stamp: str='') -> dict:
    st = load_state(state_path)
    for pool, path in request_files.items():
        sha = state.sha256_file(path)
        rec = st['shards'].get(pool)
        if rec and rec['input_sha256'] == sha:
            failed = rec.get('status') in TERMINAL_FAIL
            if not failed or not retry_failed:
                continue
        file_id = backend.upload(path)
        batch_id = backend.create(file_id, endpoint, window, {'prompt_version': prompt_version, 'pool': pool})
        st['shards'][pool] = {'input_sha256': sha, 'file_id': file_id, 'batch_id': batch_id, 'status': 'submitted', 'output_file_id': None, 'error_file_id': None, 'created_at': created_stamp}
        save_state(state_path, st)
    return st

def poll(backend: Backend, state_path: Path) -> dict:
    st = load_state(state_path)
    for pool, rec in st['shards'].items():
        if rec['status'] in TERMINAL_OK or rec['status'] in TERMINAL_FAIL:
            continue
        info = backend.status(rec['batch_id'])
        rec.update(status=info['status'], output_file_id=info['output_file_id'], error_file_id=info['error_file_id'])
    save_state(state_path, st)
    return st

def cancel(backend: Backend, state_path: Path) -> dict:
    st = load_state(state_path)
    for pool, rec in st['shards'].items():
        if rec['status'] in TERMINAL_OK | TERMINAL_FAIL:
            continue
        rec['status'] = backend.cancel(rec['batch_id'])
    save_state(state_path, st)
    return st

def all_terminal(st: dict) -> bool:
    return bool(st['shards']) and all((r['status'] in TERMINAL_OK | TERMINAL_FAIL for r in st['shards'].values()))

def fetch(backend: Backend, state_path: Path, raw_dir: Path) -> list[str]:
    st = load_state(state_path)
    fetched = []
    for pool, rec in st['shards'].items():
        if rec['status'] not in TERMINAL_OK or not rec['output_file_id']:
            continue
        out = raw_dir / f'batch_output_pool{pool}.jsonl.gz'
        if out.exists() and rec.get('fetched_sha') == state.sha256_file(out):
            continue
        content = backend.download(rec['output_file_id'])
        lines = content.decode('utf-8').splitlines()
        state.atomic_write_jsonl_gz(out, lines)
        rec['fetched_sha'] = state.sha256_file(out)
        fetched.append(pool)
    save_state(state_path, st)
    return fetched
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}

def _sync_one(client, line: dict, max_retries: int) -> dict | None:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            if line.get('url') == '/v1/chat/completions':
                resp = client.chat.completions.create(**line['body'])
            else:
                resp = client.responses.create(**line['body'])
        except Exception as e:
            status = getattr(e, 'status_code', None)
            if isinstance(status, int) and status not in _TRANSIENT_STATUS and (status < 500):
                return {'id': f"sync-{line['custom_id']}", 'custom_id': line['custom_id'], 'response': {'status_code': status, 'request_id': '', 'body': {}}, 'error': f'{type(e).__name__}: {e}'}
            if attempt == max_retries - 1:
                return None
            time.sleep(delay)
            delay *= 2
            continue
        return {'id': f"sync-{line['custom_id']}", 'custom_id': line['custom_id'], 'response': {'status_code': 200, 'request_id': getattr(resp, 'id', ''), 'body': resp.model_dump(mode='json')}, 'error': None}
    return None

def collect_sync(client, request_files: dict[str, Path], state_path: Path, raw_dir: Path, *, workers: int=8, max_retries: int=5, created_stamp: str='') -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    st = load_state(state_path)
    summary = {}
    for pool, path in request_files.items():
        sha = state.sha256_file(path)
        rec = st['shards'].get(pool)
        out = raw_dir / f'batch_output_pool{pool}.jsonl.gz'
        if rec and rec['input_sha256'] == sha and (rec['status'] in TERMINAL_OK) and out.exists() and (rec.get('fetched_sha') == state.sha256_file(out)):
            summary[pool] = {'status': 'already_collected'}
            continue
        lines = state.read_jsonl(path)
        done: dict[str, dict] = {}
        partial = raw_dir / f'sync_partial_pool{pool}.jsonl'
        if partial.exists():
            for raw in partial.read_text(encoding='utf-8').splitlines():
                try:
                    d = json.loads(raw)
                    done[d['custom_id']] = d
                except ValueError:
                    continue
        pending = [l for l in lines if l['custom_id'] not in done]
        gave_up = 0
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(partial, 'a', encoding='utf-8') as pf:

            def record(res: dict) -> None:
                pf.write(json.dumps(res, ensure_ascii=False) + '\n')
                pf.flush()
                done[res['custom_id']] = res
            if pending:
                probe = _sync_one(client, pending[0], max_retries)
                if probe is None or probe.get('error'):
                    raise SystemExit(f"sync collection aborted (pool {pool}): first request failed {('after exhausting retries' if probe is None else 'permanently: ' + probe['error'])}. Nothing was spent beyond this probe.")
                record(probe)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_sync_one, client, l, max_retries) for l in pending[1:]]
                    for fut in as_completed(futs):
                        res = fut.result()
                        if res is None:
                            gave_up += 1
                        else:
                            record(res)
        if gave_up == 0 and len(done) == len(lines):
            state.atomic_write_jsonl_gz(out, [json.dumps(done[l['custom_id']], ensure_ascii=False) for l in lines])
            st['shards'][pool] = {'input_sha256': sha, 'file_id': None, 'batch_id': f'sync-{sha[:12]}', 'status': 'completed', 'output_file_id': None, 'error_file_id': None, 'created_at': created_stamp, 'backend': 'sync', 'fetched_sha': state.sha256_file(out)}
            save_state(state_path, st)
            partial.unlink()
            summary[pool] = {'requests': len(lines), 'status': 'completed', 'api_errors': sum((1 for d in done.values() if d.get('error')))}
        else:
            summary[pool] = {'requests': len(lines), 'collected': len(done), 'retry_pending': len(lines) - len(done), 'status': 'partial (rerun submit --backend sync to resume)'}
    return summary
