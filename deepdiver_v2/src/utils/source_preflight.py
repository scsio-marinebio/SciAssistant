"""Bounded, once-per-workspace source registration before Writer starts.

No network access here. The caller supplies the existing document analyzer in
non-persisting mode; this module preserves every existing source slot.
"""
import json
import os
from pathlib import Path
import threading
from weakref import WeakValueDictionary

from .source_provenance import generated_paths
from .writer_sources import _terms

LIMIT = 12
ROOTS = {'url_crawler_save_files', 'pubmed', 'arxiv', 'rag_downloads',
         'downloads', 'user_uploads', 'library_refs'}
_LOCKS = WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()


def path_key(value):
    value = str(value or '').replace('\\', '/')
    while value.startswith('./'):
        value = value[2:]
    for suffix in ('.pdf.txt', '.docx.txt', '.doc.txt'):
        if value.lower().endswith(suffix):
            return value[:-4]
    return value


def _rows(raw):
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except (ValueError, TypeError):
            continue
    return rows


def _usable(row):
    return (bool(row.get('file_path')) and row.get('doc_time') != 'Processing failed'
            and not any(term in str(row.get('information_richness', '')).lower()
                        for term in ('considered scarce', 'indicating scarcity', 'is scarce',
                                     'lacks substantive content', 'no substantive content',
                                     'very limited information', 'does not provide any substantive')))


def _local_file(workspace, value):
    relative = str(value or '').replace('\\', '/').removeprefix('./')
    if (not relative or relative.startswith('/') or ':' in relative
            or '..' in relative.split('/') or relative.split('/')[0] not in ROOTS):
        return None
    candidate = workspace / relative
    if candidate.suffix.lower() not in {'.txt', '.md', '.pdf', '.doc', '.docx'}:
        return None
    try:
        candidate.resolve().relative_to(workspace.resolve())
        if candidate.is_file() and candidate.stat().st_size:
            return relative
    except (OSError, ValueError):
        pass
    return None


def _downloaded(workspace):
    """Only successful tool provenance, never a blind directory scan."""
    found = []
    for log in sorted((workspace/'tool_call_logs').glob('*.jsonl')):
        with log.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                try:
                    call = json.loads(line)
                    if not call.get('success') or call.get('tool_name') not in {
                        'url_crawler', 'get_pubmed_article', 'arxiv_read_paper',
                        'download_files', 'scihub_get_paper', 'rag_document_saver'}:
                        continue
                    data = (call.get('output_result') or {}).get('data')
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if not isinstance(item, dict) or item.get('success') is False:
                            continue
                        path = _local_file(workspace, item.get('file_path'))
                        if path:
                            found.append(path)
                except (ValueError, TypeError, AttributeError):
                    continue
    return list(dict.fromkeys(found))


def plan_sources(workspace, rows, key_files, query, limit=LIMIT):
    workspace = Path(workspace)
    generated = {path_key(p) for p in generated_paths(workspace)}
    indexed = {}
    for row in rows:
        key = path_key(row.get('file_path'))
        if key not in indexed or _usable(row):
            indexed[key] = row
    requested = [f.get('file_path', '') for f in key_files]
    originals = [p for p in requested if path_key(p) not in generated
                 and _usable(indexed.get(path_key(p), {}))]
    candidates = []
    seen = set()
    for value in requested:
        path = _local_file(workspace, value)
        key = path_key(path)
        if path and key not in generated and key not in seen:
            seen.add(key)
            row = indexed.get(key)
            # Successful-but-scarce rows are deliberately not re-analyzed.
            if row is None or row.get('doc_time') == 'Processing failed':
                candidates.append(path)
    fallback = []
    # Supplement only when the handoff has no indexed original evidence.
    if not originals:
        topics = [_terms(query)]
        topics += [_terms(indexed[path_key(p)].get('core_content', ''))
                   for p in requested if path_key(p) in generated and path_key(p) in indexed]
        ranked = []
        for order, value in enumerate(_downloaded(workspace)):
            key = path_key(value)
            if key in generated or key in seen:
                continue
            row = indexed.get(key)
            if row is not None and not _usable(row) and row.get('doc_time') != 'Processing failed':
                continue
            try:
                with (workspace/value).open(encoding='utf-8', errors='ignore') as source:
                    head = source.read(10000)
            except OSError:
                continue
            terms = _terms(value + ' ' + head)
            score = max((len(terms & topic) for topic in topics), default=0)
            if score >= 2:
                ranked.append((-score, order, value))
        for _, _, value in sorted(ranked)[:limit]:
            fallback.append(value)
            if not _usable(indexed.get(path_key(value), {})):
                candidates.append(value)
    return {'selected': candidates[:limit], 'fallback': fallback,
            'pending_count': len(candidates), 'deferred_count': max(0, len(candidates)-limit)}


def _persist_slots(index, before, new_rows):
    """Update failed slots in place, append new ones; retain raw old records."""
    current = index.read_bytes() if index.exists() else b''
    if current != before:
        raise RuntimeError('analysis_index_changed_during_preflight')
    lines = before.decode('utf-8').splitlines(keepends=True)
    positions = {}
    for position, line in enumerate(lines):
        try:
            row = json.loads(line)
            if isinstance(row, dict) and row.get('file_path'):
                positions.setdefault(path_key(row['file_path']), []).append((position, row))
        except ValueError:
            continue
    for row in new_rows:
        key = path_key(row.get('file_path'))
        if not key:
            continue
        slots = positions.get(key, [])
        if any(_usable(old) for _, old in slots):
            continue
        serialized = json.dumps(row, ensure_ascii=False) + '\n'
        if slots:
            pos, old = slots[0]
            if old.get('doc_time') == 'Processing failed':
                # Keep the original spelling too: existing handoffs refer to it.
                row = dict(row, file_path=old['file_path'])
                lines[pos] = json.dumps(row, ensure_ascii=False) + '\n'
        else:
            if lines and not lines[-1].endswith(('\n', '\r')):
                lines[-1] += '\n'
            positions[key] = [(len(lines), row)]
            lines.append(serialized)
    index.parent.mkdir(parents=True, exist_ok=True)
    temp = index.with_name(index.name + '.preflight.tmp')
    temp.write_bytes(''.join(lines).encode('utf-8'))
    os.replace(temp, index)


def prepare_sources(workspace, key_files, query, analyze):
    # A concurrent retry must not start writing while the first call appends IDs.
    key = str(Path(workspace).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
    with lock:
        return _prepare_sources(workspace, key_files, query, analyze)


def _prepare_sources(workspace, key_files, query, analyze):
    workspace = Path(workspace)
    index = workspace/'doc_analysis/file_analysis.jsonl'
    state_path = workspace/'.writer_source_preflight.json'
    # An exclusive marker bounds retries across Writer restarts and HITL phase 2.
    try:
        with state_path.open('x', encoding='utf-8') as stream:
            json.dump({'status': 'started', 'fallback': []}, stream)
    except FileExistsError:
        try:
            state = json.loads(state_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            state = {'status': 'previous_attempt_unavailable', 'fallback': []}
        return _result(index, key_files, state, workspace, cached=True)
    state = {'status': 'checked', 'selected': [], 'fallback': [], 'deferred_count': 0}
    try:
        # Never introduce or renumber sources after any chapter has been written.
        if any((workspace/'report').glob('part_*.md')):
            state['status'] = 'skipped_existing_chapters'
        else:
            before = index.read_bytes() if index.exists() else b''
            state.update(plan_sources(workspace, _rows(before.decode('utf-8')), key_files, query))
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            if state['selected']:
                state['status'] = 'attempted'
                result = analyze([{'file_path': p, 'task': query} for p in state['selected']])
                if not result.success or not isinstance(result.data, list):
                    raise RuntimeError(str(result.error or 'source_analysis_failed'))
                allowed = {path_key(p) for p in state['selected']}
                records = [r for r in result.data if isinstance(r, dict)
                           and path_key(r.get('file_path')) in allowed]
                _persist_slots(index, before, records)
                state['analyzed_count'] = sum(_usable(r) for r in records)
                state['status'] = 'completed'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)[:300]
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    return _result(index, key_files, state, workspace)


def _result(index, key_files, state, workspace, cached=False):
    rows = _rows(index.read_text(encoding='utf-8')) if index.exists() else []
    generated = {path_key(p) for p in generated_paths(workspace)}
    usable = {path_key(r.get('file_path')): r['file_path'] for r in rows
              if _usable(r) and path_key(r.get('file_path')) not in generated}
    files = [dict(f, file_path=usable.get(path_key(f.get('file_path')), f.get('file_path')))
             for f in key_files]
    seen = {path_key(f.get('file_path')) for f in files}
    for path in state.get('fallback', []):
        key = path_key(path)
        if key in usable and key not in seen:
            files.append({'file_path': usable[key]})
            seen.add(key)
    return {'key_files': files, 'metadata': dict(state, cached=cached,
            usable_handoff_sources=len(seen & set(usable)),
            no_original_evidence=not bool(seen & set(usable)))}
