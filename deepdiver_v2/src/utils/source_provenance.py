"""Recover confirmed agent-authored text identity from successful workspace tool calls.

No filename suffix guessing and no URL-based exclusion of user documents.
Read-only: old workspaces need no migration; damaged log lines are ignored.
"""
import json
import logging
from pathlib import Path


def generated_paths(workspace):
    """Recognize confirmed authored text; successful later downloads replace it.

    Uploaded/library originals and their conversion outputs retain protection.
    Conflicting records without comparable timestamps remain unclassified.
    """
    from datetime import datetime, timezone
    protected = {'user_uploads', 'library_refs', 'downloads',
                 'rag_downloads', 'arxiv', 'report'}
    downloads = {'url_crawler', 'get_pubmed_article', 'arxiv_read_paper',
                 'download_files', 'scihub_get_paper', 'rag_document_saver'}
    events = {}

    def record(value, kind, timestamp):
        path = str(value or '').replace('\\', '/')
        while path.startswith('./'):
            path = path[2:]
        parts = path.split('/')
        if (not path or path.startswith('/') or ':' in path or '..' in parts
                or any(part in protected for part in parts)
                or Path(path).suffix.lower() not in {'.md', '.json', '.txt'}):
            return
        try:
            time = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            # Mixed aware/naive timestamps cannot safely be compared.
            stamp = (time.tzinfo is not None, time.replace(tzinfo=timezone.utc).timestamp()
                     if time.tzinfo is None else time.timestamp())
        except (ValueError, TypeError, OverflowError):
            stamp = None
        events.setdefault(path, []).append((kind, stamp))

    try:
        for log in sorted((Path(workspace) / 'tool_call_logs').glob('*.jsonl')):
            with log.open(encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    try:
                        call = json.loads(line)
                        if not call.get('success'):
                            continue
                        args = call.get('input_args') or {}
                        tool = call.get('tool_name')
                        timestamp = call.get('timestamp')
                        if tool == 'file_write' or (
                            tool == 'str_replace_based_edit_tool' and args.get('action') == 'create'
                        ):
                            record(args.get('file_path'), 'generated', timestamp)
                        elif tool in downloads:
                            result = call.get('output_result') or {}
                            if result.get('success') is False:
                                continue
                            data = result.get('data')
                            for item in data if isinstance(data, list) else [data]:
                                if isinstance(item, dict) and item.get('success') is not False:
                                    record(item.get('file_path'), 'downloaded', timestamp)
                    except (ValueError, TypeError, AttributeError):
                        continue
    except OSError as exc:
        logging.getLogger(__name__).warning('[SourceProvenance] log read unavailable: %s', exc)
    generated = set()
    for path, history in events.items():
        kinds = {kind for kind, _ in history}
        if kinds == {'generated'}:
            generated.add(path)
        elif kinds == {'generated', 'downloaded'}:
            stamps = [stamp for _, stamp in history]
            if any(stamp is None for stamp in stamps) or len({stamp[0] for stamp in stamps}) != 1:
                continue
            latest = max(stamps)
            if {kind for kind, stamp in history if stamp == latest} == {'generated'}:
                generated.add(path)
    return generated


def annotate_sources(rows, workspace):
    generated = generated_paths(workspace)
    return [dict(row, source_role=(
        'generated_background' if str(row.get('file_path', '')).replace('\\', '/').removeprefix('./') in generated
        else 'source'
    )) if isinstance(row, dict) else row for row in rows]


def partition_files(files, rows):
    roles = {r.get('file_path'): r.get('source_role') for r in rows if isinstance(r, dict)}
    background = [f for f in files if roles.get(f.get('file_path')) == 'generated_background']
    sources = [f for f in files if roles.get(f.get('file_path')) != 'generated_background']
    return sources, background
