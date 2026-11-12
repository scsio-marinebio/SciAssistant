"""Offline role separation and bounded retrieval for Writer handoffs.

This is candidate retrieval, not claim-level provenance verification. Only the
known fetcher directory is promoted; unknown paths keep the existing behavior.
"""
import re


_STOP_WORDS = set('the and for with from that this are was were have has into '
                 'report research study analysis source content information '
                 'high detailed relevant based using about which their'.split())


def _terms(text):
    text = str(text or '').lower()[:12000]
    words = set(re.findall(r'[a-z][a-z0-9-]{2,}', text)) - _STOP_WORDS
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return words


def _path(path):
    return str(path or '').replace('\\', '/').removeprefix('./')


def select_original_sources(valid_rows, requested_files, query, limit=12):
    """Return (external paths, background paths), or empty lists to keep legacy flow.

    Only server-annotated generated backgrounds are separated.
    Existing original sources (including user uploads) retain their order.
    Round-robin ranking keeps different summary topics represented.
    """
    requested = [item.get('file_path') for item in requested_files]
    matched = list(dict.fromkeys(p for p in requested if p in valid_rows))
    background = [p for p in matched if valid_rows[p].get('source_role') == 'generated_background']
    if not background:
        return [], []
    original = [p for p in matched if p not in background]
    if original:
        return original, background
    matched = background
    candidates = [p for p in valid_rows
                  if _path(p).startswith('url_crawler_save_files/')
                  and '..' not in _path(p).split('/')]
    query_terms = _terms(query)
    rankings = []
    for background in matched:
        topic = _terms(background + ' ' + str(valid_rows[background].get('core_content', '')))
        ranked = []
        for order, path in enumerate(candidates):
            row = valid_rows[path]
            terms = _terms(path + ' ' + str(row.get('core_content', '')) + ' ' +
                           str(row.get('task_relevance', '')))
            overlap = len(terms & topic)
            # At least two topic terms; generic query matches alone are insufficient.
            if overlap >= 2:
                ranked.append((overlap + len(terms & query_terms), order, path))
        rankings.append([p for _, _, p in sorted(ranked, key=lambda r: (-r[0], r[1]))])
    selected = []
    while len(selected) < limit and any(rankings):
        for ranking in rankings:
            while ranking and ranking[0] in selected:
                ranking.pop(0)
            if ranking and len(selected) < limit:
                selected.append(ranking.pop(0))
    # Weak evidence should not replace a usable handoff.
    if len(selected) < 3:
        return [], matched
    return selected, matched
