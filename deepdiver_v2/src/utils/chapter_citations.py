"""Bounded citation-only repair; never rewrite chapter body or guess source IDs."""
import logging
import re
from .report_quality import (protect_document_numbers, restore_document_numbers,
                             protect_numeric_math, restore_numeric_math,
                             normalize_and_validate_report)


def _protected(text):
    text, docs = protect_document_numbers(text)
    text, math = protect_numeric_math(text)
    return text, docs, math


def citation_ids(text):
    return set(map(int, re.findall(r'\[(\d+)\]', _protected(text)[0])))


def without_citations(text, remove=None):
    protected, docs, math = _protected(text)
    protected = re.sub(r'\[(\d+)\]',
                       lambda m: '' if remove is None or int(m[1]) in remove else m[0], protected)
    return restore_document_numbers(restore_numeric_math(protected, math), docs)


def _project_citations(original, candidate):
    """Transfer citation positions only; never return the model's rewritten body.

    Match every non-whitespace character in order, allowing only typographic
    quote variants. Preserve ASCII word boundaries to reject whitespace edits
    such as 'not able' -> 'notable'. No fuzzy matching or substring anchors.
    """
    base = without_citations(original)
    protected, docs, math = _protected(candidate)
    insertions = []
    fragments = []
    end = 0
    length = 0
    for match in re.finditer(r'\[(\d+)\]', protected):
        fragment = restore_document_numbers(restore_numeric_math(protected[end:match.start()], math), docs)
        fragments.append(fragment)
        length += len(fragment)
        insertions.append((length, match.group(0)))
        end = match.end()
    fragments.append(restore_document_numbers(restore_numeric_math(protected[end:], math), docs))
    body = ''.join(fragments)
    fold = str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"})

    def signature(text):
        positions = [i for i, char in enumerate(text) if not char.isspace()]
        chars = ''.join(text[i] for i in positions).translate(fold)
        boundaries = {j for j in range(1, len(positions))
                      if positions[j] > positions[j-1]+1
                      and text[positions[j-1]].isascii() and text[positions[j]].isascii()
                      and (text[positions[j-1]].isalnum() or text[positions[j-1]] == '_')
                      and (text[positions[j]].isalnum() or text[positions[j]] == '_')}
        return chars, boundaries, positions

    left, left_boundaries, left_pos = signature(base)
    right, right_boundaries, right_pos = signature(body)
    if not left or left != right or left_boundaries != right_boundaries:
        return None
    from bisect import bisect_left
    patches = {}
    for offset, marker in insertions:
        count = bisect_left(right_pos, offset)
        target = left_pos[count-1]+1 if count else 0
        # Match the original side of an existing whitespace run when possible.
        if offset and body[offset-1].isspace():
            target = left_pos[count] if count < len(left_pos) else len(base)
        patches.setdefault(target, []).append(marker)
    result = base
    for offset in sorted(patches, reverse=True):
        result = result[:offset] + ''.join(patches[offset]) + result[offset:]
    return result if without_citations(result) == base else None


def review_chapter_citations(content, sources, cross_sources, repair, chinese=True):
    """sources: {global ID: actual evidence}; cross_sources require original text.

    A single semantic repair is attempted for empty coverage or out-of-context IDs.
    Cross-chapter evidence is offered only on that repair, not silently accepted.
    Accepted output retains the original body byte-for-byte after citation removal.
    """
    ids = citation_ids(content)
    invalid = ids - set(sources)
    missing = bool(sources) and not ids
    status = {'missing': missing, 'invalid_ids': sorted(invalid), 'repair_attempted': False,
              'repaired': False, 'degraded': False, 'no_original_evidence': not bool(sources)}
    if not sources and not ids:
        status['degraded'] = True
        return content, status
    if not missing and not invalid:
        return content, status
    evidence = dict(sources)
    evidence.update({i: text for i, text in cross_sources.items() if i in invalid and text})
    status['repair_attempted'] = True
    try:
        candidate = repair(content, evidence, status)
        if '<chapter_content>' in candidate:
            candidate = candidate.split('<chapter_content>', 1)[1].split('</chapter_content>', 1)[0].strip()
        candidate, markers = normalize_and_validate_report(candidate, source_citations=True)
        candidate_ids = citation_ids(candidate)
        if not markers and candidate_ids and candidate_ids <= set(evidence):
            projected = _project_citations(content, candidate)
            if projected is not None and citation_ids(projected) == candidate_ids:
                status['repaired'] = True
                return projected, status
    except Exception as exc:
        logging.getLogger(__name__).warning('[ChapterCitations] repair unavailable: %s', exc)
    # Do not retain unverifiable IDs just because they happen to exist globally.
    cleaned = without_citations(content, invalid)
    status['degraded'] = True
    return cleaned, status
