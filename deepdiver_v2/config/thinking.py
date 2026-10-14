"""Explicit thinking request formats; independent of endpoint domain names."""
from typing import Any, Dict, Optional

FORMATS = {'enable_thinking', 'thinking', 'chat_template_enable_thinking', 'chat_template_think'}
MODES = {'enabled', 'disabled', 'auto', 'default'}
EFFORTS = {'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}


def parse_legacy_think_mode(value: Optional[str]) -> Optional[bool]:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y'}:
        return True
    if normalized in {'0', 'false', 'no', 'n'}:
        return False
    raise ValueError('MODEL_THINK_MODE must be true/false or unset; use MODEL_THINKING_MODE for auto/default')


def normalize_thinking_settings(fmt=None, mode=None, effort=None, legacy=None):
    def option(value, choices, name):
        normalized = value.strip().lower() if isinstance(value, str) else value
        if normalized is None or normalized == '':
            return None
        if not isinstance(normalized, str) or normalized not in choices:
            raise ValueError(f'{name} must be one of {", ".join(sorted(choices))}, or unset')
        return normalized

    fmt = option(fmt, FORMATS, 'MODEL_THINKING_FORMAT')
    mode = option(mode, MODES, 'MODEL_THINKING_MODE')
    effort = option(effort, EFFORTS, 'MODEL_REASONING_EFFORT')
    if legacy is not None and not isinstance(legacy, bool):
        raise ValueError('MODEL_THINK_MODE must resolve to a boolean or None')
    if legacy is not None and mode is not None:
        if mode != ('enabled' if legacy else 'disabled'):
            raise ValueError('MODEL_THINK_MODE conflicts with MODEL_THINKING_MODE; remove the legacy setting')
    if fmt is None and mode not in (None, 'default'):
        raise ValueError('MODEL_THINKING_MODE requires an explicit MODEL_THINKING_FORMAT')
    if mode == 'auto' and fmt != 'thinking':
        raise ValueError('auto requires MODEL_THINKING_FORMAT=thinking; boolean formats cannot express auto')
    if effort and (mode == 'disabled' or (mode is None and legacy is False and fmt is not None)):
        raise ValueError('MODEL_REASONING_EFFORT cannot accompany disabled thinking')
    return fmt, mode, effort


def build_thinking_fields(settings: Dict[str, Any], provider: str) -> Dict[str, Any]:
    legacy = settings.get('think_mode')
    fmt, mode, effort = normalize_thinking_settings(
        settings.get('thinking_format'), settings.get('thinking_mode'),
        settings.get('reasoning_effort'), legacy,
    )
    if fmt is None and provider == 'pangu' and mode is None and legacy is not None:
        fmt = 'chat_template_think'
    if fmt is not None and mode is None and legacy is not None:
        mode = 'enabled' if legacy else 'disabled'
    if effort and mode == 'disabled':
        raise ValueError('MODEL_REASONING_EFFORT cannot accompany disabled thinking')
    fields = {}
    if fmt and mode not in (None, 'default'):
        if fmt == 'thinking':
            fields['thinking'] = {'type': mode}
        elif fmt == 'enable_thinking':
            fields['enable_thinking'] = mode == 'enabled'
        else:
            key = 'think' if fmt == 'chat_template_think' else 'enable_thinking'
            fields['chat_template_kwargs'] = {key: mode == 'enabled'}
    if effort:
        fields['reasoning_effort'] = effort
    return fields


def validate_thinking_payload(body: Dict[str, Any]) -> None:
    """Do not allow extra_body to mix multiple known thinking transports."""
    nested = body.get('chat_template_kwargs')
    switches = [body[key] for key in ('enable_thinking', 'thinking') if key in body]
    if isinstance(nested, dict):
        switches.extend(nested[key] for key in ('think', 'enable_thinking') if key in nested)
    if len(switches) > 1:
        raise ValueError('Conflicting thinking formats in request body; send only one thinking switch')
