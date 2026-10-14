import time
import logging
import json
import re
import hashlib
import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from config.thinking import build_thinking_fields, validate_thinking_payload

from config.config import (
    get_config,
    get_model_provider,
    build_llm_request_body,
    build_model_request_headers,
    normalize_tool_call_mode,
)

logger = logging.getLogger(__name__)


class LLMOutputTruncatedError(RuntimeError):
    """Raised when a tool-producing response stops at the token limit."""


class LLMEmptyContentError(RuntimeError):
    """Raised when an upstream success response contains no usable model output."""


@dataclass
class LLMToolTurn:
    """One assistant turn with transport-native tool-call metadata."""

    content: str = ""
    reasoning_content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    assistant_message: Dict[str, Any] = field(default_factory=dict)
    tool_call_mode: str = "text"


def prepare_openai_tools(tool_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a validated deep copy of the active Agent's function schemas."""
    if not isinstance(tool_schemas, list) or not tool_schemas:
        raise ValueError("Native tool-call mode requires a non-empty tool schema list")

    prepared = copy.deepcopy(tool_schemas)
    seen_names = set()
    for index, schema in enumerate(prepared):
        if not isinstance(schema, dict) or schema.get("type") != "function":
            raise ValueError(f"Tool schema at index {index} must have type='function'")
        function = schema.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Tool schema at index {index} is missing function metadata")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool schema at index {index} has an invalid function name")
        if name in seen_names:
            raise ValueError(f"Duplicate native tool schema name: {name}")
        seen_names.add(name)
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(
                f"Native tool schema '{name}' parameters must be a JSON object schema"
            )
    # Fail before the HTTP request if an MCP adapter exposed non-JSON data.
    json.dumps(prepared, ensure_ascii=False)
    return prepared


def _normalize_openai_tool_calls(
    raw_calls: Any,
    *,
    require_ids: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize native calls without discarding ids required by role=tool."""
    if not raw_calls:
        return []
    calls = raw_calls if isinstance(raw_calls, list) else [raw_calls]
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"Native tool call at index {index} is not an object")
        function_block = call.get("function") if isinstance(call.get("function"), dict) else None
        if function_block is None and "name" in call:
            function_block = call
        if not isinstance(function_block, dict):
            raise ValueError(f"Native tool call at index {index} is missing function metadata")
        name = function_block.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Native tool call at index {index} has no function name")
        call_type = call.get("type", "function")
        if require_ids and call_type != "function":
            raise ValueError(
                f"Native tool call '{name}' has unsupported type: {call_type!r}"
            )
        call_id = call.get("id")
        if require_ids and (not isinstance(call_id, str) or not call_id):
            raise ValueError(f"Native tool call '{name}' has no id")
        if require_ids and call_id in seen_ids:
            raise ValueError(f"Duplicate native tool_call id: {call_id}")
        if isinstance(call_id, str) and call_id:
            seen_ids.add(call_id)
        raw_arguments = function_block.get("arguments")
        arguments: Any = raw_arguments
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                if require_ids:
                    raise ValueError(
                        f"Native tool call '{name}' arguments are not valid JSON"
                    )
                # Text compatibility mode may still consume provider-returned
                # tool_calls through the legacy parser path.
                arguments = raw_arguments
        if require_ids and not isinstance(arguments, dict):
            raise ValueError(
                f"Native tool call '{name}' arguments must decode to a JSON object"
            )
        normalized.append({
            "id": call_id,
            "type": call_type,
            "name": name,
            "arguments": arguments,
            "raw_arguments": raw_arguments,
        })
    return normalized


def _write_debug_llm_response(
    *,
    model_name: Optional[str],
    url: Optional[str],
    status_code: Optional[int],
    payload: Optional[Any] = None,
    text: Optional[str] = None,
) -> None:
    cfg = get_config()
    if not getattr(cfg, "debug_mode", False):
        return
    try:
        base_dir = Path(__file__).resolve().parents[3]
        log_path = base_dir / "logs" / "llm_raw.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": model_name,
            "url": url,
            "status_code": status_code,
        }
        if payload is not None:
            record["response_json"] = payload
        if text is not None:
            record["response_text"] = text
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("Failed to write raw LLM response: %s", exc)


def llm_chat(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    max_retries: Optional[int] = None,
    retry_sleep: float = 5.0,
    reject_truncated: bool = False,
    retry_truncated_same_request: bool = True,
    preserve_reasoning_on_argumentless_tool: bool = False,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    tool_call_mode: Optional[str] = None,
    return_tool_turn: bool = False,
) -> Union[str, LLMToolTurn]:
    """Unified chat-completions call.

    Returns assistant text content with automatic fallback to reasoning_content.
    Raises on final failure after retries.
    """
    cfg = get_config()
    model_cfg = cfg.get_custom_llm_config()

    url = model_cfg.get("url")
    token = model_cfg.get("token")
    model_name = (model or model_cfg.get("model"))
    req_timeout = int(timeout or model_cfg.get("timeout", 180))
    retries = int(max_retries if max_retries is not None else getattr(cfg, "max_retries", 3))

    provider = (model_cfg.get("provider") or "").strip().lower()
    if not provider or provider == "auto":
        provider = get_model_provider(model_name)
    # Native calling applies only to calls that explicitly supply an Agent tool
    # schema.  The same client is also used for ordinary internal summarization
    # and repair completions; those must remain tool-free even when the active
    # deployment enables native tools for the four Agent loops.
    resolved_tool_call_mode = normalize_tool_call_mode(
        (tool_call_mode or model_cfg.get("tool_call_mode"))
        if tool_schemas is not None
        else "text"
    )

    headers = build_model_request_headers(token)

    body = build_llm_request_body(
        model_name=model_name,
        messages=messages,
        temperature=temperature if temperature is not None else cfg.model_temperature,
        max_tokens=max_tokens if max_tokens is not None else cfg.model_max_tokens,
    )
    if provider == "pangu":
        chat_template = model_cfg.get("chat_template")
        if chat_template:
            # Only send chat_template and spaces_between_special_tokens when
            # we are providing our own template. Managed inference services
            # (e.g. intranet pangu_ultra_moe) apply their own template and
            # do not expect these parameters.
            body.setdefault("chat_template", chat_template)
            spaces_between_special_tokens = model_cfg.get("spaces_between_special_tokens")
            if spaces_between_special_tokens is None:
                spaces_between_special_tokens = False
            body.setdefault("spaces_between_special_tokens", spaces_between_special_tokens)

    # Explicit transport selection; only Pangu retains its legacy default.
    body.update(build_thinking_fields(model_cfg, provider))

    if extra_body:
        for k, v in extra_body.items():
            if v is not None:
                body[k] = v
    validate_thinking_payload(body)

    if resolved_tool_call_mode == "native":
        body["tools"] = prepare_openai_tools(tool_schemas or [])
        body.setdefault("tool_choice", "auto")

    request_messages = body.get("messages") if isinstance(body, dict) else None
    request_message_count = len(request_messages) if isinstance(request_messages, list) else 0
    request_chars = 0
    if isinstance(request_messages, list):
        for request_message in request_messages:
            if isinstance(request_message, dict):
                request_content = request_message.get("content")
                if isinstance(request_content, str):
                    request_chars += len(request_content)
    request_fingerprint = hashlib.sha256(
        json.dumps(request_messages or [], ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        monotonic_clock = getattr(time, "monotonic", lambda: 0.0)
        attempt_started_at = monotonic_clock()
        logger.info(
            "[llm_chat] request start model=%s provider=%s url=%s "
            "attempt=%s/%s timeout=%ss messages=%s chars=%s max_tokens=%s "
            "enable_thinking=%s chat_template_think=%s thinking_type=%s "
            "chat_template_enable_thinking=%s reasoning_effort=%s tool_call_mode=%s tools=%s "
            "request_fingerprint=%s",
            model_name,
            provider,
            url,
            attempt + 1,
            retries,
            req_timeout,
            request_message_count,
            request_chars,
            body.get("max_tokens"),
            body.get("enable_thinking"),
            (body.get("chat_template_kwargs") or {}).get("think")
            if isinstance(body.get("chat_template_kwargs"), dict)
            else None,
            (body.get("thinking") or {}).get("type") if isinstance(body.get("thinking"), dict) else None,
            (body.get("chat_template_kwargs") or {}).get("enable_thinking")
            if isinstance(body.get("chat_template_kwargs"), dict) else None,
            body.get("reasoning_effort"),
            resolved_tool_call_mode,
            len(body.get("tools") or []),
            request_fingerprint,
        )
        try:
            resp = requests.post(url=url, headers=headers, json=body, timeout=req_timeout, verify=False)
            elapsed_seconds = monotonic_clock() - attempt_started_at
            logger.info(
                "[llm_chat] response received model=%s status_code=%s elapsed=%.2fs "
                "attempt=%s/%s request_fingerprint=%s",
                model_name,
                resp.status_code,
                elapsed_seconds,
                attempt + 1,
                retries,
                request_fingerprint,
            )
            if resp.status_code >= 400:
                messages = body.get("messages") if isinstance(body, dict) else None
                message_count = len(messages) if isinstance(messages, list) else 0
                message_chars = 0
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str):
                                message_chars += len(content)
                response_preview = resp.text or ""
                if len(response_preview) > 1200:
                    response_preview = f"{response_preview[:1200]}..."
                logger.error(
                    "LLM API HTTP %s for %s (model=%s, keys=%s, temp=%s, max_tokens=%s, messages=%s, chars=%s). Response: %s",
                    resp.status_code,
                    url,
                    model_name,
                    list(body.keys()),
                    body.get("temperature"),
                    body.get("max_tokens"),
                    message_count,
                    message_chars,
                    response_preview,
                )
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                _write_debug_llm_response(
                    model_name=model_name,
                    url=url,
                    status_code=resp.status_code,
                    text=resp.text,
                )
                raise
            _write_debug_llm_response(
                model_name=model_name,
                url=url,
                status_code=resp.status_code,
                payload=data,
            )
            if isinstance(data, dict) and data.get("error"):
                raise ValueError(f"LLM API error: {data['error']}")
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                raise ValueError(f"LLM API response missing 'choices': {data}")

            # 永远开启的紧凑输出诊断日志：记录 finish_reason 与 token 用量，
            # 便于快速定位输出被 max_tokens 截断（finish_reason=length）的情况。
            finish_reason = (choices[0] or {}).get("finish_reason")
            usage = data.get("usage") if isinstance(data, dict) else {}
            completion_tokens = (usage or {}).get("completion_tokens")
            prompt_tokens = (usage or {}).get("prompt_tokens")
            if finish_reason == "length":
                logger.warning(
                    "[llm_chat] finish_reason=length 输出触顶被截断 (model=%s, completion_tokens=%s, prompt_tokens=%s, max_tokens=%s)",
                    model_name, completion_tokens, prompt_tokens, body.get("max_tokens"),
                )
                if reject_truncated or resolved_tool_call_mode == "native":
                    raise LLMOutputTruncatedError(
                        "LLM output truncated: finish_reason=length "
                        f"(model={model_name}, completion_tokens={completion_tokens}, "
                        f"max_tokens={body.get('max_tokens')})"
                    )
            else:
                logger.info(
                    "[llm_chat] finish_reason=%s (model=%s, completion_tokens=%s, prompt_tokens=%s, max_tokens=%s)",
                    finish_reason, model_name, completion_tokens, prompt_tokens, body.get("max_tokens"),
                )

            message = (choices[0] or {}).get("message", {})

            content = message.get("content")
            # Different OpenAI-compatible gateways use either field name.
            # Keep reasoning separate from the business payload whenever the
            # upstream service provides it explicitly.
            reasoning_content = message.get("reasoning_content") or message.get("reasoning")

            # pangu_ultra_moe currently has a gateway/template compatibility
            # issue when chat_template_kwargs.think is boolean false: the
            # reasoning text is placed in `content`, followed by a stray
            # `</think>` and the actual answer.  Always prefer the suffix after
            # the final closing marker so downstream JSON/metadata/tool parsers
            # only see the business response.  Keep sending a real JSON boolean
            # in the request; the string "false" was verified to still enable
            # reasoning and therefore is not a genuine no-think mode.
            if isinstance(content, str) and "</think>" in content:
                leaked_reasoning, final_content = content.rsplit("</think>", 1)
                if final_content.strip():
                    if not reasoning_content and leaked_reasoning.strip():
                        reasoning_content = re.sub(
                            r"^\s*<think>\s*",
                            "",
                            leaked_reasoning,
                            count=1,
                            flags=re.IGNORECASE,
                        ).strip()
                    logger.warning(
                        "LLM reasoning leaked into message.content; using only content after final </think> "
                        "(model=%s, leaked_chars=%s, final_chars=%s)",
                        model_name,
                        len(leaked_reasoning),
                        len(final_content),
                    )
                    content = final_content.strip()
            tool_calls = message.get("tool_calls")
            if not tool_calls and message.get("function_call"):
                tool_calls = [message.get("function_call")]
            normalized_tool_calls = _normalize_openai_tool_calls(
                tool_calls,
                require_ids=resolved_tool_call_mode == "native",
            )

            if resolved_tool_call_mode == "native" and message.get("function_call") and not message.get("tool_calls"):
                raise ValueError(
                    "Native tool-call mode requires message.tool_calls with ids; "
                    "legacy function_call cannot form a role=tool round trip"
                )

            if resolved_tool_call_mode == "native":
                allowed_tool_names = {
                    item["function"]["name"] for item in body.get("tools", [])
                }
                unknown_tool_names = sorted({
                    call["name"] for call in normalized_tool_calls
                    if call["name"] not in allowed_tool_names
                })
                if unknown_tool_names:
                    raise ValueError(
                        "Native response requested tools outside the active Agent whitelist: "
                        f"{unknown_tool_names}"
                    )

            raw_content = content if isinstance(content, str) else ""
            raw_reasoning_content = (
                reasoning_content if isinstance(reasoning_content, str) else ""
            )
            assistant_history_message: Dict[str, Any] = {
                "role": message.get("role") or "assistant",
                "content": message.get("content"),
            }
            if message.get("tool_calls"):
                assistant_history_message["tool_calls"] = copy.deepcopy(message["tool_calls"])
            if message.get("reasoning_content") is not None:
                assistant_history_message["reasoning_content"] = message.get("reasoning_content")
            elif message.get("reasoning") is not None:
                assistant_history_message["reasoning"] = message.get("reasoning")

            if return_tool_turn and resolved_tool_call_mode == "native":
                if not raw_content.strip() and not raw_reasoning_content.strip() and not normalized_tool_calls:
                    raise LLMEmptyContentError(
                        "Upstream LLM returned neither content nor native tool_calls "
                        f"(model={model_name}, finish_reason={finish_reason})"
                    )
                logger.info(
                    "[llm_chat] native tool response model=%s tool_calls=%s ids_present=%s",
                    model_name,
                    len(normalized_tool_calls),
                    all(bool(call.get("id")) for call in normalized_tool_calls),
                )
                return LLMToolTurn(
                    content=raw_content.strip(),
                    reasoning_content=raw_reasoning_content.strip(),
                    finish_reason=finish_reason,
                    tool_calls=normalized_tool_calls,
                    assistant_message=assistant_history_message,
                    tool_call_mode=resolved_tool_call_mode,
                )

            # 空值 fallback（同时处理 None 和空字符串 ""）
            if not content:
                content = reasoning_content
            elif (
                preserve_reasoning_on_argumentless_tool
                and isinstance(reasoning_content, str)
                and reasoning_content.strip()
                and isinstance(content, str)
                and "<arg_key>" not in content
                and re.search(
                    r"<tool_call>\s*[A-Za-z0-9_\-]+\s*"
                    r"(?:</invoke>|</tool_call>|$)",
                    content,
                    re.IGNORECASE,
                )
            ):
                # GLM-5.2 may place the complete outline/argument planning in
                # reasoning_content while content contains only an empty tool
                # marker. Preserve it for the Writer corrective turn.
                content = f"<think>{reasoning_content.strip()}</think>\n{content}"
            content_is_empty = content is None or (
                isinstance(content, str) and not content.strip()
            )
            if content_is_empty and normalized_tool_calls:
                content = json.dumps(normalized_tool_calls, ensure_ascii=False)
            elif content is not None and normalized_tool_calls:
                if "[unused11]" not in content and "```json" not in content:
                    tool_payload = json.dumps(normalized_tool_calls, ensure_ascii=False)
                    content = f"{content}\n```json\n{tool_payload}\n```"
            if content is None or (
                isinstance(content, str) and not content.strip()
            ):
                logger.error(
                    "LLM returned empty content from a successful HTTP response "
                    "(model=%s, finish_reason=%s, prompt_tokens=%s, completion_tokens=%s, "
                    "messages=%s, chars=%s, request_fingerprint=%s)",
                    model_name,
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    request_message_count,
                    request_chars,
                    request_fingerprint,
                )
                raise LLMEmptyContentError(
                    "Upstream LLM returned empty content "
                    f"(model={model_name}, finish_reason={finish_reason}, "
                    f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                    f"messages={request_message_count}, chars={request_chars}, "
                    f"request_fingerprint={request_fingerprint})"
                )
            final_content = str(content).strip()
            if return_tool_turn:
                return LLMToolTurn(
                    content=final_content,
                    reasoning_content=raw_reasoning_content.strip(),
                    finish_reason=finish_reason,
                    tool_calls=[],
                    assistant_message={"role": "assistant", "content": final_content},
                    tool_call_mode=resolved_tool_call_mode,
                )
            return final_content
        except Exception as e:
            last_err = e
            elapsed_seconds = monotonic_clock() - attempt_started_at
            logger.warning(
                "[llm_chat] request failed model=%s elapsed=%.2fs attempt=%s/%s "
                "error_type=%s error=%s request_fingerprint=%s",
                model_name,
                elapsed_seconds,
                attempt + 1,
                retries,
                type(e).__name__,
                e,
                request_fingerprint,
            )
            if isinstance(e, LLMOutputTruncatedError) and not retry_truncated_same_request:
                raise
            if attempt == retries - 1:
                break
            time.sleep(retry_sleep)

    if isinstance(last_err, LLMEmptyContentError):
        raise LLMEmptyContentError(
            f"Upstream LLM returned empty content after {retries} retries: {last_err}"
        ) from last_err
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")
