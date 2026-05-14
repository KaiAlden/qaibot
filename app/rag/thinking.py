from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedOutput:
    thinking: str
    answer: str


@dataclass(frozen=True)
class StreamPart:
    kind: str
    text: str


_ANSWER_MARKER_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\s*"
    r"(final\s+answer|final\s+output|正式回答|最终回答|回答)"
    r"\s*(?:\*\*)?\s*[:：]\s*"
)
_THINKING_MARKER_RE = re.compile(
    r"(?is)^\s*(?:here(?:'|’)?s\s+a\s+thinking\s+process|"
    r"thought\s+process|thinking\s+process|thinking|analysis|分析|思考过程|思考)"
    r"(?:\b|[:：])?"
)


def parse_model_output(
    text: str,
    start_tag: str = "<think>",
    end_tag: str = "</think>",
    answer_start_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
) -> ParsedOutput:
    text = text or ""
    if not text:
        return ParsedOutput("", "")

    lower_text = text.lower()
    start_lower = start_tag.lower()
    end_lower = end_tag.lower()
    answer_start_lower = answer_start_tag.lower()
    answer_end_lower = answer_end_tag.lower()

    start = lower_text.find(start_lower)
    end = lower_text.find(end_lower)
    answer_start = lower_text.find(answer_start_lower)
    if start >= 0 and end > start:
        thinking = text[start + len(start_tag) : end]
        answer_source = text[:start] + text[end + len(end_tag) :]
        answer = _extract_answer(answer_source, answer_start_tag, answer_end_tag)
        return ParsedOutput(_clean(thinking), _clean(answer))

    if start >= 0 and answer_start > start:
        thinking = text[start + len(start_tag) : answer_start]
        answer = _extract_answer(text[answer_start:], answer_start_tag, answer_end_tag)
        return ParsedOutput(_clean(thinking), _clean(answer))

    if start < 0 and end >= 0:
        thinking = text[:end]
        answer_source = text[end + len(end_tag) :]
        answer = _extract_answer(answer_source, answer_start_tag, answer_end_tag)
        return ParsedOutput(_clean(thinking), _clean(answer))

    if answer_start >= 0:
        prefix = text[:answer_start]
        answer = _extract_answer(text[answer_start:], answer_start_tag, answer_end_tag)
        thinking = prefix if _looks_like_thinking(prefix) else ""
        if not thinking and prefix.strip():
            answer = prefix + answer
        return ParsedOutput(_clean(thinking), _clean(answer))

    marker = _find_answer_marker(text)
    if marker and _looks_like_thinking(text[: marker.start()]):
        thinking = text[: marker.start()]
        answer = text[marker.end() :]
        return ParsedOutput(_clean(thinking), _clean(answer))

    if _looks_like_thinking(text):
        return ParsedOutput(_clean(text), "")

    return ParsedOutput("", _clean(text))


class ThinkingStreamParser:
    def __init__(
        self,
        start_tag: str = "<think>",
        end_tag: str = "</think>",
        answer_start_tag: str = "<answer>",
        answer_end_tag: str = "</answer>",
        buffer_chars: int = 1200,
    ):
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.answer_start_tag = answer_start_tag
        self.answer_end_tag = answer_end_tag
        self.buffer_chars = max(buffer_chars, 64)
        self._buffer = ""
        self._state = "unknown"

    def feed(self, text: str) -> list[StreamPart]:
        if not text:
            return []
        if self._state == "answer":
            answer = self._strip_answer_tags(text)
            return [StreamPart("answer", answer)] if answer else []

        self._buffer += text
        if self._state == "thinking":
            return self._drain_thinking()
        if self._state == "await_answer":
            return self._drain_await_answer()
        return self._drain_unknown()

    def finish(self) -> list[StreamPart]:
        if not self._buffer:
            return []
        buffer = self._buffer
        self._buffer = ""

        if self._state == "thinking":
            return [StreamPart("thinking", buffer), StreamPart("thinking_done", "")]
        if self._state == "await_answer":
            self._state = "answer"
            answer = self._strip_answer_tags(buffer)
            return [StreamPart("answer", answer)] if answer else []
        if self._state == "answer":
            return [StreamPart("answer", self._strip_answer_tags(buffer))]

        parsed = parse_model_output(
            buffer,
            self.start_tag,
            self.end_tag,
            self.answer_start_tag,
            self.answer_end_tag,
        )
        parts: list[StreamPart] = []
        if parsed.thinking:
            parts.append(StreamPart("thinking", parsed.thinking))
            parts.append(StreamPart("thinking_done", ""))
        if parsed.answer:
            parts.append(StreamPart("answer", parsed.answer))
        return parts

    def _drain_unknown(self) -> list[StreamPart]:
        lower_buffer = self._buffer.lower()
        start = lower_buffer.find(self.start_tag.lower())
        end = lower_buffer.find(self.end_tag.lower())
        answer_start = lower_buffer.find(self.answer_start_tag.lower())
        marker = _find_answer_marker(self._buffer)

        if start >= 0:
            before = self._buffer[:start]
            self._buffer = self._buffer[start + len(self.start_tag) :]
            self._state = "thinking"
            parts: list[StreamPart] = []
            if before.strip():
                parts.append(StreamPart("thinking" if _looks_like_thinking(before) else "answer", before))
            parts.extend(self._drain_thinking())
            return parts

        if end >= 0:
            thinking = self._buffer[:end]
            rest = self._buffer[end + len(self.end_tag) :]
            self._buffer = rest
            self._state = "await_answer"
            parts = [StreamPart("thinking", thinking), StreamPart("thinking_done", "")]
            parts.extend(self._drain_await_answer())
            return parts

        if answer_start >= 0:
            prefix = self._buffer[:answer_start]
            rest = self._buffer[answer_start + len(self.answer_start_tag) :]
            self._buffer = ""
            self._state = "answer"
            parts: list[StreamPart] = []
            if _looks_like_thinking(prefix):
                parts.extend([StreamPart("thinking", prefix), StreamPart("thinking_done", "")])
            elif prefix.strip():
                parts.append(StreamPart("answer", prefix))
            if rest:
                parts.append(StreamPart("answer", self._strip_answer_tags(rest)))
            return parts

        if marker and _looks_like_thinking(self._buffer[: marker.start()]):
            thinking = self._buffer[: marker.start()]
            answer = self._buffer[marker.end() :]
            self._buffer = ""
            self._state = "answer"
            parts = [StreamPart("thinking", thinking), StreamPart("thinking_done", "")]
            if answer:
                parts.append(StreamPart("answer", answer))
            return parts

        if _looks_like_thinking(self._buffer):
            self._state = "thinking"
            return self._drain_thinking()

        return []

    def _drain_thinking(self) -> list[StreamPart]:
        lower_buffer = self._buffer.lower()
        end = lower_buffer.find(self.end_tag.lower())

        split_at = None
        split_len = 0
        if end >= 0:
            split_at = end
            split_len = len(self.end_tag)

        if split_at is not None:
            thinking = self._buffer[:split_at]
            rest = self._buffer[split_at + split_len :]
            self._buffer = rest
            self._state = "await_answer"
            parts = [StreamPart("thinking", thinking), StreamPart("thinking_done", "")]
            parts.extend(self._drain_await_answer())
            return parts

        keep = max(len(self.end_tag), len(self.answer_start_tag), 32)
        if len(self._buffer) <= keep:
            return []
        emit = self._buffer[:-keep]
        self._buffer = self._buffer[-keep:]
        return [StreamPart("thinking", emit)]

    def _drain_await_answer(self) -> list[StreamPart]:
        lower_buffer = self._buffer.lower()
        answer_start = lower_buffer.find(self.answer_start_tag.lower())
        answer_end = lower_buffer.find(self.answer_end_tag.lower())

        if answer_start >= 0:
            answer = self._buffer[answer_start + len(self.answer_start_tag) :]
            self._buffer = ""
            self._state = "answer"
            answer = self._strip_answer_tags(answer)
            return [StreamPart("answer", answer)] if answer else []

        if answer_end >= 0:
            answer = self._strip_answer_tags(self._buffer)
            self._buffer = ""
            self._state = "answer"
            return [StreamPart("answer", answer)] if answer else []

        if self._is_partial_tag_prefix(self._buffer):
            return []

        stripped = self._buffer.lstrip()
        if not stripped:
            return []
        if stripped.startswith("<"):
            keep = max(len(self.answer_start_tag), len(self.answer_end_tag), 16)
            if len(stripped) < keep:
                return []

        self._buffer = ""
        self._state = "answer"
        return [StreamPart("answer", self._strip_answer_tags(stripped))]

    def _strip_answer_tags(self, text: str) -> str:
        lower_text = text.lower()
        start = lower_text.find(self.answer_start_tag.lower())
        if start >= 0:
            text = text[:start] + text[start + len(self.answer_start_tag) :]
            lower_text = text.lower()
        end = lower_text.find(self.answer_end_tag.lower())
        if end >= 0:
            text = text[:end] + text[end + len(self.answer_end_tag) :]
        return text

    def _is_partial_tag_prefix(self, text: str) -> bool:
        stripped = text.lstrip()
        if not stripped:
            return True
        candidates = [
            self.start_tag.lower(),
            self.end_tag.lower(),
            self.answer_start_tag.lower(),
            self.answer_end_tag.lower(),
        ]
        lower = stripped.lower()
        return any(candidate.startswith(lower) for candidate in candidates)


def summarize_thinking(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def _extract_answer(text: str, start_tag: str, end_tag: str) -> str:
    lower_text = text.lower()
    start = lower_text.find(start_tag.lower())
    if start >= 0:
        content_start = start + len(start_tag)
        end = lower_text.find(end_tag.lower(), content_start)
        if end >= 0:
            return text[content_start:end]
        return text[content_start:]

    end = lower_text.find(end_tag.lower())
    if end >= 0:
        return text[:end] + text[end + len(end_tag) :]
    return text


def _find_answer_marker(text: str) -> re.Match[str] | None:
    return _ANSWER_MARKER_RE.search(text)


def _looks_like_thinking(text: str) -> bool:
    sample = text.strip()
    if not sample:
        return False
    if sample.lower().startswith("<think>"):
        return True
    if _THINKING_MARKER_RE.search(sample[:400]):
        return True
    lower_sample = sample[:700].lower()
    thinking_prefixes = (
        "用户询问",
        "用户问题",
        "当前体质",
        "当前环境",
        "检索资料",
        "参考资料",
        "思考步骤",
        "分析步骤",
        "回答思路",
        "需要说明",
    )
    return (
        "analyze the request" in lower_sample
        or "retrieve knowledge" in lower_sample
        or "structure the response" in lower_sample
        or "self-correction" in lower_sample
        or any(prefix in sample[:700] for prefix in thinking_prefixes)
    )


def _clean(text: str) -> str:
    return text.strip()
