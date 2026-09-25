"""Provider-agnostic LLM layer.

Everything above this file (proposer, search loop) talks to `LLMProvider.complete`.
Swapping Gemini for Claude, a vLLM/Ollama server, or a CHIA `LLMCallBase` backend
is a one-line config change:

    make_provider({"provider": "vertex", "model": "gemini-3.1-pro-preview", "location": "global"})
    make_provider({"provider": "gemini_api", "model": "gemini-3.1-pro-preview"})  # off-cloud, GEMINI_API_KEY
    make_provider({"provider": "openai_compat", "base_url": "http://localhost:8000/v1", "model": "..."})
    make_provider({"provider": "anthropic", "model": "claude-sonnet-5"})
    make_provider({"provider": "chia", "llm": <chia LLMCallBase instance>})
    make_provider({"provider": "fake", "replies": ["..."]})

Third-party SDKs are imported lazily so this module has no hard dependencies.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


@dataclass
class Msg:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str = ""
    model: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_s: float = 0.0


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: Sequence[Msg],
        *,
        json_mode: bool = False,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Raised when a provider fails after its own retries. Callers decide on fallback."""


def _retry(fn: Callable[[], Any], tries: int = 3, base: float = 2.0) -> Any:
    last: Optional[Exception] = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # provider SDKs raise many types; surface the last one
            last = e
            time.sleep(base * (2**i))
    raise LLMError(f"failed after {tries} tries: {last}") from last


class VertexGemini:
    """Gemini via google-genai, with two backends selected by `use_vertex`:
      * Vertex (default): `genai.Client(vertexai=True, project, location)` -- needs GCP + Application Default
        Credentials. This is what the cloud experiments used.
      * Developer API (`use_vertex=False`, registered as 'gemini_api'/'google_ai'): `genai.Client(api_key=...)`
        with GEMINI_API_KEY / GOOGLE_API_KEY. Works off-cloud."""

    def __init__(self, model: str = "gemini-3.1-pro-preview", project: Optional[str] = None,
                 location: str = "global", use_vertex: bool = True, api_key: Optional[str] = None,
                 api_key_env: str = "GEMINI_API_KEY", **_: Any):
        self.use_vertex = use_vertex
        self.name = "vertex" if use_vertex else "gemini_api"
        self.model = model
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    def _c(self):
        if self._client is None:
            from google import genai

            if self.use_vertex:
                self._client = genai.Client(vertexai=True, project=self.project, location=self.location)
            else:
                if not self.api_key:
                    raise LLMError("gemini_api provider needs an API key: set GEMINI_API_KEY (or GOOGLE_API_KEY), "
                                   "or pass api_key")
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete(self, system, messages, *, json_mode=False, max_tokens=4096, temperature=None):
        contents = [{"role": "model" if m.role == "assistant" else "user",
                     "parts": [{"text": m.content}]} for m in messages]
        # Thinking tokens count against the cap, so leave generous headroom.
        cfg: Dict[str, Any] = {"system_instruction": system, "max_output_tokens": max(max_tokens, 2048)}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if temperature is not None:
            cfg["temperature"] = temperature
        t0 = time.time()
        r = _retry(lambda: self._c().models.generate_content(model=self.model, contents=contents, config=cfg))
        u = getattr(r, "usage_metadata", None)
        return LLMResponse(r.text or "", self.name, self.model,
                           getattr(u, "prompt_token_count", None),
                           getattr(u, "candidates_token_count", None), time.time() - t0)


class OpenAICompat:
    """Any /v1/chat/completions server: OpenAI, vLLM, Ollama, OpenRouter, Together, ..."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY", **_: Any):
        self.name = "openai_compat"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")

    def complete(self, system, messages, *, json_mode=False, max_tokens=4096, temperature=None):
        import requests

        body: Dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.time()

        def call():
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=300)
            r.raise_for_status()
            return r.json()

        d = _retry(call)
        u = d.get("usage", {})
        return LLMResponse(d["choices"][0]["message"]["content"] or "", self.name, self.model,
                           u.get("prompt_tokens"), u.get("completion_tokens"), time.time() - t0)


class AnthropicClaude:
    """Claude via the anthropic SDK (API key) or AnthropicVertex (ADC, no key)."""

    def __init__(self, model: str = "claude-sonnet-5", vertex: bool = False, project: Optional[str] = None,
                 region: str = "global", **_: Any):
        self.name = "anthropic_vertex" if vertex else "anthropic"
        self.model = model
        self.vertex, self.project, self.region = vertex, project or os.environ.get("GOOGLE_CLOUD_PROJECT"), region
        self._client = None

    def _c(self):
        if self._client is None:
            import anthropic

            self._client = (anthropic.AnthropicVertex(project_id=self.project, region=self.region)
                            if self.vertex else anthropic.Anthropic())
        return self._client

    def complete(self, system, messages, *, json_mode=False, max_tokens=4096, temperature=None):
        kw: Dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "system": system,
                              "messages": [{"role": m.role, "content": m.content} for m in messages]}
        if temperature is not None:
            kw["temperature"] = temperature
        t0 = time.time()
        r = _retry(lambda: self._c().messages.create(**kw))
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, self.name, self.model, r.usage.input_tokens, r.usage.output_tokens,
                           time.time() - t0)


class ChiaLLM:
    """Adapter over a CHIA `LLMCallBase` (claude, vertex, opencode, antigravity, ...).

    CHIA backends take one user string, so history is flattened. Use this to run
    the same proposer inside CHIA's runtime (e.g. under the Evolver node).
    """

    def __init__(self, llm: Any, call: Optional[Callable[[str], Any]] = None, **_: Any):
        self.name = f"chia:{type(llm).__name__}"
        self.llm, self._call = llm, call

    def complete(self, system, messages, *, json_mode=False, max_tokens=4096, temperature=None):
        prompt = "\n\n".join(f"[{m.role}]\n{m.content}" for m in messages)
        if json_mode:
            prompt += "\n\nRespond with a single JSON object and nothing else."
        t0 = time.time()
        res = self._call(prompt) if self._call else self.llm.prompt(prompt)
        if not getattr(res, "success", True):
            raise LLMError("CHIA backend reported failure")
        return LLMResponse(res.result, self.name, getattr(self.llm, "model", ""), None, None, time.time() - t0)


class FakeProvider:
    """Deterministic provider for tests and offline development."""

    def __init__(self, replies: Optional[List[str]] = None, fn: Optional[Callable[[str], str]] = None, **_: Any):
        self.name = "fake"
        self.replies, self.fn, self.calls = list(replies or []), fn, []

    def complete(self, system, messages, *, json_mode=False, max_tokens=4096, temperature=None):
        self.calls.append((system, list(messages)))
        if self.fn:
            text = self.fn(messages[-1].content)
        elif self.replies:
            text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        else:
            text = "{}"
        return LLMResponse(text, "fake", "fake")


class FallbackProvider:
    """Try providers in order. Reports (never silently hides) every fallback via `on_fallback`."""

    def __init__(self, providers: Sequence[LLMProvider], on_fallback: Optional[Callable[[str, Exception], None]] = None):
        self.name = "fallback(" + ",".join(p.name for p in providers) + ")"
        self.providers, self.on_fallback = list(providers), on_fallback

    def complete(self, *a, **kw):
        last: Optional[Exception] = None
        for p in self.providers:
            try:
                return p.complete(*a, **kw)
            except Exception as e:
                last = e
                if self.on_fallback:
                    self.on_fallback(p.name, e)
        raise LLMError(f"all providers failed: {last}")


class GeminiAPI(VertexGemini):
    """Gemini via the off-cloud Developer API (api key), i.e. VertexGemini with use_vertex defaulted off."""

    def __init__(self, **kw: Any):
        kw.setdefault("use_vertex", False)
        super().__init__(**kw)


_REGISTRY = {
    "vertex": VertexGemini, "gemini": VertexGemini,
    "gemini_api": GeminiAPI, "google_ai": GeminiAPI,
    "openai_compat": OpenAICompat, "openai": OpenAICompat, "vllm": OpenAICompat, "ollama": OpenAICompat,
    "anthropic": AnthropicClaude, "claude": AnthropicClaude,
    "chia": ChiaLLM, "fake": FakeProvider,
}


def make_provider(spec: Dict[str, Any]) -> LLMProvider:
    spec = dict(spec)
    kind = spec.pop("provider")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown provider {kind!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**spec)


def extract_json(text: str) -> Any:
    """Parse the first JSON value in `text`, tolerating markdown fences and prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        for i, ch in enumerate(t):
            if ch in "[{":
                try:
                    return dec.raw_decode(t[i:])[0]
                except json.JSONDecodeError:
                    continue
        raise
