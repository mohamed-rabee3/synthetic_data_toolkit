import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from dotenv import load_dotenv

load_dotenv()


def setup_logging(config: dict) -> logging.Logger:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("log_file", "pipeline.log")

    logger = logging.getLogger("pipeline")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_configs(configs_dir: str | Path = "configs") -> dict:
    configs_dir = Path(configs_dir)
    pipeline = load_yaml(configs_dir / "pipeline.yaml")
    pipeline["chunking"] = load_yaml(configs_dir / "chunking.yaml")
    pipeline["generation"] = load_yaml(configs_dir / "generation.yaml")
    pipeline["cleaning"] = load_yaml(configs_dir / "cleaning.yaml")
    pipeline["evaluation"] = load_yaml(configs_dir / "evaluation.yaml")

    # Apply environment variable overrides
    _apply_env_overrides(pipeline)
    return pipeline


def _apply_env_overrides(config: dict):
    gen = config.setdefault("generation_llm", {})
    judge = config.setdefault("judge_llm", {})

    # ── Generation provider switch ──
    gen_provider = os.getenv("GENERATION_PROVIDER", "").lower()
    if gen_provider == "ollama":
        gen["provider"] = "ollama"
        ollama_mode = os.getenv("OLLAMA_MODE", "local")
        gen["mode"] = ollama_mode
        if os.getenv("OLLAMA_HOST"):
            gen["host"] = os.environ["OLLAMA_HOST"]
        if os.getenv("OLLAMA_API_KEY"):
            gen["api_key"] = os.environ["OLLAMA_API_KEY"]
        gen["model"] = os.getenv("OLLAMA_GENERATION_MODEL") or os.getenv("OLLAMA_MODEL") or gen.get("model", "qwen3")
        gen["think"] = judge.get("think", False)
        gen["timeout_seconds"] = judge.get("timeout_seconds", 600)
    elif gen_provider == "gemini" or not gen_provider:
        gen["provider"] = gen.get("provider", "gemini")
        if os.getenv("GEMINI_MODE"):
            gen["mode"] = os.environ["GEMINI_MODE"]
        if os.getenv("GOOGLE_API_KEY"):
            gen["api_key"] = os.environ["GOOGLE_API_KEY"]
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            gen["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        if os.getenv("GOOGLE_CLOUD_LOCATION"):
            gen["location"] = os.environ["GOOGLE_CLOUD_LOCATION"]

    # ── Judge — only apply Ollama env overrides if provider is ollama ──
    if judge.get("provider") == "ollama":
        if os.getenv("OLLAMA_MODE"):
            judge["mode"] = os.environ["OLLAMA_MODE"]
        if os.getenv("OLLAMA_HOST"):
            judge["host"] = os.environ["OLLAMA_HOST"]
        if os.getenv("OLLAMA_MODEL"):
            judge["model"] = os.environ["OLLAMA_MODEL"]
        if os.getenv("OLLAMA_API_KEY"):
            judge["api_key"] = os.environ["OLLAMA_API_KEY"]
    elif judge.get("provider") == "gemini":
        if os.getenv("GEMINI_MODE"):
            judge["mode"] = os.environ["GEMINI_MODE"]
        if os.getenv("GOOGLE_API_KEY"):
            judge["api_key"] = os.environ["GOOGLE_API_KEY"]
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            judge["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        if os.getenv("GOOGLE_CLOUD_LOCATION"):
            judge["location"] = os.environ["GOOGLE_CLOUD_LOCATION"]

    # ── Optional pipeline overrides ──
    if os.getenv("PIPELINE_DOMAIN_CONTEXT"):
        config["domain_context"] = os.environ["PIPELINE_DOMAIN_CONTEXT"]
    if os.getenv("LOG_LEVEL"):
        config.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"]


def load_prompt(prompts_dir: str | Path, filename: str) -> str:
    path = Path(prompts_dir) / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def render_prompt(template: str, **kwargs) -> str:
    """Safe prompt rendering that only replaces known {placeholder} keys.

    Unlike str.format(), JSON examples like {"score": 5} are left intact
    because the regex only matches {word_chars} — not {"quoted_key"...}.
    """
    def _replacer(match: re.Match) -> str:
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        return match.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3, etc.)."""
    return _THINK_RE.sub("", text)


_tiktoken_cache: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, tokenizer_config: dict) -> int:
    method = tokenizer_config.get("method", "tiktoken")

    if method == "tiktoken":
        encoding_name = tokenizer_config.get("tiktoken_encoding", "o200k_base")
        if encoding_name not in _tiktoken_cache:
            _tiktoken_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        return len(_tiktoken_cache[encoding_name].encode(text))

    if method == "character_ratio":
        ratio = tokenizer_config.get("chars_per_token", 3.5)
        return int(len(text) / ratio)

    raise ValueError(f"Unknown tokenizer method: {method}")


def write_jsonl(path: str | Path, items: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, items: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_json(path: str | Path, data: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dirs(config: dict):
    paths = config.get("paths", {})
    for key in ["intermediate_dir", "output_dir", "reports_dir", "checkpoint_dir"]:
        d = paths.get(key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


def repair_json(raw: str) -> str | None:
    text = strip_thinking(raw).strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Fix trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Try parsing directly
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Try extracting a JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidate = match.group()
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Try extracting a JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            json.loads(candidate)
            return f"[{candidate}]"
        except json.JSONDecodeError:
            pass

    return None
