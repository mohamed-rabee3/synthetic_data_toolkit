import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("pipeline.embeddings")

_model_cache = {}
_genai_client = None


def _get_genai_client(config: dict):
    global _genai_client
    if _genai_client is None:
        from google import genai
        from google.genai.types import HttpOptions

        api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google embedding requires an API key. "
                "Set GOOGLE_API_KEY in .env or embedding_model.api_key in cleaning.yaml"
            )

        gemini_mode = os.getenv("GEMINI_MODE", "ai_studio")
        if gemini_mode == "vertex_ai":
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
            _genai_client = genai.Client(
                vertexai=True,
                api_key=api_key,
                http_options=HttpOptions(api_version="v1"),
            )
            logger.info("Embedding client: Vertex AI express mode")
        else:
            os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            _genai_client = genai.Client(api_key=api_key)
            logger.info("Embedding client: AI Studio mode")
    return _genai_client


def _embed_google(texts: list[str], config: dict) -> np.ndarray:
    client = _get_genai_client(config)
    model = config.get("model_name", "gemini-embedding-001")
    batch_size = config.get("batch_size", 64)

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(model=model, contents=batch)
        all_embeddings.extend(e.values for e in result.embeddings)
        logger.debug("Embedded batch %d-%d / %d", i + 1, i + len(batch), len(texts))

    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


def _embed_sentence_transformers(texts: list[str], config: dict) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_name = config.get("model_name", "intfloat/multilingual-e5-large")
    device = config.get("device", "cpu")
    batch_size = config.get("batch_size", 64)

    cache_key = f"{model_name}_{device}"
    if cache_key not in _model_cache:
        logger.info("Loading embedding model: %s on %s", model_name, device)
        _model_cache[cache_key] = SentenceTransformer(model_name, device=device)
    model = _model_cache[cache_key]

    prefix = "query: "
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


def embed_texts(
    texts: list[str],
    config: dict,
    prefix: str = "query: ",
) -> np.ndarray:
    provider = config.get("provider", "sentence_transformers")
    logger.info("Embedding %d texts (provider=%s)", len(texts), provider)

    if provider == "google_genai":
        return _embed_google(texts, config)
    return _embed_sentence_transformers(texts, config)


def save_embeddings(embeddings: np.ndarray, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info("Saved embeddings (%s) to %s", embeddings.shape, path)


def load_embeddings(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    embeddings = np.load(path)
    logger.info("Loaded embeddings (%s) from %s", embeddings.shape, path)
    return embeddings
