import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("pipeline.embeddings")

_model_cache = {}


def get_embedding_model(config: dict):
    from sentence_transformers import SentenceTransformer

    model_name = config.get("model_name", "intfloat/multilingual-e5-large")
    device = config.get("device", "cpu")

    cache_key = f"{model_name}_{device}"
    if cache_key not in _model_cache:
        logger.info("Loading embedding model: %s on %s", model_name, device)
        _model_cache[cache_key] = SentenceTransformer(model_name, device=device)
    return _model_cache[cache_key]


def embed_texts(
    texts: list[str],
    config: dict,
    prefix: str = "query: ",
) -> np.ndarray:
    model = get_embedding_model(config)
    batch_size = config.get("batch_size", 64)

    prefixed = [f"{prefix}{t}" for t in texts]

    logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


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
