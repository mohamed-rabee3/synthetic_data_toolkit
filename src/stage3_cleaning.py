import asyncio
import json
import logging
import random
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from rapidfuzz import fuzz

from src.embeddings import embed_texts, load_embeddings, save_embeddings
from src.llm_client import UnifiedLLMClient
from src.utils import (
    load_prompt,
    normalize_text,
    read_jsonl,
    render_prompt,
    write_json,
    write_jsonl,
)

logger = logging.getLogger("pipeline.stage3")


# ---------------------------------------------------------------------------
# 3a. Structural Validation
# ---------------------------------------------------------------------------


def validate_structural(items: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    cfg = config["cleaning"]["structural"]
    required = set(cfg.get("required_fields", []))
    allowed_cats = set(cfg.get("allowed_categories", []))
    allowed_diffs = set(cfg.get("allowed_difficulties", []))

    q_min = cfg.get("question_min_chars", 15)
    q_max = cfg.get("question_max_chars", 500)
    a_min = cfg.get("answer_min_chars", 30)
    a_max = cfg.get("answer_max_chars", 2000)
    ec_min = cfg.get("evaluation_criteria_min_chars", 30)
    ec_max = cfg.get("evaluation_criteria_max_chars", 1000)

    valid = []
    discarded = []

    for item in items:
        reason = None

        missing = required - set(item.keys())
        if missing:
            reason = f"missing_fields: {missing}"
        elif not all(isinstance(item.get(f), str) and item.get(f) for f in required):
            reason = "empty_required_field"
        elif not (q_min <= len(item.get("question", "")) <= q_max):
            reason = f"question_length: {len(item.get('question', ''))}"
        elif not (a_min <= len(item.get("answer", "")) <= a_max):
            reason = f"answer_length: {len(item.get('answer', ''))}"
        elif len(item.get("evaluation_criteria", "")) < ec_min:
            reason = f"eval_criteria_too_short: {len(item.get('evaluation_criteria', ''))}"
        elif ec_max and len(item.get("evaluation_criteria", "")) > ec_max:
            reason = f"eval_criteria_too_long: {len(item.get('evaluation_criteria', ''))}"
        elif item.get("category") not in allowed_cats:
            reason = f"invalid_category: {item.get('category')}"
        elif item.get("difficulty") not in allowed_diffs:
            reason = f"invalid_difficulty: {item.get('difficulty')}"

        if reason:
            item["discard_reason"] = reason
            discarded.append(item)
        else:
            valid.append(item)

    logger.info("Structural validation: %d valid, %d discarded", len(valid), len(discarded))
    return valid, discarded


# ---------------------------------------------------------------------------
# 3b. Saudi Dialect Validation
# ---------------------------------------------------------------------------


def _count_markers(text: str, markers: list[str]) -> int:
    count = 0
    for marker in markers:
        if marker in text:
            count += 1
    return count


def validate_dialect_markers(
    items: list[dict], config: dict
) -> tuple[list[dict], list[dict]]:
    cfg = config["cleaning"]["dialect"]
    markers = cfg.get("saudi_markers", [])
    q_min = cfg.get("question_min_markers", 1)
    a_min = cfg.get("answer_min_markers", 2)
    msa_indicators = cfg.get("msa_indicators", [])
    msa_max = cfg.get("msa_max_indicators", 3)

    valid = []
    discarded = []

    for item in items:
        q_count = _count_markers(item.get("question", ""), markers)
        a_count = _count_markers(item.get("answer", ""), markers)

        item["dialect_marker_count"] = q_count + a_count

        if q_count < q_min or a_count < a_min:
            item["discard_reason"] = (
                f"insufficient_dialect_markers: q={q_count}<{q_min} or a={a_count}<{a_min}"
            )
            discarded.append(item)
            continue

        msa_count = _count_markers(item.get("answer", ""), msa_indicators)
        if msa_count > msa_max:
            item["msa_flagged"] = True
            logger.debug("MSA flagged (count=%d): %.100s", msa_count, item.get("question", ""))

        valid.append(item)

    logger.info("Dialect marker check: %d valid, %d discarded", len(valid), len(discarded))
    return valid, discarded


async def validate_dialect_llm(
    items: list[dict],
    llm_client: UnifiedLLMClient,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    cfg = config["cleaning"]["dialect"]["llm_dialect_check"]
    if not cfg.get("enabled", False):
        return items, []

    sample_pct = cfg.get("sample_percentage", 15)
    min_score = cfg.get("min_score", 4)
    prompts_dir = config["paths"]["prompts_dir"]
    prompt_template = load_prompt(prompts_dir, cfg["prompt_file"])

    sample_size = max(1, int(len(items) * sample_pct / 100))
    sampled_indices = set(random.sample(range(len(items)), min(sample_size, len(items))))

    concurrency_cfg = config["cleaning"].get("stage_concurrency", {})
    semaphore = asyncio.Semaphore(concurrency_cfg.get("max_concurrent_llm_calls", 10))
    batch_size = concurrency_cfg.get("batch_size", 20)

    failed_indices: set[int] = set()

    async def check_one(idx: int):
        item = items[idx]
        prompt = render_prompt(
            prompt_template,
            question=item.get("question", ""),
            answer=item.get("answer", ""),
        )
        async with semaphore:
            resp = await llm_client.call(
                role="judge",
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.get("temperature", 0.1),
                max_tokens=cfg.get("max_tokens", 200),
                json_mode=True,
            )
        if resp.success:
            try:
                result = json.loads(resp.text)
                score = result.get("score", 5)
                if score < min_score:
                    failed_indices.add(idx)
            except (json.JSONDecodeError, TypeError):
                pass

    # Process in batches
    indices_to_check = sorted(sampled_indices)
    for i in range(0, len(indices_to_check), batch_size):
        batch = indices_to_check[i : i + batch_size]
        await asyncio.gather(*[check_one(idx) for idx in batch])

    valid = []
    discarded = []
    for i, item in enumerate(items):
        if i in failed_indices:
            item["discard_reason"] = "llm_dialect_check_failed"
            discarded.append(item)
        else:
            valid.append(item)

    logger.info(
        "LLM dialect check (%d sampled): %d valid, %d discarded",
        len(sampled_indices),
        len(valid),
        len(discarded),
    )
    return valid, discarded


# ---------------------------------------------------------------------------
# 3c. Answer Grounding Validation
# ---------------------------------------------------------------------------


async def validate_grounding(
    items: list[dict],
    llm_client: UnifiedLLMClient,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    cfg = config["cleaning"]["grounding"]
    llm_cfg = cfg.get("llm_grounding_check", {})
    if not llm_cfg.get("enabled", True):
        return items, []

    actions = llm_cfg.get("actions", {})
    prompts_dir = config["paths"]["prompts_dir"]
    prompt_template = load_prompt(prompts_dir, llm_cfg["prompt_file"])

    sample_pct = llm_cfg.get("sample_percentage", 100)
    if sample_pct < 100:
        sample_size = max(1, int(len(items) * sample_pct / 100))
        check_indices = set(random.sample(range(len(items)), min(sample_size, len(items))))
    else:
        check_indices = set(range(len(items)))

    concurrency_cfg = config["cleaning"].get("stage_concurrency", {})
    semaphore = asyncio.Semaphore(concurrency_cfg.get("max_concurrent_llm_calls", 10))
    batch_size = concurrency_cfg.get("batch_size", 20)

    classifications: dict[int, str] = {}

    async def check_one(idx: int):
        item = items[idx]
        prompt = render_prompt(
            prompt_template,
            chunk_text=item.get("context", ""),
            question=item.get("question", ""),
            answer=item.get("answer", ""),
        )
        async with semaphore:
            resp = await llm_client.call(
                role="judge",
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_cfg.get("temperature", 0.1),
                max_tokens=llm_cfg.get("max_tokens", 300),
                json_mode=True,
            )
        if resp.success:
            try:
                result = json.loads(resp.text)
                classifications[idx] = result.get("classification", "fully_supported")
            except (json.JSONDecodeError, TypeError):
                classifications[idx] = "fully_supported"
        else:
            classifications[idx] = "fully_supported"

    indices_to_check = sorted(check_indices)
    for i in range(0, len(indices_to_check), batch_size):
        batch = indices_to_check[i : i + batch_size]
        await asyncio.gather(*[check_one(idx) for idx in batch])
        logger.info(
            "Grounding check: batch %d/%d complete",
            i // batch_size + 1,
            (len(indices_to_check) + batch_size - 1) // batch_size,
        )

    valid = []
    discarded = []
    for i, item in enumerate(items):
        cls = classifications.get(i, "fully_supported")
        action = actions.get(cls, "keep")
        item["grounding_classification"] = cls
        if action == "discard":
            item["discard_reason"] = f"grounding_{cls}"
            discarded.append(item)
        else:
            valid.append(item)

    logger.info("Grounding check: %d valid, %d discarded", len(valid), len(discarded))
    return valid, discarded


# ---------------------------------------------------------------------------
# 3d. Deduplication
# ---------------------------------------------------------------------------


def dedup_exact(items: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    cfg = config["cleaning"]["deduplication"]["exact_match"]
    if not cfg.get("enabled", True):
        return items, []

    do_normalize = cfg.get("normalize", True)
    seen: set[str] = set()
    valid = []
    discarded = []

    for item in items:
        q = item.get("question", "")
        key = normalize_text(q) if do_normalize else q
        if key in seen:
            item["discard_reason"] = "exact_duplicate"
            discarded.append(item)
        else:
            seen.add(key)
            valid.append(item)

    logger.info("Exact dedup: %d valid, %d discarded", len(valid), len(discarded))
    return valid, discarded


def dedup_fuzzy(items: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    cfg = config["cleaning"]["deduplication"]["fuzzy"]
    if not cfg.get("enabled", True):
        return items, []

    threshold = cfg.get("similarity_threshold", 0.85) * 100
    keep_strategy = cfg.get("keep_strategy", "longer_answer")
    warn_threshold = cfg.get("brute_force_warn_threshold", 20000)

    if len(items) > warn_threshold:
        logger.warning(
            "Fuzzy dedup: %d items exceeds brute_force_warn_threshold (%d). "
            "Consider switching to 'arabic_stem' blocking.",
            len(items),
            warn_threshold,
        )

    questions = [item.get("question", "") for item in items]
    to_remove: set[int] = set()

    for i in range(len(questions)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(questions)):
            if j in to_remove:
                continue
            score = fuzz.ratio(questions[i], questions[j])
            if score >= threshold:
                if keep_strategy == "longer_answer":
                    shorter = (
                        j
                        if len(items[i].get("answer", ""))
                        >= len(items[j].get("answer", ""))
                        else i
                    )
                else:
                    shorter = j
                to_remove.add(shorter)

    valid = []
    discarded = []
    for i, item in enumerate(items):
        if i in to_remove:
            item["discard_reason"] = "fuzzy_duplicate"
            discarded.append(item)
        else:
            valid.append(item)

    logger.info("Fuzzy dedup: %d valid, %d discarded", len(valid), len(discarded))
    return valid, discarded


def dedup_semantic(
    items: list[dict],
    embeddings: np.ndarray,
    config: dict,
) -> tuple[list[dict], list[dict], np.ndarray]:
    cfg = config["cleaning"]["deduplication"]["semantic"]
    if not cfg.get("enabled", True):
        return items, [], embeddings

    threshold = cfg.get("cosine_threshold", 0.92)
    keep_strategy = cfg.get("keep_strategy", "better_rubric")

    if cfg.get("calibration_mode", False):
        _run_calibration(embeddings, cfg)

    n = len(items)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    k = min(20, n)
    scores, indices = index.search(embeddings.astype(np.float32), k)

    to_remove: set[int] = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j_pos in range(1, k):
            j = int(indices[i][j_pos])
            if j == i or j in to_remove:
                continue
            sim = float(scores[i][j_pos])
            if sim >= threshold:
                if keep_strategy == "better_rubric":
                    shorter = (
                        j
                        if len(items[i].get("evaluation_criteria", ""))
                        >= len(items[j].get("evaluation_criteria", ""))
                        else i
                    )
                else:
                    shorter = j
                to_remove.add(shorter)

    valid = []
    discarded = []
    valid_indices = []
    for i, item in enumerate(items):
        if i in to_remove:
            item["discard_reason"] = "semantic_duplicate"
            discarded.append(item)
        else:
            valid.append(item)
            valid_indices.append(i)

    valid_embeddings = embeddings[valid_indices] if valid_indices else np.array([])

    logger.info("Semantic dedup: %d valid, %d discarded", len(valid), len(discarded))
    return valid, discarded, valid_embeddings


def _run_calibration(embeddings: np.ndarray, cfg: dict):
    import matplotlib.pyplot as plt

    sample_size = min(cfg.get("calibration_sample_size", 300), len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample = embeddings[indices].astype(np.float32)

    sims = np.dot(sample, sample.T)
    upper_tri = sims[np.triu_indices(sample_size, k=1)]

    plt.figure(figsize=(10, 6))
    plt.hist(upper_tri, bins=100, edgecolor="black", alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Pairwise Cosine Similarity Distribution (Calibration)")
    plt.axvline(x=cfg.get("cosine_threshold", 0.92), color="r", linestyle="--", label="Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/coverage/calibration_histogram.png", dpi=150)
    plt.close()
    logger.info("Calibration histogram saved to reports/coverage/calibration_histogram.png")


# ---------------------------------------------------------------------------
# 3e. Coverage Visualization
# ---------------------------------------------------------------------------


async def generate_tags(
    items: list[dict],
    llm_client: UnifiedLLMClient,
    config: dict,
) -> list[dict]:
    cfg = config["cleaning"]["coverage"]["tag_histogram"]
    if not cfg.get("enabled", True):
        return items

    prompts_dir = config["paths"]["prompts_dir"]
    prompt_template = load_prompt(prompts_dir, cfg["prompt_file"])

    concurrency_cfg = config["cleaning"].get("stage_concurrency", {})
    semaphore = asyncio.Semaphore(concurrency_cfg.get("max_concurrent_llm_calls", 10))
    batch_size = concurrency_cfg.get("batch_size", 20)

    async def tag_one(idx: int):
        item = items[idx]
        prompt = render_prompt(prompt_template, question=item.get("question", ""))
        async with semaphore:
            resp = await llm_client.call(
                role="judge",
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.get("temperature", 0.1),
                max_tokens=cfg.get("max_tokens", 50),
                json_mode=True,
            )
        if resp.success:
            try:
                result = json.loads(resp.text)
                items[idx]["tags"] = result.get("tags", [])
            except (json.JSONDecodeError, TypeError):
                items[idx]["tags"] = []
        else:
            items[idx]["tags"] = []

    for i in range(0, len(items), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(items))))
        await asyncio.gather(*[tag_one(idx) for idx in batch_indices])

    logger.info("Tagging complete for %d items", len(items))
    return items


def normalize_tags(items: list[dict], config: dict, embeddings_cfg: dict):
    norm_cfg = config["cleaning"]["coverage"]["tag_histogram"].get("normalization", {})
    if not norm_cfg.get("enabled", True):
        return items

    all_tags = []
    for item in items:
        all_tags.extend(item.get("tags", []))

    if not all_tags:
        return items

    tag_counts = Counter(all_tags)
    unique_tags = list(tag_counts.keys())

    if norm_cfg.get("method") == "embedding_clustering":
        threshold = norm_cfg.get("merge_similarity_threshold", 0.85)
        tag_embeddings = embed_texts(unique_tags, embeddings_cfg, prefix="query: ")

        sims = np.dot(tag_embeddings, tag_embeddings.T)
        merge_map: dict[str, str] = {}

        for i in range(len(unique_tags)):
            if unique_tags[i] in merge_map:
                continue
            for j in range(i + 1, len(unique_tags)):
                if unique_tags[j] in merge_map:
                    continue
                if sims[i][j] >= threshold:
                    canonical = (
                        unique_tags[i]
                        if tag_counts[unique_tags[i]] >= tag_counts[unique_tags[j]]
                        else unique_tags[j]
                    )
                    other = unique_tags[j] if canonical == unique_tags[i] else unique_tags[i]
                    merge_map[other] = canonical

        for item in items:
            item["tags"] = [merge_map.get(t, t) for t in item.get("tags", [])]

        logger.info("Tag normalization: merged %d tags", len(merge_map))

    return items


def create_coverage_visualizations(
    items: list[dict],
    embeddings: np.ndarray,
    config: dict,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    tsne_cfg = config["cleaning"]["coverage"].get("tsne", {})
    if tsne_cfg.get("enabled", True) and len(embeddings) > 1:
        logger.info("Generating t-SNE plots")
        perplexity = min(tsne_cfg.get("perplexity", 30), len(embeddings) - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=tsne_cfg.get("n_iterations", 1000),
            random_state=tsne_cfg.get("random_state", 42),
        )
        coords = tsne.fit_transform(embeddings.astype(np.float32))

        prefix = tsne_cfg.get("output_prefix", "tsne_")
        color_by_fields = tsne_cfg.get("color_by", ["source_file", "category"])

        for field in color_by_fields:
            labels = [item.get(field, "unknown") for item in items]
            unique_labels = sorted(set(labels))
            color_map = {label: i for i, label in enumerate(unique_labels)}
            colors = [color_map[l] for l in labels]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                cmap="tab20",
                alpha=0.6,
                s=20,
            )
            plt.title(f"Q&A Distribution by {field}")
            plt.colorbar(scatter, ticks=range(len(unique_labels)))
            plt.tight_layout()

            out_path = reports_dir / f"{prefix}{field}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            logger.info("Saved t-SNE plot: %s", out_path)

    # Tag histogram
    tag_cfg = config["cleaning"]["coverage"].get("tag_histogram", {})
    if tag_cfg.get("enabled", True):
        all_tags = []
        for item in items:
            all_tags.extend(item.get("tags", []))
        if all_tags:
            tag_counts = Counter(all_tags)
            top_tags = tag_counts.most_common(30)
            tags, counts = zip(*top_tags)

            plt.figure(figsize=(14, 6))
            plt.barh(range(len(tags)), counts, color="steelblue")
            plt.yticks(range(len(tags)), tags)
            plt.xlabel("Count")
            plt.title("Topic Distribution (Top 30 Tags)")
            plt.tight_layout()

            out_path = reports_dir / f"{tag_cfg.get('output_file', 'topic_histogram')}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            logger.info("Saved tag histogram: %s", out_path)


# ---------------------------------------------------------------------------
# Main Stage 3 Entry Point
# ---------------------------------------------------------------------------


async def run_stage3(
    config: dict,
    llm_client: UnifiedLLMClient,
    raw_qa: list[dict],
) -> tuple[list[dict], np.ndarray]:
    intermediate_dir = Path(config["paths"]["intermediate_dir"])
    clean_cfg = config["cleaning"]
    output_cfg = clean_cfg["output"]

    cleaned_path = intermediate_dir / output_cfg["cleaned_qa_file"]
    discarded_path = intermediate_dir / output_cfg["discarded_file"]
    report_path = intermediate_dir / output_cfg["cleaning_report_file"]
    embeddings_path = intermediate_dir / output_cfg.get("embeddings_file", "embeddings.npy")

    if cleaned_path.exists():
        logger.info("Stage 3 outputs already exist, loading from disk")
        cleaned = read_jsonl(cleaned_path)
        embeddings = load_embeddings(embeddings_path)
        if embeddings is not None:
            return cleaned, embeddings

    logger.info("Stage 3: Cleaning %d raw QA pairs", len(raw_qa))
    all_discarded: list[dict] = []
    report: dict = {"initial_count": len(raw_qa)}

    # 3a. Structural validation
    items, discarded = validate_structural(raw_qa, config)
    all_discarded.extend(discarded)
    report["after_structural"] = len(items)

    # Substep order from config
    substep_order = clean_cfg.get("substep_order", ["grounding", "dialect", "tagging"])

    for substep in substep_order:
        if substep == "grounding":
            items, discarded = await validate_grounding(items, llm_client, config)
            all_discarded.extend(discarded)
            report["after_grounding"] = len(items)

        elif substep == "dialect":
            items, discarded = validate_dialect_markers(items, config)
            all_discarded.extend(discarded)
            report["after_dialect_markers"] = len(items)

            items, discarded = await validate_dialect_llm(items, llm_client, config)
            all_discarded.extend(discarded)
            report["after_dialect_llm"] = len(items)

        elif substep == "tagging":
            items = await generate_tags(items, llm_client, config)

    # 3d. Deduplication
    items, discarded = dedup_exact(items, config)
    all_discarded.extend(discarded)
    report["after_exact_dedup"] = len(items)

    items, discarded = dedup_fuzzy(items, config)
    all_discarded.extend(discarded)
    report["after_fuzzy_dedup"] = len(items)

    # Compute embeddings for semantic dedup and visualization
    embedding_cfg = clean_cfg["coverage"]["embedding_model"]
    questions = [item.get("question", "") for item in items]
    embeddings = embed_texts(questions, embedding_cfg)

    items, discarded, embeddings = dedup_semantic(items, embeddings, config)
    all_discarded.extend(discarded)
    report["after_semantic_dedup"] = len(items)

    # Normalize tags
    items = normalize_tags(items, config, embedding_cfg)

    # 3e. Coverage visualization
    create_coverage_visualizations(items, embeddings, config)

    # Save outputs
    report["final_count"] = len(items)
    report["total_discarded"] = len(all_discarded)

    write_jsonl(cleaned_path, items)
    write_jsonl(discarded_path, all_discarded)
    write_json(report_path, report)
    save_embeddings(embeddings, embeddings_path)

    logger.info(
        "Stage 3 complete: %d cleaned (%d discarded)",
        len(items),
        len(all_discarded),
    )
    return items, embeddings
