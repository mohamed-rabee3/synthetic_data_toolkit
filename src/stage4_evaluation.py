import asyncio
import copy
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.llm_client import UnifiedLLMClient
from src.utils import load_prompt, read_jsonl, render_prompt, write_json, write_jsonl

logger = logging.getLogger("pipeline.stage4")


# ---------------------------------------------------------------------------
# 4a. Clustering
# ---------------------------------------------------------------------------


def cluster_items(
    embeddings: np.ndarray,
    config: dict,
) -> np.ndarray:
    cluster_cfg = config["evaluation"]["clustering"]
    method = cluster_cfg.get("method", "hdbscan")

    if method == "hdbscan":
        hdb_cfg = cluster_cfg.get("hdbscan", {})
        logger.info(
            "Clustering with HDBSCAN (min_cluster_size=%d, min_samples=%d)",
            hdb_cfg.get("min_cluster_size", 10),
            hdb_cfg.get("min_samples", 5),
        )

        clusterer = HDBSCAN(
            min_cluster_size=hdb_cfg.get("min_cluster_size", 10),
            min_samples=hdb_cfg.get("min_samples", 5),
            metric=hdb_cfg.get("metric", "euclidean"),
        )
        labels = clusterer.fit_predict(embeddings.astype(np.float64))

        noise_count = int(np.sum(labels == -1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("HDBSCAN found %d clusters, %d noise points", n_clusters, noise_count)

        if hdb_cfg.get("assign_noise_to_nearest", True) and noise_count > 0:
            from sklearn.neighbors import NearestNeighbors

            non_noise_mask = labels != -1
            if non_noise_mask.any():
                nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
                nn.fit(embeddings[non_noise_mask])
                noise_indices = np.where(labels == -1)[0]
                _, nearest = nn.kneighbors(embeddings[noise_indices])
                non_noise_indices = np.where(non_noise_mask)[0]
                for i, noise_idx in enumerate(noise_indices):
                    labels[noise_idx] = labels[non_noise_indices[nearest[i][0]]]
                logger.info("Assigned %d noise points to nearest clusters", noise_count)

    elif method == "kmeans" and cluster_cfg.get("kmeans", {}).get("enabled", False):
        km_cfg = cluster_cfg["kmeans"]
        k_min = km_cfg.get("k_range_min", 3)
        k_max = km_cfg.get("k_range_max", 30)
        rs = km_cfg.get("random_state", 42)

        best_k = k_min
        best_score = -1

        for k in range(k_min, min(k_max + 1, len(embeddings))):
            km = KMeans(n_clusters=k, random_state=rs, n_init=10)
            trial_labels = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, trial_labels)
            if score > best_score:
                best_score = score
                best_k = k

        logger.info("K-Means: best k=%d (silhouette=%.3f)", best_k, best_score)
        km = KMeans(n_clusters=best_k, random_state=rs, n_init=10)
        labels = km.fit_predict(embeddings)
    else:
        labels = np.zeros(len(embeddings), dtype=int)

    return labels


# ---------------------------------------------------------------------------
# 4b. Balanced Proportional Sampling
# ---------------------------------------------------------------------------


def split_train_eval(
    items: list[dict],
    labels: np.ndarray,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    split_cfg = config["evaluation"]["splitting"]
    eval_fraction = split_cfg.get("eval_fraction", 0.12)
    min_per_cluster = split_cfg.get("min_items_per_cluster", 2)
    rs = split_cfg.get("random_state", 42)
    stratify_fields = split_cfg.get("stratify_by", [])

    rng = random.Random(rs)
    total = len(items)
    eval_budget = max(1, int(total * eval_fraction))

    # Group items by cluster
    cluster_groups: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        items[i]["cluster_id"] = int(label)
        cluster_groups[int(label)].append(i)

    # Calculate per-cluster allocation
    cluster_alloc: dict[int, int] = {}
    for cid, indices in cluster_groups.items():
        proportion = len(indices) / total
        alloc = max(min_per_cluster, int(eval_budget * proportion))
        alloc = min(alloc, len(indices))
        cluster_alloc[cid] = alloc

    # Adjust total
    total_alloc = sum(cluster_alloc.values())
    if total_alloc > eval_budget:
        scale = eval_budget / total_alloc
        for cid in cluster_alloc:
            cluster_alloc[cid] = max(
                min_per_cluster,
                int(cluster_alloc[cid] * scale),
            )

    eval_indices: set[int] = set()
    for cid, alloc in cluster_alloc.items():
        indices = cluster_groups[cid]
        rng.shuffle(indices)

        if stratify_fields:
            selected = _stratified_sample(
                items, indices, alloc, stratify_fields, rng
            )
        else:
            selected = indices[:alloc]

        eval_indices.update(selected)

    eval_items = [items[i] for i in sorted(eval_indices)]
    train_items = [items[i] for i in range(total) if i not in eval_indices]

    logger.info(
        "Split: %d train, %d eval (%.1f%%)",
        len(train_items),
        len(eval_items),
        len(eval_items) / total * 100,
    )
    return train_items, eval_items


def _stratified_sample(
    items: list[dict],
    indices: list[int],
    n: int,
    fields: list[str],
    rng: random.Random,
) -> list[int]:
    if n >= len(indices):
        return indices

    # Simple proportional stratification on the first field
    field = fields[0]
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        val = items[idx].get(field, "unknown")
        groups[str(val)].append(idx)

    selected: list[int] = []
    for group_val, group_indices in groups.items():
        proportion = len(group_indices) / len(indices)
        group_alloc = max(1, int(n * proportion))
        rng.shuffle(group_indices)
        selected.extend(group_indices[:group_alloc])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [i for i in indices if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected


# ---------------------------------------------------------------------------
# 4c. Rephrased Eval Split
# ---------------------------------------------------------------------------


async def rephrase_eval_items(
    eval_items: list[dict],
    llm_client: UnifiedLLMClient,
    config: dict,
) -> list[dict]:
    rephrase_cfg = config["evaluation"]["rephrasing"]
    prompts_dir = config["paths"]["prompts_dir"]
    prompt_template = load_prompt(prompts_dir, rephrase_cfg["prompt_file"])
    batch_size = rephrase_cfg.get("batch_size", 10)

    rephrased_items: list[dict] = []

    async def rephrase_one(item: dict) -> dict:
        prompt = render_prompt(
            prompt_template,
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            evaluation_criteria=item.get("evaluation_criteria", ""),
            category=item.get("category", ""),
            difficulty=item.get("difficulty", ""),
        )

        resp = await llm_client.call(
            role="judge",
            messages=[{"role": "user", "content": prompt}],
            temperature=rephrase_cfg.get("temperature", 0.7),
            max_tokens=rephrase_cfg.get("max_tokens", 1024),
            json_mode=True,
        )

        if resp.success:
            try:
                result = json.loads(resp.text)
                rephrased = copy.deepcopy(item)
                rephrased["original_question"] = item["question"]
                rephrased["original_answer"] = item["answer"]
                rephrased["original_evaluation_criteria"] = item["evaluation_criteria"]
                rephrased["question"] = result.get("question", item["question"])
                rephrased["answer"] = result.get("answer", item["answer"])
                if rephrase_cfg.get("generalize_evaluation_criteria", True):
                    rephrased["evaluation_criteria"] = result.get(
                        "evaluation_criteria", item["evaluation_criteria"]
                    )
                return rephrased
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: return original with markers
        fallback = copy.deepcopy(item)
        fallback["original_question"] = item["question"]
        fallback["original_answer"] = item["answer"]
        fallback["original_evaluation_criteria"] = item["evaluation_criteria"]
        fallback["rephrase_failed"] = True
        return fallback

    for i in range(0, len(eval_items), batch_size):
        batch = eval_items[i : i + batch_size]
        results = await asyncio.gather(*[rephrase_one(item) for item in batch])
        rephrased_items.extend(results)
        logger.info(
            "Rephrasing: %d/%d complete",
            min(i + batch_size, len(eval_items)),
            len(eval_items),
        )

    # Optional dialect re-validation
    if rephrase_cfg.get("re_validate_dialect", True):
        from src.stage3_cleaning import _count_markers

        dialect_cfg = config["cleaning"]["dialect"]
        markers = dialect_cfg.get("saudi_markers", [])
        a_min = dialect_cfg.get("answer_min_markers", 2)

        failures = 0
        for item in rephrased_items:
            a_count = _count_markers(item.get("answer", ""), markers)
            if a_count < a_min and not item.get("rephrase_failed"):
                item["dialect_revalidation_failed"] = True
                failures += 1

        if failures:
            logger.warning("%d rephrased items failed dialect re-validation", failures)

    rephrase_failures = sum(1 for i in rephrased_items if i.get("rephrase_failed"))
    logger.info(
        "Rephrasing complete: %d items (%d failures)",
        len(rephrased_items),
        rephrase_failures,
    )
    return rephrased_items


# ---------------------------------------------------------------------------
# 4d. Eval Mirror + Final Assembly
# ---------------------------------------------------------------------------


def create_eval_mirror(eval_items: list[dict], rephrased_items: list[dict]) -> list[dict]:
    mirror = []
    for orig, rephrased in zip(eval_items, rephrased_items):
        m = copy.deepcopy(orig)
        m["rephrased_question"] = rephrased.get("question", "")
        m["rephrased_answer"] = rephrased.get("answer", "")
        mirror.append(m)
    return mirror


def assemble_final_outputs(
    train_items: list[dict],
    eval_items: list[dict],
    rephrased_items: list[dict],
    mirror_items: list[dict],
    config: dict,
) -> dict:
    include_mirror = config["evaluation"]["eval_mirror"].get("include_in_training_set", True)

    # Mark mirror items in training set
    eval_question_set = {item.get("question", "") for item in eval_items}

    for item in train_items:
        item["is_eval_mirror"] = False

    if include_mirror:
        for item in mirror_items:
            train_copy = copy.deepcopy(item)
            train_copy["is_eval_mirror"] = True
            if "rephrased_question" in train_copy:
                del train_copy["rephrased_question"]
            if "rephrased_answer" in train_copy:
                del train_copy["rephrased_answer"]
            train_items.append(train_copy)

    # Build report
    train_non_mirror = sum(1 for i in train_items if not i.get("is_eval_mirror"))
    train_mirror = sum(1 for i in train_items if i.get("is_eval_mirror"))

    cluster_dist: dict[str, dict] = defaultdict(lambda: {"train": 0, "eval": 0})
    for item in train_items:
        cid = str(item.get("cluster_id", 0))
        cluster_dist[cid]["train"] += 1
    for item in rephrased_items:
        cid = str(item.get("cluster_id", 0))
        cluster_dist[cid]["eval"] += 1

    diff_dist = {"train": Counter(), "eval": Counter()}
    cat_dist = {"train": Counter(), "eval": Counter()}
    for item in train_items:
        diff_dist["train"][item.get("difficulty", "unknown")] += 1
        cat_dist["train"][item.get("category", "unknown")] += 1
    for item in rephrased_items:
        diff_dist["eval"][item.get("difficulty", "unknown")] += 1
        cat_dist["eval"][item.get("category", "unknown")] += 1

    rephrase_failures = sum(1 for i in rephrased_items if i.get("rephrase_failed"))

    report = {
        "total_items": train_non_mirror + len(rephrased_items),
        "train_items": train_non_mirror,
        "eval_items": len(rephrased_items),
        "eval_mirror_in_train": train_mirror,
        "clusters_found": len(cluster_dist),
        "cluster_distribution": dict(cluster_dist),
        "difficulty_distribution": {
            "train": dict(diff_dist["train"]),
            "eval": dict(diff_dist["eval"]),
        },
        "category_distribution": {
            "train": dict(cat_dist["train"]),
            "eval": dict(cat_dist["eval"]),
        },
        "rephrasing_failures": rephrase_failures,
    }

    return report


# ---------------------------------------------------------------------------
# Main Stage 4 Entry Point
# ---------------------------------------------------------------------------


async def run_stage4(
    config: dict,
    llm_client: UnifiedLLMClient,
    cleaned_items: list[dict],
    embeddings: np.ndarray,
):
    output_dir = Path(config["paths"]["output_dir"])
    eval_cfg = config["evaluation"]
    output_cfg = eval_cfg["output"]

    train_path = output_dir / output_cfg["train_file"]
    eval_rephrased_path = output_dir / output_cfg["eval_rephrased_file"]
    eval_mirror_path = output_dir / output_cfg["eval_mirror_file"]
    report_path = output_dir / output_cfg["split_report_file"]

    if train_path.exists():
        logger.info("Stage 4 outputs already exist, skipping")
        return

    logger.info("Stage 4: Creating evaluation dataset from %d items", len(cleaned_items))

    # 4a. Clustering
    labels = cluster_items(embeddings, config)

    # 4b. Balanced split
    train_items, eval_items = split_train_eval(cleaned_items, labels, config)

    # 4c. Rephrase eval items
    rephrased_items = await rephrase_eval_items(eval_items, llm_client, config)

    # 4d. Create eval mirror
    mirror_items = create_eval_mirror(eval_items, rephrased_items)

    # Assemble and save
    report = assemble_final_outputs(
        train_items, eval_items, rephrased_items, mirror_items, config
    )

    write_jsonl(train_path, train_items)
    write_jsonl(eval_rephrased_path, rephrased_items)
    write_jsonl(eval_mirror_path, mirror_items)
    write_json(report_path, report)

    logger.info(
        "Stage 4 complete: train=%d, eval_rephrased=%d, eval_mirror=%d",
        len(train_items),
        len(rephrased_items),
        len(mirror_items),
    )
