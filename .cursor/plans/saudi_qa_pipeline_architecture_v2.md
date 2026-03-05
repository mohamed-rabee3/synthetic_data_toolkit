# Saudi Dialect Synthetic Q&A Dataset Pipeline
## Full Architectural Plan — Custom Implementation (No Distilabel)

---

## Pipeline Overview

This document describes the complete architecture for generating a high-quality synthetic Q&A training dataset in Saudi Arabic dialect (اللهجة السعودية) from company markdown documentation. The pipeline is implemented entirely in custom Python with no dependency on distilabel or any orchestration framework. All configuration is externalized into YAML files.

The pipeline has four major stages:

```
┌─────────────────────┐
│  Markdown Files      │  (already available)
│  (.md documents)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1: SEMANTIC TEXT CHUNKING                 │
│                                                  │
│  1a. Document Summary Generation (LLM call)      │
│  1b. Sentence Boundary Detection (Regex)         │
│  1c. Table Isolation (Regex)                     │
│  1d. Adaptive Chunk Assembly (token window)      │
│                                                  │
│  Output: chunks.jsonl + summaries.json           │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: Q&A GENERATION (Saudi Dialect)         │
│                                                  │
│  2a. Prompt Builder (3-layer architecture)        │
│  2b. Initial Generation Pass                     │
│  2c. Iterative Coverage Loop (per chunk)         │
│  2d. Multi-Model Merge (optional)                │
│                                                  │
│  Output: raw_qa.jsonl                            │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3: DATA CLEANING + COVERAGE ANALYSIS      │
│                                                  │
│  3a. Structural Validation                       │
│  3b. Saudi Dialect Validation                    │
│  3c. Answer Grounding Validation                 │
│  3d. Deduplication (3 levels)                    │
│  3e. Coverage Visualization                      │
│                                                  │
│  Output: cleaned_qa.jsonl + coverage_report/     │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 4: EVALUATION DATASET CREATION            │
│                                                  │
│  4a. Embed + Cluster (HDBSCAN primary)           │
│  4b. Balanced Proportional Sampling              │
│  4c. Rephrased Eval Split (LLM rewrite)          │
│  4d. Eval Mirror (verbatim copy)                 │
│                                                  │
│  Output: train.jsonl                             │
│          eval_rephrased.jsonl                    │
│          eval_mirror.jsonl                       │
└─────────────────────────────────────────────────┘
```

---

## Configuration Architecture

All runtime configuration lives in YAML files. No magic numbers or settings are hardcoded in Python source. The pipeline reads configuration at startup and passes the relevant section to each stage.

### File Structure

```
project/
├── configs/
│   ├── pipeline.yaml          # Master config: paths, global settings, stage toggles
│   ├── chunking.yaml          # Stage 1: chunking parameters
│   ├── generation.yaml        # Stage 2: LLM settings, prompts, iteration limits
│   ├── cleaning.yaml          # Stage 3: thresholds, dialect markers, dedup settings
│   └── evaluation.yaml        # Stage 4: clustering, splitting, rephrasing settings
├── prompts/
│   ├── summary_system.txt     # System prompt for document summarization
│   ├── summary_user.txt       # User prompt template for summarization
│   ├── qa_gen_system.txt      # System prompt for Q&A generation
│   ├── qa_gen_user.txt        # User prompt template for initial generation
│   ├── qa_gen_followup.txt    # User prompt template for iterative follow-up
│   ├── dialect_spec.txt       # Saudi dialect specification (appended to system prompt)
│   ├── grounding_check.txt    # Prompt for LLM-based grounding validation
│   ├── dialect_check.txt      # Prompt for LLM-based dialect scoring
│   ├── tagging.txt            # Prompt for topic tag assignment
│   └── rephrase.txt           # Prompt for eval set rephrasing + criteria generalization
├── src/
│   ├── stage1_chunking.py
│   ├── stage2_generation.py
│   ├── stage3_cleaning.py
│   ├── stage4_evaluation.py
│   ├── llm_client.py          # Unified LLM calling interface
│   ├── embeddings.py          # Embedding generation utilities
│   └── utils.py               # Shared helpers (token counting, IO, logging)
├── run_pipeline.py            # Master entry point
├── data/
│   ├── input/                 # Place markdown files here
│   ├── intermediate/          # Stage outputs (chunks, raw QA, cleaned QA)
│   └── output/                # Final train/eval splits
└── reports/
    └── coverage/              # Visualizations from Stage 3
```

### Why Separate YAML Files Instead of One

Each stage has enough configuration surface area that a single monolithic YAML becomes hard to maintain. Separate files allow:

- A team member to tune chunking parameters without touching generation prompts.
- Version control to show clearly which stage's config changed.
- Easier A/B testing by swapping one config file at a time.

The master `pipeline.yaml` ties them together and controls which stages run.

---

### Config File: `pipeline.yaml`

```yaml
# ─────────────────────────────────────────────
# Master pipeline configuration
# ─────────────────────────────────────────────

project_name: "saudi-qa-dataset"

paths:
  input_docs_dir: "data/input/"              # Directory containing .md files
  intermediate_dir: "data/intermediate/"      # Chunks, raw QA, cleaned QA
  output_dir: "data/output/"                  # Final train/eval splits
  reports_dir: "reports/coverage/"            # Visualizations
  prompts_dir: "prompts/"                     # All prompt template files
  checkpoint_dir: "data/intermediate/.checkpoints/"  # For fault tolerance

# Which stages to run (useful for re-running individual stages)
stages:
  run_chunking: true
  run_generation: true
  run_cleaning: true
  run_evaluation: true

# Domain context — injected into generation prompts
# Describe your company/documents in one sentence for the LLM
domain_context: "هذه وثائق لشركة تقنية سعودية متخصصة في حلول الدفع الإلكتروني"

# Global LLM settings (can be overridden per stage)
default_llm:
  provider: "openai"          # "openai" | "anthropic" | "azure_openai"
  model: "gpt-4o-mini"
  api_key_env_var: "OPENAI_API_KEY"   # Name of env var holding the key
  base_url: null                       # Override for Azure or proxies
  timeout_seconds: 120
  max_retries: 3
  retry_delay_seconds: 5
  requests_per_minute: 500             # Rate limit cap

# Logging
logging:
  level: "INFO"                # DEBUG | INFO | WARNING | ERROR
  log_file: "pipeline.log"
  log_failed_items: true       # Write failed/discarded items to a separate file
```

---

### Config File: `chunking.yaml`

```yaml
# ─────────────────────────────────────────────
# Stage 1: Semantic Text Chunking
# ─────────────────────────────────────────────

# Document summary generation
summary:
  enabled: true
  llm:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.3
    max_tokens: 300
  # If a document exceeds this token count, truncate before summarizing
  max_input_tokens: 4000
  system_prompt_file: "summary_system.txt"
  user_prompt_file: "summary_user.txt"

# Sentence boundary detection
sentence_detection:
  # Regex patterns for sentence endings
  # These are applied in order; first match wins
  arabic_sentence_endings: true        # Handle ؟ and ، and ۔
  # Known abbreviations that should NOT trigger a split
  abbreviations:
    - "e.g."
    - "i.e."
    - "vs."
    - "Dr."
    - "Inc."
    - "Ltd."
    - "etc."
    - "v."
    - "No."
    - "Sr."
    - "Jr."
  # Patterns that look like sentence boundaries but aren't
  ignore_decimal_numbers: true         # "3.14" is not a sentence end
  ignore_url_dots: true                # "www.example.com" is not 3 sentences

# Table isolation
table_detection:
  enabled: true
  # Minimum number of rows (including header + separator) to qualify as a table
  min_rows: 3
  # Include the nearest heading above the table as prefix context
  include_preceding_heading: true

# Chunk assembly
chunking:
  min_chunk_tokens: 200
  max_chunk_tokens: 5000
  overlap_sentences: 2                 # Repeat last N sentences at start of next chunk
  force_break_on_headings: true        # Always start a new chunk at any heading
  heading_levels_to_break_on:          # Which heading levels force a break
    - 1    # h1
    - 2    # h2
    - 3    # h3
  # If the final chunk of a file is below min_chunk_tokens, merge it backward
  merge_short_trailing_chunks: true

# Token counting
tokenizer:
  # Use the actual tokenizer of your generation model for accurate counts
  # Options: "tiktoken" (for OpenAI models), "transformers" (for HF models), "character_ratio"
  method: "tiktoken"
  # Only used if method is "tiktoken"
  tiktoken_encoding: "o200k_base"       # For GPT-4o family
  # Only used if method is "transformers"
  hf_tokenizer_name: null
  # Only used if method is "character_ratio" (fallback, least accurate)
  # WARNING: This is unreliable for Arabic. Use only if you cannot install tiktoken.
  chars_per_token: 3.5

# Output
output:
  chunks_file: "chunks.jsonl"           # Saved in intermediate_dir
  summaries_file: "summaries.json"      # Saved in intermediate_dir
```

---

### Config File: `generation.yaml`

```yaml
# ─────────────────────────────────────────────
# Stage 2: Q&A Generation
# ─────────────────────────────────────────────

# LLM for generation
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.8
  max_tokens: 4096
  # Response format — request JSON mode if provider supports it
  response_format: "json"

# Prompt files (loaded from prompts_dir)
prompts:
  system_prompt_file: "qa_gen_system.txt"
  user_prompt_file: "qa_gen_user.txt"
  followup_prompt_file: "qa_gen_followup.txt"

# Prompt template variables
# The user prompt files should contain placeholders like {document_summary},
# {chunk_text}, {domain_context}, {section_hierarchy}, {existing_questions}
# These are filled at runtime from chunk metadata + pipeline.yaml domain_context.

# Saudi dialect specification
# This entire block is injected verbatim into the system prompt
# so it can be tuned without touching Python code
dialect_spec:
  inject_into_system_prompt: true
  # The file containing detailed dialect instructions (lexical, grammatical, examples)
  # This is appended to the system prompt at runtime
  dialect_instructions_file: "dialect_spec.txt"

# Iterative coverage loop
iteration:
  max_iterations: 10                    # Safety cap — stop even if model keeps generating
  # How to format existing questions when feeding back
  # Options: "json_list" | "numbered_list" | "bullet_list"
  feedback_format: "numbered_list"
  # If a followup iteration returns broken JSON, how to handle it
  # Options: "retry_once" | "skip_iteration" | "stop_loop"
  parse_failure_strategy: "retry_once"
  # Max retries for a single iteration's parse failure before moving on
  parse_failure_max_retries: 2
  # The iterative loop runs INSIDE a single function per chunk.
  # It is NOT a multi-step DAG — it is a blocking loop that makes
  # multiple sequential LLM calls for one chunk before moving to the next.
  # This is simpler to implement and debug than a DAG-based approach.

  # Within-loop deduplication gate
  # After each iteration returns new questions, each new question is compared
  # against all accumulated questions using rapidfuzz character-level similarity.
  # If a new question exceeds the threshold against ANY existing question,
  # it is silently discarded as a near-duplicate before being added.
  # After discarding, if zero genuinely new questions remain, the iteration
  # counts as "stagnant" even though the model returned non-empty output.
  within_loop_dedup:
    enabled: true
    similarity_threshold: 0.80         # rapidfuzz.fuzz.ratio; discard new Q above this

  # Stagnation detection
  # If N consecutive iterations produce zero genuinely new questions
  # (after the within-loop dedup gate filters out near-duplicates),
  # terminate the loop early — the model is spinning its wheels.
  stagnation_limit: 2                  # Stop after N consecutive stagnant iterations

# Multi-model generation (optional)
multi_model:
  enabled: false
  # If enabled, run generation independently with each model, then merge in Stage 3.
  # Each model runs its OWN iterative coverage loop.
  # Cross-model question sharing does NOT happen during generation —
  # it happens post-hoc during deduplication in Stage 3.
  models:
    - provider: "openai"
      model: "gpt-4o-mini"
      label: "gpt4o-mini"
    - provider: "anthropic"
      model: "claude-sonnet-4-20250514"
      label: "claude-sonnet"
    # - provider: "openai"
    #   model: "gpt-4o"
    #   label: "gpt4o"

# Batch processing
batching:
  # How many chunks to process before writing a checkpoint
  checkpoint_every_n_chunks: 25
  # Concurrent chunk processing (set to 1 for sequential, >1 for async)
  concurrency: 3
  # If a chunk fails entirely (all retries exhausted), skip it and log
  skip_failed_chunks: true

# Output
output:
  raw_qa_file: "raw_qa.jsonl"           # Saved in intermediate_dir
  # Per-model output files (only if multi_model.enabled is true)
  # Named automatically: raw_qa_{label}.jsonl
```

---

### Config File: `cleaning.yaml`

```yaml
# ─────────────────────────────────────────────
# Stage 3: Data Cleaning & Coverage Analysis
# ─────────────────────────────────────────────

# ── 3a. Structural Validation ──

structural:
  required_fields:
    - "question"
    - "answer"
    - "evaluation_criteria"
    - "category"
    - "difficulty"
  question_min_chars: 15
  question_max_chars: 500
  answer_min_chars: 30
  answer_max_chars: 2000
  evaluation_criteria_min_chars: 30
  allowed_categories:
    - "factual"
    - "procedural"
    - "reasoning"
    - "troubleshooting"
    - "conceptual"
    - "comparative"
  allowed_difficulties:
    - "easy"
    - "medium"
    - "hard"
  # Attempt to repair broken JSON before discarding
  attempt_json_repair: true
  # Repair strategies applied in order:
  # 1. Strip markdown code fences (```json ... ```)
  # 2. Fix trailing commas before ] or }
  # 3. Attempt to extract JSON array with regex
  # 4. Give up and discard

# ── 3b. Saudi Dialect Validation ──

dialect:
  # Saudi dialect markers — a question/answer containing these is likely Saudi dialect
  saudi_markers:
    - "وش"
    - "ايش"
    - "إيش"
    - "ليش"
    - "ابي"
    - "أبي"
    - "ابغى"
    - "أبغى"
    - "اقدر"
    - "أقدر"
    - "وين"
    - "شلون"
    - "يعني"
    - "كذا"
    - "حق"
    - "عشان"
    - "علشان"
    - "زي"
    - "كيف"
    - "مره"
    - "واجد"
    - "حيل"
    - "ذحين"
    - "الحين"
    - "توه"
    - "يبي"
    - "تبي"
    - "نبي"
    - "طيب"
    - "اوكي"
    - "مو"
    - "ماهو"
    - "لازم"
    - "المفروض"
    - "قاعد"
    - "يقدر"
    - "تقدر"
    - "حقي"
    - "حقك"
    - "عندي"
    - "عندك"

  # Minimum marker counts
  question_min_markers: 1
  answer_min_markers: 2

  # MSA contamination — strong MSA words rarely used in Saudi dialect
  msa_indicators:
    - "لذلك"
    - "ينبغي"
    - "يتوجب"
    - "نظراً"
    - "بالتالي"
    - "علاوة على ذلك"
    - "إذ أن"
    - "من ثم"
    - "حيث أن"
    - "فضلاً عن"
    - "يتعين"
    - "على صعيد"
    - "تجدر الإشارة"
  msa_max_indicators: 3               # Flag if answer exceeds this count

  # LLM-based dialect check (optional, expensive)
  llm_dialect_check:
    enabled: true
    sample_percentage: 15              # Check 15% of pairs, randomly sampled
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
      temperature: 0.1
      max_tokens: 200
    prompt_file: "dialect_check.txt"
    min_score: 4                        # Out of 5; discard below this

# ── 3c. Answer Grounding Validation ──

grounding:
  # Language-aware grounding strategy
  # IMPORTANT: If source docs are in English/MSA and answers are in Saudi dialect,
  # token overlap is meaningless. This setting controls which method to use.
  source_language: "english"            # "english" | "msa" | "saudi_dialect" | "mixed"
  answer_language: "saudi_dialect"

  # Method 1: Token overlap (ONLY used when source and answer are same language)
  token_overlap:
    enabled: false                      # Disabled by default because of language mismatch
    min_overlap_ratio: 0.15

  # Method 2: LLM-based grounding check (ALWAYS use when languages differ)
  llm_grounding_check:
    enabled: true
    # Check every pair (not sampled) because grounding is critical
    sample_percentage: 100
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
      temperature: 0.1
      max_tokens: 300
    prompt_file: "grounding_check.txt"
    # The grounding prompt receives: chunk text, question, answer
    # It must output: "fully_supported" | "partially_supported" | "not_supported"
    # Action per result:
    actions:
      fully_supported: "keep"
      partially_supported: "keep"        # Keep but flag for optional human review
      not_supported: "discard"

# ── 3d. Deduplication ──

deduplication:
  # Level 1: Exact match (always on, zero cost)
  exact_match:
    enabled: true
    # Normalize before comparing: strip whitespace, normalize unicode
    normalize: true

  # Level 2: Fuzzy string similarity
  fuzzy:
    enabled: true
    # Library: "difflib" (stdlib) or "rapidfuzz" (faster, requires install)
    library: "rapidfuzz"
    similarity_threshold: 0.85          # Pairs above this are considered near-duplicates
    # When duplicates found, keep the one with the longer answer
    keep_strategy: "longer_answer"
    # Blocking strategy to reduce O(n²) pairwise comparisons
    # Options:
    #   "none"         — Brute force all-pairs. Fine for datasets under 20k items.
    #                    Arabic morphology makes n-gram blocking unreliable
    #                    (same root word has many surface forms due to prefixes/suffixes),
    #                    so brute force is the safest default.
    #   "arabic_stem"  — Block by shared Arabic stems using ISRIStemmer (requires nltk)
    #                    or tashaphyne. Two questions sharing at least one stem are compared.
    #                    More accurate for Arabic but adds an NLP dependency.
    blocking_strategy: "none"
    # If dataset exceeds this size with blocking_strategy "none", log a warning
    # suggesting the user switch to "arabic_stem" for performance
    brute_force_warn_threshold: 20000

  # Level 3: Semantic deduplication via embeddings
  semantic:
    enabled: true
    # NOTE: The cosine similarity threshold MUST be calibrated on your data.
    # Arabic dialect text may cluster tighter than English in multilingual models.
    # Recommended: run a calibration pass on 200-300 pairs, manually label
    # 30-50 true-duplicate and 30-50 non-duplicate pairs, then find the
    # threshold that best separates them.
    cosine_threshold: 0.92              # Starting point — calibrate before production
    calibration_mode: false             # If true, output similarity distribution histogram
    calibration_sample_size: 300        # Pairs to sample for calibration
    # When semantic duplicates found, keep the one with better evaluation_criteria
    keep_strategy: "better_rubric"

# ── 3e. Coverage Visualization ──

coverage:
  # Embedding model for visualization and semantic dedup
  embedding_model:
    provider: "sentence_transformers"
    model_name: "intfloat/multilingual-e5-large"
    # Batch size for embedding generation
    batch_size: 64
    # Device: "cpu" | "cuda" | "mps"
    device: "cpu"

  # t-SNE plot
  tsne:
    enabled: true
    perplexity: 30
    n_iterations: 1000
    random_state: 42
    # Color coding for the plot
    color_by:
      - "source_file"                   # One plot colored by source document
      - "category"                      # One plot colored by Q&A category
      - "generation_model"              # One plot colored by model (if multi-model used)
    output_format: "png"                # "png" | "html" (plotly interactive)
    output_prefix: "tsne_"             # Files: tsne_source_file.png, tsne_category.png, etc.

  # Tag-based topic histogram
  tag_histogram:
    enabled: true
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
      temperature: 0.1
      max_tokens: 50
    prompt_file: "tagging.txt"
    tags_per_question: 3                # Ask LLM for 1-3 tags per question
    # Tag normalization — to prevent "account-setup" vs "account-registration" fragmentation
    normalization:
      enabled: true
      # Strategy: After all tags are collected, cluster them with embeddings.
      # Tags within cosine similarity > threshold are merged under the most frequent form.
      method: "embedding_clustering"
      merge_similarity_threshold: 0.85
      # Alternative: provide a controlled vocabulary and map all tags to nearest entry
      # method: "controlled_vocabulary"
      # vocabulary_file: "tag_vocabulary.yaml"
    output_format: "png"
    output_file: "topic_histogram"

# ── Stage-level concurrency for LLM calls ──
# All LLM-based substeps in Stage 3 (dialect check, grounding check, tagging)
# share this concurrency config since they hit the same rate-limited LLM pool.
stage_concurrency:
  max_concurrent_llm_calls: 10
  requests_per_minute: 500             # Overrides global default for this stage
  batch_size: 20                       # Collect N items, send concurrently, wait for all, repeat

# Processing order for LLM-based substeps
# The three substeps run SEQUENTIALLY as sub-stages (not interleaved).
# Order matters because earlier steps may discard items, avoiding wasted LLM calls
# in later steps. Within each substep, items are processed in concurrent batches.
#
# Recommended order (most-discarding-first):
#   1. grounding check  — highest discard rate, removes hallucinated answers
#   2. dialect check    — moderate discard rate, removes non-Saudi content
#   3. tagging          — zero discard, only adds metadata (runs on survivors only)
substep_order:
  - "grounding"
  - "dialect"
  - "tagging"

# Output
output:
  cleaned_qa_file: "cleaned_qa.jsonl"
  discarded_file: "discarded_qa.jsonl"  # All removed items with discard reason
  cleaning_report_file: "cleaning_report.json"  # Statistics summary
```

---

### Config File: `evaluation.yaml`

```yaml
# ─────────────────────────────────────────────
# Stage 4: Evaluation Dataset Creation
# ─────────────────────────────────────────────

# ── 4a. Clustering ──

clustering:
  # Primary method: HDBSCAN (does not require pre-specifying K)
  # HDBSCAN is preferred over K-Means for this use case because:
  # - Arabic embeddings from multilingual models may not form spherical clusters
  # - HDBSCAN handles clusters of varying density and shape
  # - No need for the ambiguous "elbow method"
  method: "hdbscan"

  hdbscan:
    min_cluster_size: 10               # Minimum points to form a cluster
    min_samples: 5                      # Core point density threshold
    # metric: "euclidean" applied on the embedding space
    metric: "euclidean"
    # Items that HDBSCAN labels as noise (-1) are assigned to nearest cluster
    assign_noise_to_nearest: true

  # Fallback: K-Means (if you prefer explicit cluster count)
  kmeans:
    enabled: false
    # If enabled, determine K via silhouette score (more reliable than elbow for embeddings)
    k_selection_method: "silhouette"    # "silhouette" | "elbow"
    k_range_min: 3
    k_range_max: 30
    random_state: 42

  # Uses the same embedding model as coverage visualization (from cleaning.yaml)
  # No need to re-embed — reuses the embeddings already computed in Stage 3

# ── 4b. Eval Split ──

splitting:
  eval_fraction: 0.12                   # 12% of total dataset for eval
  # Proportional sampling: each cluster contributes proportionally
  # Also stratified within each cluster by:
  stratify_by:
    - "difficulty"                      # Proportional easy/medium/hard
    - "category"                        # Proportional factual/procedural/etc.
    - "source_file"                     # Avoid pulling all eval from one doc
  random_state: 42
  # Minimum items per cluster in eval set (if a cluster is too small, take at least this many)
  min_items_per_cluster: 2

# ── 4c. Rephrased Eval Split ──

rephrasing:
  llm:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7
    max_tokens: 1024
  prompt_file: "rephrase.txt"
  # The rephrasing prompt performs two tasks in a single LLM call:
  #
  # PART 1 — Rephrase Q&A:
  #   1. Reword the question (different phrasing, same meaning)
  #   2. Rewrite the answer (different structure, same information)
  #   3. Preserve Saudi dialect throughout
  #   4. Keep category and difficulty UNCHANGED
  #
  # PART 2 — Generalize evaluation criteria:
  #   5. Rewrite the evaluation_criteria rubric so it describes required
  #      content CONCEPTUALLY rather than referencing exact phrases from
  #      the original answer. This prevents rubric misalignment — if the
  #      original rubric says "must mention تسجيل الدخول" but the rephrased
  #      answer uses "الدخول على حسابك", the phrase-specific rubric would
  #      incorrectly flag the rephrased answer as incomplete.
  #   The generalized rubric should describe WHAT information must be present,
  #   not the exact words used to express it.
  generalize_evaluation_criteria: true
  batch_size: 10
  # Verify rephrased output still passes dialect validation
  re_validate_dialect: true

# ── 4d. Eval Mirror ──

eval_mirror:
  # The eval mirror is the ORIGINAL (un-rephrased) version of the eval items.
  # These items ARE INCLUDED in the training set intentionally.
  # This is by design — the mirror's purpose is to detect overfitting.
  #
  # During training, track three loss curves:
  #   1. Training loss (should decrease)
  #   2. Rephrased eval loss (should decrease if model generalizes)
  #   3. Eval mirror loss (if this drops much faster than rephrased → overfitting)
  #
  # Because mirror items appear in training, a dropping mirror loss is expected.
  # The SIGNAL is the GAP between mirror loss and rephrased loss.
  # A growing gap = the model memorizes surface forms but doesn't generalize.
  include_in_training_set: true

# Output
output:
  train_file: "train.jsonl"
  eval_rephrased_file: "eval_rephrased.jsonl"
  eval_mirror_file: "eval_mirror.jsonl"
  split_report_file: "split_report.json"   # Cluster distribution, stratification stats
```

---

## STAGE 1: Semantic Text Chunking

### 1.1 Purpose

The chunking stage transforms raw markdown documents into semantically coherent text segments. Each segment becomes the "context" fed to the Q&A generation model. The quality of chunking directly determines the quality of generated questions.

The key principle is **semantic grouping over hard token cutoffs**. Text is never sliced at an arbitrary token count. Instead, the pipeline respects sentence boundaries, section boundaries, and special structures like tables.

### 1.2 Document-Level Summary Generation

**When this runs:** Before any chunking begins, as the very first substep of Stage 1.

**What it does:** For each markdown file, the pipeline sends the file content (or truncated to `chunking.yaml → summary.max_input_tokens`) to an LLM with a summarization prompt. The prompt asks: "Summarize this company document in 3-5 sentences. What is it about? What is its purpose? What domain does it cover?"

**How it integrates:** The resulting summary is stored in `summaries.json` as a dictionary keyed by filename. When Stage 2 builds generation prompts, it looks up the summary for the chunk's source file and injects it as the first context layer of the prompt. This gives the generation model awareness of the broader document even when it only sees one chunk.

**Implementation detail:** This step makes one LLM call per markdown file (not per chunk). For a typical company with 20-50 documentation files, this is 20-50 API calls — negligible cost. The summary LLM can be a cheaper model than the generation model since summarization is a simpler task.

**Fault tolerance:** If a summary call fails after retries, the pipeline proceeds without a summary for that file. The generation prompt is designed to work without the summary layer — it's a quality enhancement, not a hard requirement.

### 1.3 Sentence Boundary Detection

The full text of each markdown file is split into sentence-level units before chunk assembly.

**Regex design considerations:**

Standard sentence endings: periods, question marks, exclamation marks, Arabic question mark (؟), followed by whitespace or end-of-string.

Abbreviation handling: A configurable list of abbreviations (in `chunking.yaml → sentence_detection.abbreviations`) is loaded at startup. Before applying the sentence-end regex, known abbreviation patterns are temporarily masked so their periods don't trigger false splits.

Decimal number protection: If `ignore_decimal_numbers` is true, patterns like `3.14`, `v2.0`, `$500.00` are masked before sentence splitting.

Arabic-specific: If `arabic_sentence_endings` is true, the regex also recognizes ؟ (Arabic question mark) and handles Arabic comma (،) as a non-sentence-ending punctuation. The Arabic full stop (۔) is treated as a sentence boundary.

Markdown list items: Lines beginning with `- `, `* `, or `N. ` (numbered list) are each treated as an individual sentence-level unit regardless of whether they end with a period.

Heading lines: Lines starting with `#` are treated as standalone units and additionally flagged as heading boundaries for the chunk assembler.

**Output:** An ordered list of sentence objects per file, each carrying: `text`, `source_file`, `is_heading` (bool), `heading_level` (int or null), `heading_hierarchy` (the path of headings above this sentence, e.g., "Getting Started > Registration > Email").

### 1.4 Table Isolation

Markdown tables are detected via regex: a contiguous block of lines where each line starts and ends with `|`, with a header separator row containing `---` or `:---:` patterns.

**Behavior:**

- A detected table is extracted as a single atomic unit. It is never split across chunks, even if it exceeds `max_chunk_tokens`. If oversized, it is emitted as a single chunk with a metadata flag `oversized: true`.
- The table chunk includes the nearest preceding heading as a prefix (configurable via `table_detection.include_preceding_heading`).
- The minimum row count (from `table_detection.min_rows`) prevents false positives — a stray line with pipes is not a table.
- After extraction, the table is removed from the sentence stream so it doesn't get double-counted during chunk assembly.

### 1.5 Adaptive Chunk Assembly

With sentences and tables identified, chunks are assembled using a token-window approach.

**Algorithm (pseudocode):**

```
current_chunk = []
current_tokens = 0

for each sentence in sentence_stream:
    if sentence is TABLE:
        finalize current_chunk (if meets min_chunk_tokens)
        emit table as standalone chunk
        continue

    if sentence is HEADING and force_break_on_headings:
        if heading_level in heading_levels_to_break_on:
            finalize current_chunk (if meets min_chunk_tokens)
            start new chunk with this heading

    if current_tokens + sentence_tokens > max_chunk_tokens:
        finalize current_chunk
        start new chunk with overlap_sentences from end of previous chunk
        add current sentence

    else:
        add sentence to current_chunk
        current_tokens += sentence_tokens

at end of file:
    if current_chunk < min_chunk_tokens and previous chunk exists:
        merge current_chunk backward into previous chunk
    else:
        finalize current_chunk
```

**Token counting:** Uses the tokenizer configured in `chunking.yaml → tokenizer`. The plan strongly recommends `tiktoken` for OpenAI models or the HuggingFace tokenizer for other models. The `character_ratio` method is a fallback only — it is explicitly unreliable for Arabic text because Arabic tokenization patterns vary significantly from the character-to-token ratios typical of Latin scripts. The YAML config carries a warning comment about this.

**Chunk metadata:** Each emitted chunk carries: `chunk_id` (format: `{filename}__chunk_{N}`), `chunk_text`, `source_file`, `section_hierarchy`, `token_count`, `contains_table`, `chunk_index` (sequential position in file), `is_oversized` (bool).

### 1.6 Checkpoint and Output

After all files are processed, the complete chunk list is written to `intermediate_dir/chunks.jsonl` (one JSON object per line). The summaries dictionary is written to `intermediate_dir/summaries.json`.

These files serve as the checkpoint for Stage 1. If Stage 2 crashes, it reads from these files instead of re-running chunking.

---

## STAGE 2: Question & Answer Generation

### 2.1 Purpose

For each chunk, an LLM generates question-answer pairs strictly grounded in the chunk content, written in Saudi Arabic dialect. The design ensures no hallucination, high diversity, built-in evaluation criteria, and full content coverage through iterative generation.

### 2.2 The Unified LLM Client

Since the pipeline no longer relies on distilabel's LLM abstractions, a custom `llm_client.py` module provides a unified interface for calling any supported LLM provider.

**What it handles:**

- Provider abstraction: A single `call_llm(messages, config)` function that routes to OpenAI, Anthropic, or Azure based on the `provider` field in config.
- Retry logic: Exponential backoff with jitter on rate limit errors (429) and server errors (500/502/503). Configurable `max_retries` and `retry_delay_seconds` from YAML.
- Rate limiting: A token-bucket rate limiter that respects `requests_per_minute` from YAML. This prevents hitting API rate limits in the first place rather than relying solely on retries.
- Timeout handling: Per `timeout_seconds` from YAML. If a call times out, it counts as a retry.
- JSON mode: If the provider supports native JSON response format (OpenAI does via `response_format: { type: "json_object" }`), the client enables it. For providers that don't, the client relies on prompt instructions alone.
- Logging: Every call is logged with: timestamp, model, token counts (prompt + completion), latency, success/failure. This enables cost tracking and debugging.

### 2.3 Prompt Architecture

Every generation call receives a prompt composed of three context layers, assembled at runtime from the prompt template files and chunk metadata.

**Layer 1 — Document Summary** (from `summaries.json`):
Provides the LLM with broad context about the source document. Injected into the user prompt template at the `{document_summary}` placeholder.

**Layer 2 — The Specific Chunk** (from `chunks.jsonl`):
The raw chunk text. Injected at `{chunk_text}`. The prompt template must include an explicit grounding instruction around this: "All answers must be derived exclusively from the information present in this chunk. Do not use any external knowledge. Do not fabricate information."

**Layer 3 — Domain Context String** (from `pipeline.yaml → domain_context`):
A short configurable description of the company/domain. Injected at `{domain_context}`.

**Additional metadata injected:**
- `{section_hierarchy}` — the heading path for the chunk
- `{source_file}` — the filename (useful for the model to understand document type)

**System prompt composition:**
The system prompt is built by concatenating:
1. The base system prompt from `qa_gen_system.txt`
2. The dialect specification from `dialect_spec.txt`
3. Few-shot examples (embedded at the end of the system prompt)

This is done at pipeline startup, not per-call, since the system prompt is the same for all chunks.

### 2.4 Saudi Dialect Specification

The dialect specification lives in a dedicated file (`prompts/dialect_spec.txt`) rather than being hardcoded. This allows tuning dialect requirements without modifying code.

**Content of `dialect_spec.txt`:**

Lexical markers organized by function (question words, verbs, modifiers, connectors, negation). Each entry shows the Saudi form alongside the MSA equivalent it replaces.

Grammatical patterns: Saudi negation ("ما" + verb), possession ("عندي"/"عندك"), progressive ("قاعد" + verb), future ("بـ" prefix).

Register instructions: conversational tone, as if a real Saudi customer is asking and a real Saudi colleague is answering. Explicit instruction to avoid Egyptian ("ازاي"), Iraqi ("شكو"), or Levantine markers.

Two to three complete few-shot examples showing input context → expected Q&A output with all fields (question, answer, evaluation_criteria, category, difficulty).

### 2.5 Per-Chunk Output Schema

The generation prompt requests structured JSON output with five fields per Q&A pair:

**`question`** (string): The question in Saudi dialect.

**`answer`** (string): The answer in Saudi dialect, grounded entirely in the chunk.

**`evaluation_criteria`** (string): A grading rubric describing what a correct answer must contain. This is the key innovation — instead of matching against a verbatim ground truth, automated evaluation checks whether responses satisfy the rubric's requirements. The criteria should specify: key points that must be covered, what constitutes a partial answer, and what would be incorrect. Minimum 30 characters to ensure substantive rubrics.

**`category`** (string): One of six values — `factual`, `procedural`, `reasoning`, `troubleshooting`, `conceptual`, `comparative`.

**`difficulty`** (string): One of `easy`, `medium`, `hard`, assessed by the model based on inference complexity.

### 2.6 Iterative Coverage Loop

This is the most critical generation design choice. Instead of requesting a fixed number of questions, the system uses a per-chunk iterative loop to achieve full content coverage.

**Detailed loop mechanics:**

**The loop runs inside a single function call per chunk.** It is a blocking sequential process: call LLM → parse response → decide whether to continue → call LLM again. It is NOT structured as separate pipeline stages or a DAG. This design was chosen because the loop state (accumulated questions) is local to a single chunk and doesn't need to be shared across chunks.

**Iteration 1 (Initial Pass):**
Assemble the prompt from `qa_gen_user.txt` with all three context layers. Send to LLM. Parse the JSON response to extract Q&A pairs. If parsing fails, apply repair strategies from `generation.yaml → iteration.parse_failure_strategy`:

- `retry_once`: Re-send the same prompt, up to `parse_failure_max_retries` times.
- `skip_iteration`: Accept what was parsed (possibly nothing) and move to next iteration.
- `stop_loop`: Terminate the loop for this chunk and move to the next chunk.

Store successfully parsed Q&A pairs in an accumulator list.

**Iteration 2+ (Follow-up Passes):**
Assemble the follow-up prompt from `qa_gen_followup.txt`. This template includes:
- The same chunk text
- The accumulated questions formatted according to `iteration.feedback_format`:
  - `json_list`: The questions as a JSON array of strings
  - `numbered_list`: Questions numbered 1, 2, 3... as plain text
  - `bullet_list`: Questions as dash-prefixed lines
- The instruction: "Review the chunk again. Are there important details NOT covered by the existing questions? If yes, generate additional Q&A pairs. If the chunk is fully covered, return an empty JSON array `[]`."

Parse the response. Before accepting new Q&A pairs into the accumulator, each new question passes through a **within-loop deduplication gate**: it is compared against all existing accumulated questions using `rapidfuzz.fuzz.ratio`. If the similarity score exceeds `within_loop_dedup.similarity_threshold` (default 0.80) against any existing question, the new question is silently discarded as a near-duplicate. This is a fast character-level comparison — no embeddings or LLM calls — so it adds negligible latency.

After filtering, count how many genuinely new questions survived the dedup gate. If zero survived, the iteration is marked as **stagnant** — the model returned content but all of it was rephrased duplicates of existing questions. A stagnation counter is incremented. If the stagnation counter reaches `stagnation_limit` (default 2) consecutive stagnant iterations, the loop terminates. If at least one genuinely new question survived, the stagnation counter resets to zero and the new questions are added to the accumulator.

If the model explicitly returns an empty array or zero Q&A pairs (before dedup), the loop terminates immediately (coverage achieved).

**Termination conditions (any one triggers stop):**
1. Model returns zero new questions (explicit empty response).
2. Stagnation limit reached — `stagnation_limit` consecutive iterations where the model returned questions but all were filtered as near-duplicates by the within-loop dedup gate.
3. `max_iterations` reached (from YAML).
4. Parse failure with `stop_loop` strategy.

**Concurrency across chunks:**
While the loop for a single chunk is sequential, multiple chunks can be processed concurrently (controlled by `batching.concurrency` in YAML). Each concurrent worker runs its own independent loop for its assigned chunk. An async semaphore limits the number of simultaneous LLM calls to stay within the rate limit.

### 2.7 Multi-Model Generation

If `multi_model.enabled` is true in YAML, the entire generation stage runs independently for each configured model. Each model processes all chunks through its own iterative coverage loop. The outputs are saved to separate files (`raw_qa_gpt4o-mini.jsonl`, `raw_qa_claude-sonnet.jsonl`, etc.).

**Cross-model interaction:** There is none during generation. Each model runs blind to the other models' outputs. Cross-model diversity benefits are realized in Stage 3 during deduplication — semantically identical questions from different models are detected and only the best version is kept, while genuinely unique questions from each model are retained.

This design is intentional. Feeding Model A's questions into Model B's follow-up prompts would create a dependency chain that complicates fault tolerance and makes it impossible to re-run one model independently. The post-hoc merge is simpler and equally effective.

### 2.8 Checkpointing and Fault Tolerance

After every `checkpoint_every_n_chunks` chunks are processed, the accumulated results are flushed to disk (appended to the output JSONL file). A checkpoint metadata file records which chunk IDs have been completed.

If the pipeline crashes and is restarted, Stage 2 reads the checkpoint file and skips already-completed chunks. This means a crash at chunk 150 out of 500 only requires re-processing chunks 151-500, not starting over.

Each chunk's output includes a `generation_metadata` field storing: model used, iteration count, total questions generated, timestamp, and any parse failures encountered. This metadata enables post-hoc analysis of generation quality per model and per iteration.

---

## STAGE 3: Data Cleaning & Coverage Analysis

### 3.1 Structural Validation

Every raw output is parsed and validated against the schema defined in `cleaning.yaml → structural`.

**Parse pipeline:**

1. Take the raw LLM output string for a chunk.
2. If `attempt_json_repair` is true, apply repairs in sequence: strip markdown code fences → fix trailing commas → regex-extract JSON array.
3. Attempt `json.loads()`. If it fails after repair, log the raw output to `discarded_qa.jsonl` with reason `parse_failed` and skip.
4. For each Q&A object in the parsed array, check:
   - All `required_fields` are present and non-empty strings.
   - `question` length is within `[question_min_chars, question_max_chars]`.
   - `answer` length is within `[answer_min_chars, answer_max_chars]`.
   - `evaluation_criteria` length ≥ `evaluation_criteria_min_chars`.
   - `category` is in `allowed_categories`.
   - `difficulty` is in `allowed_difficulties`.
5. Items failing any check are logged to `discarded_qa.jsonl` with the specific failure reason and the values that triggered rejection.

### 3.1.1 LLM Substep Processing Order and Concurrency

Stage 3 contains three LLM-calling substeps (grounding check, dialect check, tagging). These run **sequentially as sub-stages, not interleaved** — the pipeline completes all grounding checks before starting any dialect checks, and completes all dialect checks before starting any tagging. This ordering is defined in `cleaning.yaml → substep_order` and matters because earlier substeps discard items, reducing the number of LLM calls in later substeps:

1. **Grounding check first** — highest discard rate. Removes hallucinated answers before spending money on dialect checks.
2. **Dialect check second** — moderate discard rate. Removes non-Saudi content before spending money on tagging.
3. **Tagging last** — zero discard rate. Only adds metadata. Runs exclusively on items that survived both prior checks.

Within each substep, items are processed in **concurrent batches** controlled by `cleaning.yaml → stage_concurrency`. The pipeline collects `batch_size` items (default 20), fires all LLM calls in that batch concurrently (up to `max_concurrent_llm_calls` via an async semaphore), waits for the entire batch to complete, then moves to the next batch. This gives predictable progress reporting ("grounding: batch 14/75 complete") and respects the per-stage `requests_per_minute` rate limit.

### 3.2 Saudi Dialect Validation

**Marker-based scoring:** For each Q&A pair, count distinct Saudi dialect markers (from the YAML list) in the question and answer separately.

- Question must contain ≥ `question_min_markers` markers.
- Answer must contain ≥ `answer_min_markers` markers.
- A pair with zero combined markers is discarded.

**MSA contamination detection:** Count strong MSA indicators (from the YAML list) in the answer. If the count exceeds `msa_max_indicators`, the pair is flagged. It is not auto-discarded because some technical content legitimately uses formal vocabulary — but the flag is recorded in the cleaned output for optional human review.

**LLM-based dialect check:** If enabled, a random sample of `sample_percentage` percent of pairs is sent to an LLM with the `dialect_check.txt` prompt. The prompt asks the LLM to rate dialect authenticity from 1-5 and identify non-Saudi elements. Pairs scoring below `min_score` are discarded. This catches subtle dialect issues (wrong verb conjugation patterns, unnatural register mixing) that keyword matching misses.

### 3.3 Answer Grounding Validation

This step is critically **language-aware**. The config tracks `source_language` and `answer_language` explicitly.

**When source and answer are the same language** (e.g., both MSA or both Saudi dialect): Token overlap is a viable check. Enabled via `token_overlap.enabled`, comparing answer tokens against chunk tokens with a minimum overlap ratio.

**When source and answer are different languages** (e.g., source is English, answer is Saudi dialect): Token overlap is meaningless and is **disabled by default** in the YAML. Instead, the LLM-based grounding check handles this case.

**LLM-based grounding check:** Each Q&A pair is sent to a judge LLM with the source chunk, question, and answer. The prompt (from `grounding_check.txt`) asks: "Given ONLY the context provided, is this answer factually supported? Classify as: `fully_supported`, `partially_supported`, or `not_supported`. Explain your reasoning briefly."

The action for each classification is configurable in YAML:
- `fully_supported` → keep
- `partially_supported` → keep (but flagged for optional human review)
- `not_supported` → discard

This check runs on 100% of pairs (not sampled) because grounding is the most important quality dimension — a hallucinated answer in Saudi dialect is worse than no answer.

### 3.4 Deduplication

Three levels, applied sequentially:

**Level 1 — Exact match:** After unicode normalization and whitespace stripping, remove pairs with identical question strings. Zero computational cost.

**Level 2 — Fuzzy string similarity:** Using `rapidfuzz` (recommended in YAML, falls back to `difflib`), compute pairwise similarity between all question strings. Pairs exceeding the `similarity_threshold` (0.85) are flagged. The pair with the shorter answer is discarded (`keep_strategy: longer_answer`).

The default `blocking_strategy` is `"none"` (brute-force all-pairs comparison). Traditional n-gram blocking (e.g., "only compare questions sharing a 4-gram") is unreliable for Arabic because Arabic morphology is highly inflected — the same root word appears in many surface forms due to prefixes, suffixes, and clitics. For example, "يسجلون" (they register), "التسجيل" (registration), and "سجّل" (register/imperative) all share the root س-ج-ل but share zero 4-grams. An n-gram blocking strategy would miss near-duplicate questions that differ only in verb conjugation or attached pronouns.

Brute-force all-pairs `rapidfuzz` comparison completes in under a minute for datasets up to ~20,000 items on modern hardware. If the dataset exceeds `brute_force_warn_threshold`, the pipeline logs a warning suggesting the user switch to `"arabic_stem"` blocking, which uses an Arabic stemmer (`ISRIStemmer` from NLTK or `tashaphyne`) to extract root stems and blocks by shared stems. This is more accurate for Arabic than character n-grams but adds an NLP dependency.

**Level 3 — Semantic deduplication:** All questions are embedded using the model from `cleaning.yaml → coverage.embedding_model`. A FAISS index is built, and for each question, the K nearest neighbors are retrieved. Pairs exceeding the `cosine_threshold` are flagged.

**Threshold calibration:** The `cosine_threshold` of 0.92 is a starting point that **must be calibrated on your data**. If `calibration_mode` is true, the pipeline:
1. Samples `calibration_sample_size` pairs.
2. Computes all pairwise cosine similarities within the sample.
3. Outputs a histogram of similarity scores.
4. You manually inspect pairs at various similarity levels to determine where "duplicate" ends and "legitimately different" begins.
5. Update the threshold in YAML based on your findings.

This calibration step is essential because multilingual embedding models compress Arabic text differently than English, and Saudi dialect may cluster even tighter than MSA.

### 3.5 Coverage Visualization

**t-SNE Embedding Plot:**

All question embeddings (already computed for semantic dedup) are projected to 2D using t-SNE with parameters from YAML. Multiple plots are generated, each color-coded by a different attribute (`source_file`, `category`, `generation_model`).

What to look for:
- **Gaps**: empty regions between clusters → missing topic coverage. Identify which document sections map to those regions and re-run generation on those specific chunks.
- **Dense blobs**: excessively tight clusters → redundancy. Consider thinning.
- **Outliers**: isolated points → possibly hallucinated or off-topic questions that slipped past cleaning. Inspect manually.

**Tag-Based Topic Histogram:**

Each question is sent to an LLM with the `tagging.txt` prompt, which asks for 1-3 short hyphenated topic tags.

**Tag normalization** (addressing issue #9 from the review): Raw LLM-assigned tags are inconsistent ("account-setup" vs. "account-registration" vs. "new-account" all mean the same thing). The pipeline applies post-hoc normalization:

If `method: embedding_clustering`: Embed all unique tags, compute pairwise similarities, and merge tags above the `merge_similarity_threshold`. The most frequent tag in each merged group becomes the canonical form, and all other forms are replaced.

If `method: controlled_vocabulary`: A YAML file provides a fixed list of allowed tags. Each LLM-assigned tag is embedded and mapped to the nearest allowed tag by cosine similarity.

The normalized tag frequencies are plotted as a histogram. Compare this distribution against a rough estimate of your source documents' topic distribution to identify under-covered and over-covered areas.

---

## STAGE 4: Evaluation Dataset Creation

### 4.1 Purpose

Split the cleaned dataset into training and evaluation sets using a **cluster-balanced** strategy (not random splitting), then create a rephrased eval set and an eval mirror for robust training diagnostics.

### 4.2 Embedding and Clustering

The question embeddings from Stage 3 are reused (no re-computation needed).

**Primary method — HDBSCAN:**
HDBSCAN is used instead of K-Means because:
- It does not require pre-specifying the number of clusters.
- It handles clusters of varying size and density, which is realistic for document-derived Q&A (some topics have many questions, others have few).
- It naturally identifies noise points (items that don't belong to any cluster), which are then assigned to the nearest cluster via `assign_noise_to_nearest`.
- It avoids the ambiguity of the elbow method, which often produces no clear elbow for high-dimensional embedding data.

HDBSCAN parameters (`min_cluster_size`, `min_samples`) are configurable in YAML. After clustering, each question has a `cluster_id` label.

**Fallback — K-Means:**
If HDBSCAN produces unsatisfactory results (e.g., one giant cluster and many noise points), K-Means can be enabled in YAML with `k_selection_method: silhouette` (the silhouette score is more reliable than the elbow method for embeddings).

### 4.3 Balanced Proportional Sampling

**Target eval size:** Determined by `splitting.eval_fraction` (default 12%).

**Sampling algorithm:**

1. For each cluster, calculate its proportion of the total dataset.
2. Allocate that same proportion of the eval budget to each cluster. For example: if cluster 0 has 300/2000 items (15%) and the eval budget is 240 items, cluster 0 contributes 36 items.
3. Within each cluster, apply stratified sampling that balances the `stratify_by` fields (from YAML): difficulty distribution, category distribution, source file distribution. This uses iterative proportional fitting — not just random sampling — to match the training set's distribution as closely as possible.
4. If a cluster has fewer items than its allocation, take all items from that cluster and redistribute the shortfall proportionally across remaining clusters.
5. Enforce `min_items_per_cluster` — even tiny clusters contribute at least this many items to eval so no topic is completely absent from evaluation.

The selected items form the **eval set**. The remaining items form the **training set** — but with one important addition described in section 4.5.

### 4.4 Rephrased Eval Split

Every eval item is sent to an LLM with the `rephrase.txt` prompt. The prompt performs two tasks in a single call:

**Part 1 — Rephrase Q&A:**

1. Reword the question using different phrasing while asking the same thing.
2. Rewrite the answer with different sentence structure and word choices while conveying identical information.
3. Maintain Saudi dialect throughout (the same dialect specification from Stage 2 is included in the rephrasing prompt).
4. Keep `category` and `difficulty` fields unchanged.

**Part 2 — Generalize evaluation criteria:**

The original `evaluation_criteria` rubric often references specific phrases from the original answer (e.g., "A correct answer must mention: تسجيل الدخول, then الضغط على إعدادات, then تحديث البيانات"). After rephrasing, the answer uses different words for the same concepts, so a phrase-specific rubric becomes misaligned — it would incorrectly penalize a correct rephrased answer for not using the exact original wording.

To fix this, the rephrasing prompt also rewrites the evaluation criteria to describe required content **conceptually** rather than by referencing exact phrases. The instruction to the LLM: "Rewrite this grading rubric so it describes what concepts and information a correct answer must contain, without using specific phrases or exact wordings from any particular answer. The rubric should be usable to grade any correctly-worded answer, regardless of phrasing."

Example transformation:
- **Original rubric**: "A correct answer must mention: تسجيل الدخول, then الضغط على إعدادات, then تحديث البيانات"
- **Generalized rubric**: "A correct answer must describe three sequential steps: logging into the account, navigating to the settings area, and updating the personal information."

The generalized rubric is stored in the rephrased eval set. The original phrase-specific rubric stays with the mirror set. This generalization also has a production benefit: when evaluating your fine-tuned model at inference time, you will be grading novel answers that use their own phrasing, so a concept-level rubric is inherently more appropriate than a phrase-matching one.

If `generalize_evaluation_criteria` is false in YAML, the rubric is preserved verbatim (not recommended, but available for backwards compatibility).

The rephrased output replaces the question, answer, and evaluation_criteria fields. An `original_question` field preserves the pre-rephrased version for cross-referencing.

**Validation after rephrasing:** If `re_validate_dialect` is true in YAML, the rephrased output goes through the same Saudi dialect marker check as Stage 3. If the rephrased version fails dialect validation, the pipeline retries rephrasing once, then falls back to the original if the retry also fails (logging this as a rephrasing failure).

**Purpose:** The rephrased eval set tests whether the fine-tuned model has learned the underlying concepts vs. memorized exact phrasings. If validation loss on this set decreases during training, the model is genuinely generalizing.

### 4.5 The Eval Mirror

The eval mirror is a verbatim copy of the eval items with their **original** (un-rephrased) question and answer.

**Critical design decision — the mirror items ARE included in the training set.** This is intentional and is the mechanism that makes overfitting detection work.

**Why:**

The eval mirror items appear in training data. The rephrased eval items do NOT (they were removed and then reworded). During fine-tuning, the training process sees the mirror versions. Now compare two eval loss curves:

- **Mirror loss**: The model has seen these exact question-answer pairs during training. Its loss on these should decrease as it memorizes them. This is expected.
- **Rephrased loss**: The model has NEVER seen these exact phrasings. If rephrased loss also decreases, the model is learning the concepts behind the questions. If rephrased loss plateaus or increases while mirror loss keeps dropping, the model is memorizing surface forms without understanding.

The **gap between mirror loss and rephrased loss** is your overfitting metric. A growing gap means "stop training."

This design requires the mirror items to be in the training set. If they were removed (as was incorrectly stated in the previous plan version), the mirror would just be another held-out set and the comparison would be meaningless.

### 4.6 Final Output Format

**`train.jsonl`** — Each line:
```json
{
  "question": "...",
  "answer": "...",
  "context": "...",
  "evaluation_criteria": "...",
  "category": "...",
  "difficulty": "...",
  "source_file": "...",
  "section_hierarchy": "...",
  "chunk_id": "...",
  "cluster_id": 0,
  "generation_model": "gpt-4o-mini",
  "dialect_marker_count": 4,
  "is_eval_mirror": false
}
```

Items that are also eval mirror entries have `"is_eval_mirror": true`. This flag allows you to optionally filter them during analysis but they remain in the training file for the loss tracking to work.

**`eval_rephrased.jsonl`** — Same schema but with rephrased question, answer, and generalized evaluation criteria. Plus:
```json
{
  "original_question": "...",
  "original_answer": "...",
  "original_evaluation_criteria": "..."
}
```

**`eval_mirror.jsonl`** — Same items as eval_rephrased but with the original un-rephrased question and answer. Includes:
```json
{
  "rephrased_question": "...",
  "rephrased_answer": "..."
}
```

**`split_report.json`** — Statistics:
```json
{
  "total_items": 2000,
  "train_items": 1760,
  "eval_items": 240,
  "eval_mirror_in_train": 240,
  "clusters_found": 14,
  "cluster_distribution": {"0": {"train": 265, "eval": 35}, ...},
  "difficulty_distribution": {"train": {"easy": 600, "medium": 800, "hard": 360}, ...},
  "category_distribution": {"train": {...}, "eval": {...}},
  "rephrasing_failures": 3
}
```

---

## Fault Tolerance Strategy

The previous plan had no error handling strategy. This section describes how each stage handles failures.

### Checkpointing

Each stage writes its output to an intermediate file in `intermediate_dir`. If the pipeline is stopped and restarted, the master `run_pipeline.py` checks:

1. Does `chunks.jsonl` exist and is it complete (line count matches expected file count)? If yes, skip Stage 1.
2. Does `raw_qa.jsonl` exist? Read the checkpoint metadata file to determine which chunks have been processed. Resume from the next unprocessed chunk.
3. Does `cleaned_qa.jsonl` exist? If yes, skip Stage 3.
4. If any intermediate file is partially written, the pipeline re-runs that stage from the beginning (stages are designed to overwrite, not append, except for Stage 2 which uses explicit chunk-level checkpointing).

### Per-Item Failure Handling

When a single item (chunk, Q&A pair, or eval item) fails at any stage:

- **LLM call failure** (after all retries exhausted): The item is skipped and logged to a failures file. The pipeline continues with the next item. At the end of the stage, a summary reports how many items failed and why.
- **Parse failure**: Logged with the raw LLM output for manual inspection. The item is skipped.
- **Validation failure**: The item is written to `discarded_qa.jsonl` with a structured reason field. This file is a valuable diagnostic tool — if many items are being discarded for the same reason, it suggests the generation prompt needs adjustment.

### Rate Limit Management

The `llm_client.py` module implements a sliding-window rate limiter. Before each API call, it checks: would this call exceed `requests_per_minute` in the current window? If yes, it sleeps until the window rotates. This proactive approach avoids the retry-heavy pattern of "send → get 429 → wait → retry" which wastes time and can trigger escalating rate limits from some providers.

---

## Evaluation Criteria Summary Table

| Criterion | Check Method | Threshold / Rule | Applied At |
|---|---|---|---|
| JSON structure | Regex repair + parser | Must parse to valid list of dicts | Stage 3.1 |
| Required fields | Field presence check | All 5 fields present, non-empty | Stage 3.1 |
| Question length | Character count | 15–500 characters | Stage 3.1 |
| Answer length | Character count | 30–2,000 characters | Stage 3.1 |
| Eval criteria quality | Character count | ≥ 30 characters | Stage 3.1 |
| Category validity | Enum check | One of 6 allowed values | Stage 3.1 |
| Difficulty validity | Enum check | One of 3 allowed values | Stage 3.1 |
| Saudi dialect (Q) | Marker keyword count | ≥ 1 marker in question | Stage 3.2 |
| Saudi dialect (A) | Marker keyword count | ≥ 2 markers in answer | Stage 3.2 |
| MSA contamination | MSA keyword count | ≤ 3 strong MSA indicators | Stage 3.2 |
| LLM dialect rating | LLM judge (15% sample) | Score ≥ 4/5 | Stage 3.2 |
| Answer grounding | LLM judge (100%) | "fully_supported" or "partially_supported" | Stage 3.3 |
| Exact duplicates | String matching | Remove identical questions | Stage 3.4 |
| Near-duplicates | Fuzzy string similarity | < 85% similarity (rapidfuzz) | Stage 3.4 |
| Semantic duplicates | Embedding cosine similarity | < 0.92 cosine (CALIBRATE THIS) | Stage 3.4 |
| Topic coverage | t-SNE visualization | No large gaps in embedding space | Stage 3.5 |
| Topic balance | Tag histogram (normalized) | Distribution matches source docs | Stage 3.5 |
| Eval split balance | HDBSCAN cluster-proportional | Matches training distribution per cluster | Stage 4.3 |
| Dialect after rephrase | Marker check on rephrased text | Same thresholds as Stage 3.2 | Stage 4.4 |
| Generalization test | Rephrased eval loss tracking | Must decrease during training | Stage 4.5 |
| Overfitting detection | Mirror vs Rephrased loss gap | Gap should remain small | Stage 4.5 |

---

## Technology Stack

| Component | Tool | Install |
|---|---|---|
| LLM calls (OpenAI) | `openai` Python SDK | `pip install openai` |
| LLM calls (Anthropic) | `anthropic` Python SDK | `pip install anthropic` |
| Sentence embeddings | `sentence-transformers` | `pip install sentence-transformers` |
| Token counting | `tiktoken` | `pip install tiktoken` |
| YAML config parsing | `PyYAML` | `pip install pyyaml` |
| Fuzzy deduplication | `rapidfuzz` | `pip install rapidfuzz` |
| Arabic stemming (optional, for fuzzy dedup blocking) | `nltk` ISRIStemmer or `tashaphyne` | `pip install nltk` or `pip install tashaphyne` |
| FAISS (semantic dedup) | `faiss-cpu` | `pip install faiss-cpu` |
| Clustering (HDBSCAN) | `hdbscan` | `pip install hdbscan` |
| Dimensionality reduction | `scikit-learn` (t-SNE) | `pip install scikit-learn` |
| Visualization | `matplotlib` | `pip install matplotlib` |
| Async concurrency | `asyncio` + `aiohttp` (stdlib + optional) | built-in |
| Data I/O | `json`, `pathlib` (stdlib) | built-in |
