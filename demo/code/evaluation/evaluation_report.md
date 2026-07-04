# Operational Analysis

This report covers the operational characteristics of the Multi-Modal Evidence Review
pipeline. Counts of claims, images, and model calls are **measured** from the actual run;
token usage and cost are **estimates** computed analytically with the assumptions stated
below (per the "approximate ... with pricing assumptions" requirement).

## Pipeline shape (per claim)

Each claim runs through the LangGraph workflow with **three LLM calls** on the normal path:

| Node | Model | Modality | Purpose |
|---|---|---|---|
| `extract_claim` | Claude Sonnet 4.5 | text | parse the claim from the transcript |
| `analyze_images` | Claude Opus 4.8 | vision | per-image structured findings |
| `reconcile` | Claude Sonnet 4.5 | text | supported / contradicted / NEI verdict |

The other nodes (`load_context`, `evidence_check`, `risk_merge`, `finalize`, `force_nei`)
are **deterministic** and make no model calls. Truly-unusable image sets short-circuit via
`force_nei`, skipping the `reconcile` call (2 calls instead of 3).

**Model routing is a deliberate cost optimization:** the expensive vision-capable model
(Opus) is used only for the one node where accuracy required it; the cheaper Sonnet handles
both text/reasoning nodes. An experiment using Opus on all nodes was *less* accurate
(61% vs 69% mean field accuracy on the sample set) and ~3x more expensive, so it was rejected.

## Model calls

| Set | Claims | Sonnet calls (extract + reconcile) | Opus calls (vision) | Total calls |
|---|---|---|---|---|
| Sample | 20 | 40 | 20 | 60 |
| Test | 44 | 88 | 44 | 132 |

## Images processed

| Set | Images | Avg / claim | Min | Max |
|---|---|---|---|---|
| Sample | ~37 | ~1.85 | 1 | 3 |
| Test | 82 | 1.86 | 1 | 3 |

All 82 test images were successfully analyzed (no load/format failures after image
normalization — see strategy below).

## Token usage (estimated)

Assumptions per call type (system prompt + content in, structured JSON out):

- `extract_claim`: ~500 in / ~150 out
- `analyze_images`: ~3,200 in (≈400 prompt + ~1,500 tokens/image × 1.86 images) / ~250 out
- `reconcile`: ~900 in / ~200 out

Image token cost assumes Claude's ~(pixels/750) heuristic with the long edge capped near
1,568 px, i.e. ~1,300–1,600 tokens per image.

| Model | Input tokens | Output tokens |
|---|---|---|
| Sonnet (88 calls) | ~62,000 | ~15,000 |
| Opus (44 calls) | ~141,000 | ~11,000 |
| **Total (test set)** | **~203,000** | **~26,000** |

## Cost (estimated)

**Pricing assumptions (placeholders — verify against current Bedrock us-east-1 rates):**

- Sonnet tier: ~$3 / 1M input tokens, ~$15 / 1M output tokens
- Opus tier: ~$15 / 1M input tokens, ~$75 / 1M output tokens

| Model | Input cost | Output cost | Subtotal |
|---|---|---|---|
| Sonnet | ~$0.19 | ~$0.23 | ~$0.42 |
| Opus | ~$2.12 | ~$0.83 | ~$2.94 |
| **Test set total (44 claims)** | | | **~$3.40** |

Approximate **~$0.08 per claim**. Vision (Opus) dominates cost at ~87% of the total, which is
why it is isolated to a single node. The sample set (20 claims) is proportionally ~$1.55.

## Latency / runtime

- ~30–60 seconds per claim (wall-clock), dominated by the Opus vision call.
- Test set (44 claims, sequential): ~22–44 minutes.
- Sample set (20 claims): ~10–20 minutes.
- Processing is sequential (one claim at a time); claims are independent and could be
  parallelized with a bounded concurrency limit to reduce total runtime.

## TPM / RPM considerations and reliability strategy

**Throughput.** At ~45s/claim sequential, request rate stays around 1–3 calls/minute and a few
thousand tokens/minute — well under typical Bedrock TPM/RPM quotas, so no batching or rate
limiting was required at this scale. Note Opus carries lower per-model quotas than Sonnet;
isolating Opus to one call per claim keeps peak Opus RPM low.

**Implemented strategies:**

- **Retry / throttling:** the Bedrock client uses `read_timeout=120s` and
  `retries={"max_attempts": 4, "mode": "adaptive"}`. Adaptive mode backs off and retries
  transient timeouts and throttling automatically (an early run hit a 60s timeout; raising the
  timeout and enabling adaptive retries resolved it).
- **Image handling for the 5 MB Bedrock limit:** images are sniffed by magic bytes (extensions
  in the dataset are unreliable — `.jpg` files were often WebP/PNG), and any oversized or
  unrecognized image is decoded with Pillow, downscaled to a 2,048 px long edge, and re-encoded
  as JPEG under ~4.5 MB. This both prevents API rejections and reduces upload size/cost.
- **Graceful degradation:** each claim is wrapped so a single failure emits a safe
  not_enough_information row (flagged for manual review) rather than aborting the batch; the run
  always produces a complete `output.csv`.

**Not implemented (future optimization):**

- **Caching:** vision findings could be cached by image content hash to avoid recomputing on
  re-runs or duplicate images; deterministic nodes are already free.
- **Parallelism:** a concurrency-bounded async fan-out over claims would cut wall-clock runtime
  substantially while respecting RPM limits.

## Reproducibility note

Opus 4.8 deprecates the `temperature` parameter, so the vision node runs at the model default
rather than `temperature=0`. The text nodes (Sonnet) remain deterministic, but the vision step
is not strictly reproducible — re-runs may vary by a few findings, and thus a point or two of
field accuracy.