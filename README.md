<div align="center">

# 🎯 REAR: Test-time Preference Realignment through Reward Decomposition

**Training-free preference alignment for LLMs at inference time — no reward model, no fine-tuning.**

[![arXiv](https://img.shields.io/badge/arXiv-2606.30339-b31b1b.svg)](https://arxiv.org/abs/2606.30339)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-1f6feb.svg)](https://arxiv.org/abs/2606.30339)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![Powered by vLLM](https://img.shields.io/badge/powered%20by-vLLM-ff6b6b.svg)](https://github.com/vllm-project/vllm)

[📄 Paper](https://arxiv.org/abs/2606.30339) · [🧠 Method](#-the-rear-score) · [🔧 Install](#-installation) · [🧪 Experiments](#-running-experiments)

</div>

---

Aligning LLMs with diverse user preferences usually means costly data curation and
extra training. **REAR** takes a different route: it treats preference following as
a *realignment* problem solved at **test time**. It decomposes the model's implicit
reward into a **question-related** term and a **preference-related** term and
rescales the two, producing a score computed entirely from the base model's own
token log-probabilities — so it needs **no external reward model and no fine-tuning**.
Because the score reduces to a linear combination of log-probabilities, it drops
straight into standard test-time scaling (TTS) algorithms such as best-of-$N$
sampling and DVTS.

## ✨ Highlights

- 🧩 **Reward decomposition** — split the implicit reward into question- and preference-related parts, then reweight with a single knob λ.
- 🪶 **Training-free & model-free** — no reward model, no fine-tuning; only the base model's log-probabilities.
- 🔌 **Plugs into TTS** — works with best-of-$N$ sampling and tree search (DVTS).
- 🌍 **General** — beyond preference tasks, it transfers to math, visual hallucination, and Qwen & Llama model families.
- ⚡ **Efficient** — reuses the generation model for scoring; faster than GenRM and external-RM baselines.

## 🧠 The REAR score

For a response $y$ given a question prompt and a preference text $x_p$, REAR uses two
per-token log-probability sequences from the served policy model:

| Term | Definition | Meaning |
|------|------------|---------|
| `r1` | `logP(y \| question-only prompt)` | question-related (base quality) |
| `r2` | `logP(y \| preference prompt) − logP(y \| question-only prompt)` | preference residual |

The trajectory score is the discounted sum of

```text
reward(token) = r1 + λ · r2
```

where **λ** (the *realignment strength*, `verifier.mixed_weight` in the configs)
controls how strongly the preference is enforced) weights tokens. Larger λ emphasises preference satisfaction; smaller λ
prioritises answering the question.

## 🔧 Installation

Python>=3.10 is required.

```bash
# Option A: uv (uses the pinned uv.lock)
uv sync

# Option B: pip
pip install -r requirements.txt
```

Install vLLM per the [official instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html)
if the wheel from `requirements.txt` does not match your CUDA/hardware.

<details>
<summary><b>vLLM <code>prompt_logprobs</code> guard (optional local patch)</b></summary>

REAR queries `prompt_logprobs` on chat/multimodal inputs. Some models can emit
out-of-range token ids that make the tokenizer's detokenizer crash. If you hit this,
add a small guard in the installed vLLM that replaces invalid ids with a placeholder:

- File: `<your-venv>/.../site-packages/vllm/tokenizers/detokenizer_utils.py`
- Behaviour: invalid/out-of-range ids → `"<out-of-bound>"`; decode errors → `"<unknown-error>"`

Re-apply after any vLLM upgrade.
</details>

## 🚀 Serving models

All methods talk to an OpenAI-compatible endpoint. Serve the **policy model** once and
point every run at it (default `http://localhost:30000/v1`):

```bash
bash scripts/launch-vllm-server.sh <MODEL> --host 0.0.0.0 --port 30000 --dp 8 --tp 1 --gpu-memory-utilization 0.85
# Qwen/Qwen2.5-7B-Instruct  |  Qwen/Qwen3-4B-Instruct-2507  |
# Qwen/Qwen3-VL-8B-Instruct |  meta-llama/Llama-3.1-8B-Instruct
```

The **BoN w/ GenRM** baseline reuses this same server (the base model judges its own
outputs) — no extra process. Only **BoN w/ External RM** needs a second server: the
Skywork reward model on its own port with the embedding task:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/launch-vllm-server.sh \
    Skywork/Skywork-Reward-Llama-3.1-8B --host 0.0.0.0 --port 8000 --dp 1 --tp 1 --is-embedding
```

### LLM-as-a-judge (GPT-4.1)

Open-ended benchmarks are scored by an LLM judge (default `gpt-4.1`) via the OpenAI
API. The judged-task scripts carry an `OPENAI_API_KEY` placeholder line; export your
real key (it is respected if already set) or fill it into the script:

```bash
export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=...   # optional: point at an OpenAI-compatible proxy
```

| Judge required | Judge **not** required |
|----------------|------------------------|
| PrefEval explicit / implicit, Multifaceted, MMHal, Ping-Pong | PrefEval choice, all math tasks (local scoring) |

## 🗂️ Data setup

Most benchmark data ships with the repo under `data/` (PrefEval, Multifaceted,
Ping-Pong, math sets). The only piece that is **not committed** is the MMHal-Bench
image set (~170 MB) — only the lightweight metadata is kept. Rebuild the images from
their source URLs before running the visual task:

```bash
python data/mmhal-bench/download_images.py           # writes data/mmhal-bench/images/
```

If some source URLs have expired, fetch the upstream archive and unzip its `images/`
into `data/mmhal-bench/images/`:
[huggingface.co/datasets/Shengcao1006/MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench).

## 🧪 Running experiments

`scripts/experiments/` holds one small, self-contained script per **(task, method)**.
Each is a single `run_eval.py` (or `run_ping_pong.py`) command with **λ = 20.0** and
**N = 16** by default — edit the model / λ / N inline. Paths resolve relative to the
repo, so a script runs from anywhere:

```bash
bash scripts/experiments/<task>/<method>.sh
```

**Methods** (the `<method>.sh` file names) map to the paper baselines:

| File           | Paper baseline      | Key configs                                            |
|----------------|---------------------|--------------------------------------------------------|
| `greedy.sh`    | Greedy              | `method=best_of_n verifier=greedy` (N=1, temp=0)       |
| `bon_rear.sh`  | BoN w/ REAR (ours)  | `method=rear verifier=preference`, `mixed_weight=20.0` |
| `dvts_rear.sh` | DVTS w/ REAR (ours) | `method=dvts verifier=preference`, `mixed_weight=20.0` |
| `bon_genrm.sh` | BoN w/ GenRM        | `verifier=generative_rm_server` (base model self-judges) |
| `bon_extrm.sh` | BoN w/ External RM  | `verifier=rm_server` (Skywork; needs the RM server)    |

**Tasks** (the `<task>/` directories), with base model and available methods:

| Task dir            | Paper result           | Base model            | Methods                                              |
|---------------------|------------------------|-----------------------|------------------------------------------------------|
| `prefeval_explicit` | Table 1 / 2            | Qwen2.5-7B-Instruct   | greedy, bon_rear, dvts_rear, bon_genrm, bon_extrm    |
| `prefeval_choice`   | Table 1 / 2            | Qwen2.5-7B-Instruct   | greedy, bon_rear, dvts_rear, bon_genrm, bon_extrm    |
| `prefeval_implicit` | Table 1                | Qwen2.5-7B-Instruct   | greedy, bon_rear, dvts_rear, bon_genrm, bon_extrm    |
| `multifaceted`      | Table 1                | Qwen2.5-7B-Instruct   | greedy, bon_rear, dvts_rear, bon_genrm, bon_extrm    |
| `ping_pong`         | Table 1                | Qwen2.5-7B-Instruct   | greedy, bon_rear, dvts_rear, bon_genrm, bon_extrm    |
| `math`              | Math scaling figure    | Qwen2.5-7B / Qwen3-4B | bon_rear, dvts_rear                                  |
| `mmhal`             | Table 3                | Qwen3-VL-8B-Instruct  | greedy, bon_rear, bon_genrm                          |
| `llama`             | Table 4 (cross-family) | Llama-3.1-8B-Instruct | greedy, bon_rear, dvts_rear, bon_genrm               |

For example, to run every method on one benchmark:

```bash
export OPENAI_API_KEY=sk-...          # judged tasks only
for m in greedy bon_rear dvts_rear bon_genrm; do
  bash scripts/experiments/prefeval_explicit/$m.sh
done
```

Notes:
- **Judge:** the LLM-judged tasks (PrefEval explicit/implicit, Multifaceted, MMHal,
  Ping-Pong) need `OPENAI_API_KEY`; PrefEval choice and math are scored locally.
- **Math:** one `math/bon_rear.sh` run reports both the REAR result (`weighted_sum@n`,
  mirrored to `avg_score`) and the Majority-voting result (`majority_vote@n`) over the
  same samples. Set `model_path=Qwen/Qwen3-4B-Instruct-2507` for the second model and
  raise `method.n_samples` to 64 for the full scaling curve.
- **MMHal:** run `python data/mmhal-bench/download_images.py` once before `mmhal/`.
- **External RM:** `bon_extrm.sh` needs the Skywork reward-model server (the launch
  command is in the script header; see also "Serving models").
- **Long-context (Table 2):** append `data.multi_turn_rounds=<0|5|10|20|50|100>` to a
  PrefEval command.

> **Amulet and Linear Alignment (LA)** are reported in the paper but are *not* part of
> this kit: their numbers come from the authors' external implementation at
> [github.com/zowiezhang/Amulet](https://github.com/zowiezhang/Amulet).

## 📤 Output structure

Results are written under `outputs/` following the resolved config:

```text
outputs/<dataset_identifier>/<method>/<verifier>/<model_name>/<exp_name>/
  config.yaml       # fully resolved run config
  results.jsonl     # per-example generations, scores, and metric
  metrics.json      # aggregated metrics (primary number is avg_score)
```

`<dataset_identifier>` encodes task variants, e.g. `prefeval_explicit-turn-10`
(long-context) or `math500-preference-True` (preference-guided math). For math, the
evaluator reports two selection strategies over the same candidate pool at every
budget `n ≤ N`: `weighted_sum@n` (REAR) and `majority_vote@n` (majority voting);
`avg_score` mirrors the REAR `weighted_sum` result at the full budget.

## 📝 Citation

```bibtex
@inproceedings{zhang2026rear,
  title     = {{REAR}: Test-time Preference Realignment through Reward Decomposition},
  author    = {Zhang, Fuxiang and Wang, Pengcheng and Li, Chenran and Li, Yi-Chen and
               Chen, Yuxin and Feng, Lang and Xu, Chenfeng and Tomizuka, Masayoshi and An, Bo},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
