# LLM Uncertainty — Measuring How Language Models Fail

An interactive notebook exploring how GPT-3.5-turbo handles uncertainty across 120 multiple-choice questions spanning six difficulty tiers.

## What This Is

Most discussions of LLM "hallucination" treat it as a single phenomenon. This notebook argues it is at least three distinct behaviors:

- **Knowledge uncertainty**: the model lacks a fact
- **Reasoning failure**: the model follows a bad shortcut and answers confidently anyway
- **No-context uncertainty**: the question cannot be answered from the information given

The goal is to measure how well the model's own signals — logprob confidence, token entropy, verbalized confidence, self-consistency — distinguish these cases.

## The Notebook

`llm_uncertainty_overview_v2.ipynb` covers:

1. **Calibration** — reliability diagram and per-tier confidence distributions
2. **Entropy** — token-level entropy as an uncertainty signal, and its limits
3. **Self-consistency** — sampling the same question 12 times to separate stochastic vs systematic error
4. **Verbalized confidence** — comparing stated confidence to logprob confidence (r = 0.186)
5. **Hallucination cases** — confident wrong answers at confidence ≥ 0.70
6. **Conformal prediction** — prediction sets calibrated on easy/medium/hard, tested on harder tiers

## Setup

```bash
pip install openai numpy matplotlib pandas tqdm
```

Set your OpenAI API key. In the notebook, replace `YOUR_API_KEY_FILE_PATH` with your key source, or modify the loading logic directly.

The notebook caches all API calls to `api_cache_v2.json` — running it once is sufficient.

## Key Numbers

| Tier | Accuracy |
|------|----------|
| Easy | ~100% |
| Medium | ~80% |
| Hard | ~75% |
| Tricky | ~45% |
| No-context | ~80% |
| Unknown | ~50% |

Overall: ~72% across 120 questions. Logprob–verbalized confidence correlation: **r = 0.186**.

## Companion Post

[Blog post](https://tylermartin.dev/blog/llm-uncertainty) walking through the findings without the code.
