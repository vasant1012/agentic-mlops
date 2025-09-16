SUMMARY_PROMPT = """You are an ML experiment analyst. Given the following MLflow run summaries, produce: # NOQA E501
1) Short top-line summary (2-3 sentences).
2) Best runs by validation metric (name metric if present, otherwise say 'no val metric found').
3) Up to 3 concise reasons for failures or anomalous runs (e.g., learning rate too high, OOM).
4) Short bullet list of the top 3 hyperparameter configurations to try next with exact suggested values (e.g., lr=0.0008, batch_size=128).
5) A "confidence" line describing how confident you are in the suggestions (low/medium/high) and why.

Runs:
{runs_text}

Answer in clear, bullet points and short sentences. Keep responses precise and action-oriented.
"""

RECOMMEND_PROMPT = """You are a hyperparameter tuning advisor. Use the following runs to: # NOQA E501
- Identify trends in hyperparameters vs metric (e.g., param X increasing helps).
- Propose 5 specific next trials. For each trial provide: {trial_id}, list of hyperparameters (lr, batch_size, weight_decay, other relevant), and a one-line rationale.

Runs:
{runs_text}

Return JSON array of trials like:
[
  {{"trial_id":"t1", "params": {{"lr":0.001,"batch_size":64}}, "reason":"..."}},
  ...
]
If you are not confident provide best-effort suggestions and a short explanation.
"""