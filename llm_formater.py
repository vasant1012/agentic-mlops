from typing import List, Dict, Any


def format_runs_for_llm(runs: List[Dict[str, Any]], max_runs: int = 10) -> str:
    """
    Convert structured runs to a compact text block for prompting.
    """
    sb = []
    sb.append(
        f"Total runs returned: {len(runs)}. Showing up to {max_runs} runs (most recent first).\n")  # NOQA E501
    for i, r in enumerate(runs[:max_runs], 1):
        params = ", ".join(
            [f"{k}={v}" for k, v in r["params"].items()]) if r["params"] else "none"  # NOQA E501
        metrics = ", ".join([f"{k}={v:.6f}" if isinstance(
            v, (int, float)) else f"{k}={v}" for k, v in r["metrics"].items()]) if r["metrics"] else "none"  # NOQA E501
        sb.append(f"RUN {i}: id={r['run_id']}, status={r['status']}")
        sb.append(f"  params: {params}")
        sb.append(f"  metrics: {metrics}")
        if r.get("tags"):
            sb.append(f"  tags: {r.get('tags')}")
        sb.append("")  # blank line
    return "\n".join(sb)
