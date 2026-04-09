"""
Pharma B2B Quotation Environment — Inference Script
=================================================
Mandatory variables (set in environment or .env):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    OPENAI_API_KEY The API key for the LLM.

STDOUT FORMAT
    [START] task=<task_name> env=pharma-b2b model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import warnings
import textwrap
from typing import List, Optional

# ── Load .env — check root first, then env/ subfolder ────────────────────────
for _env_candidate in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "env", ".env"),
]:
    if os.path.exists(_env_candidate):
        with open(_env_candidate, "r") as _f:
            for _line in _f:
                if "=" in _line and not _line.startswith("#"):
                    _k, _v = _line.strip().split("=", 1)
                    os.environ.setdefault(_k, _v.strip('"').strip("'"))
        break

warnings.filterwarnings("ignore")

# ── Make sure the project root is on sys.path ─────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from env.environment import PharmaQuotationEnv, TASKS, normalize_score

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

if not OPENAI_API_KEY:
    # Provide a dummy key for Phase 2 validation dry-runs
    OPENAI_API_KEY = "sk-dummy-key-for-openenv-validation"
    print("[WARNING] OPENAI_API_KEY unset. Falling back to dummy key.", flush=True)

MAX_STEPS   = 12
TEMPERATURE = 0.0   # deterministic for reproducibility
MAX_TOKENS  = 100

# Tasks to run
TASK_NAMES = list(TASKS.keys())
BENCHMARK  = "pharma-b2b"

# ── Stdout helpers ────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Quote action so spaces in brand/supplier names don't break log parsers
    action_safe = f"'{action}'"
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── LLM helper ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent operating inside a Pharma B2B Distribution Quotation system.
    Your goal is to source medications for business clients with optimal margins.

    Available actions (return ONLY ONE, exactly as shown):
      search_brands:<generic_name>
      select_brand:<brand_name>
      select_supplier:<supplier_name>
      request_confirmation
      calculate_price:<number>
      finalize

    Rules:
    - Do NOT repeat an action already listed in Past Actions Taken.
    - For calculate_price, supply a sell price at least 8% above the buy rate.
    - Call finalize only after: brand selected, supplier selected,
      request_confirmation done, and calculate_price done.
    - Return EXACTLY the action string — no explanation, no extra text.
""").strip()

def get_action(client: OpenAI, state_str: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": state_str},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Guard: keep only the first non-blank line
        text = next((ln.strip() for ln in text.splitlines() if ln.strip()), "finalize")
        return text
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return "finalize"   # safe fallback

# ── Episode runner ────────────────────────────────────────────────────────────
def run_task(client: OpenAI, task_name: str) -> float:
    """Run a single task episode; return normalised score ∈ [0.0, 1.0]."""
    env     = PharmaQuotationEnv(task=task_name)
    rewards: List[float] = []
    steps_taken = 0
    success     = False
    score       = 0.0
    done        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_action(client, str(state))
            state, reward, done, info = env.step(action)

            reward  = reward or 0.0
            error   = info.get("error") if info else None
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward,
                     done=done, error=error)

            if done:
                break

        score   = env.score()          # normalised [0, 1]
        success = score >= 0.8         # benchmark pass threshold

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    all_scores: List[float] = []
    for task_name in TASK_NAMES:
        s = run_task(client, task_name)
        all_scores.append(s)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"\n[SUMMARY] tasks={len(all_scores)} "
        f"avg_score={avg:.2f} "
        f"scores={','.join(f'{s:.2f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
