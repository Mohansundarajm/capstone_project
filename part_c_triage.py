"""Part C: support-ticket triage with guardrails.

Pipeline per ticket: input guardrails -> PII redaction -> model call (retry with backoff) ->
output validation -> optional token budget -> routing using the Part B segments.
All model calls go through the supplied offline mock_llm.py (no key, no network).
"""
import json
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# Redaction patterns: applied in exactly this order.
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = (
    "You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.'
)
FEW_SHOT_PROMPT = (
    "You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
    'Ticket: The rider never arrived with [ORDER_ID]. -> {"category": "delivery", "reason": "order not received"}\n'
    'Ticket: The app freezes at checkout. -> {"category": "technical", "reason": "app problem"}\n'
    'Ticket: Unauthorized transaction on my card for [ORDER_ID]. -> {"category": "fraud", "reason": "unauthorized payment"}'
)

# ---------------------------------------------------------------- guardrails and redaction


def redact(text, counts=None):
    """Apply the three patterns in order; optionally add substitution counts to `counts`."""
    for name, pattern, placeholder in PII_PATTERNS:
        text, n = pattern.subn(placeholder, text)
        if counts is not None:
            counts[name] = counts.get(name, 0) + n
    return text


def contains_pii(text):
    return any(p.search(text) for _, p, _ in PII_PATTERNS)


def input_guardrail(text):
    """Return a status if the ticket must never reach the model, else None."""
    if text is None or not str(text).strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


# ---------------------------------------------------------------- resilient call


def call_with_retry(messages, ticket_id, log, sleep=time.sleep):
    """Call the mock model; retry TransientLLMError at most MAX_RETRIES times with exponential backoff + jitter."""
    for retry in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            log.append({"ticket_id": ticket_id, "retry": retry, "wait": round(wait, 4)})
            print(f"  [retry] ticket={ticket_id} retry={retry} wait={wait:.4f}s")
            sleep(wait)


# ---------------------------------------------------------------- output validation


def validate_reply(content):
    """Return (status, category). Never guesses: anything ambiguous is flagged for human review."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""


# ---------------------------------------------------------------- one full run


def run_pipeline(tickets, system_prompt, token_budget=None, sleep=time.sleep, verbose=True):
    """Process tickets in file order. Returns (results DataFrame, stats dict)."""
    mock_llm.reset()
    random.seed(42)
    rows, retry_log, prompts_sent = [], [], []
    total_tokens, transient_errors = 0, 0
    redaction_counts = {}
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        status = input_guardrail(t.text)
        if status:
            row["status"] = status
            rows.append(row)
            continue
        if token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
            rows.append(row)
            continue
        clean = redact(t.text, redaction_counts)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
        prompts_sent.append(messages)
        before = len(retry_log)
        try:
            resp = call_with_retry(messages, t.ticket_id, retry_log, sleep=sleep)
        except TransientLLMError:
            row["status"] = "flagged_invalid_output"      # permanent failure after max retries -> human review
            rows.append(row)
            continue
        transient_errors += len(retry_log) - before
        used = resp.usage.total_tokens
        total_tokens += used
        row["tokens_used"] = used
        row["status"], row["category"] = validate_reply(resp.choices[0].message.content)
        rows.append(row)
    df = pd.DataFrame(rows)
    stats = {"total_tokens": total_tokens, "retries": len(retry_log), "retry_log": retry_log,
             "prompts_sent": prompts_sent}
    return df, stats


def accuracy_report(df):
    ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((ok["category"] == ok["true_category"]).sum())
    return {"classified": len(ok), "correct": correct, "accuracy": correct / len(ok) if len(ok) else 0.0}


# ---------------------------------------------------------------- routing


def route_tickets(df, segments_path="outputs/customer_segments.csv"):
    seg = pd.read_csv(segments_path).set_index("customer_id")
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    premium = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)   # no row -> not premium
    human = out["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    fraud = ~human & (out["category"] == "fraud")
    prem = ~human & ~fraud & premium
    out["route"] = "support_queue"
    out["priority"] = "NORMAL"
    out.loc[human, "route"] = "human_review"
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


# ---------------------------------------------------------------- main


def main():
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print(f"tickets: {len(tickets)}")

    # guardrail counts
    g = tickets["text"].map(input_guardrail)
    print(f"empty={int((g == 'skipped_empty').sum())} long={int((g == 'flagged_long').sum())} "
          f"(long ticket estimates {est_tokens(tickets.loc[g == 'flagged_long', 'text'].iloc[0])} tokens)")

    # redaction over all 164 tickets
    counts = {}
    redacted = [redact(x, counts) for x in tickets["text"]]
    print("redaction substitutions:", counts, "total", sum(counts.values()))
    print("PII left after re-applying patterns:", sum(contains_pii(x) for x in redacted))

    runs = {}
    for name, prompt in [("zero-shot", ZERO_SHOT_PROMPT), ("few-shot", FEW_SHOT_PROMPT)]:
        print(f"\n=== full run: {name}")
        df, st = run_pipeline(tickets, prompt)
        for msgs in st["prompts_sent"]:                      # privacy assertion on every prompt actually sent
            assert not contains_pii(msgs[-1]["content"]), "PII reached a prompt"
        assert st["retries"] == 19, f"expected 19 transient errors, got {st['retries']}"
        waits = [r["wait"] for r in st["retry_log"]]
        print(f"retries={st['retries']} (all recovered) wait range {min(waits)}..{max(waits)}s")
        print("status counts:", df["status"].value_counts().to_dict())
        rep = accuracy_report(df)
        print(f"classified={rep['classified']} correct={rep['correct']} accuracy={rep['accuracy']*100:.1f}% "
              f"total_tokens={st['total_tokens']} tokens/classified={st['total_tokens']/rep['classified']:.1f}")
        print(f"system prompt estimated tokens: {est_tokens(prompt)}")
        runs[name] = (df, st, rep)

    zdf = runs["zero-shot"][0]
    ok = zdf[zdf["status"].isin(["classified_json", "classified_keyword"])]
    print("\nZero-shot confusion table (rows=true, cols=predicted):")
    print(pd.crosstab(ok["true_category"], ok["category"]).to_string())

    # budget demo
    budget = 6000
    bdf, bst = run_pipeline(tickets, ZERO_SHOT_PROMPT, token_budget=budget)
    bdf_out = route_tickets(bdf)
    bdf_out.to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"\nBudget demo (zero-shot, TOKEN_BUDGET={budget}): budget_exhausted="
          f"{int((bdf['status'] == 'budget_exhausted').sum())}, tokens spent={bst['total_tokens']}")

    # final results
    fdf, fst = run_pipeline(tickets, FEW_SHOT_PROMPT, token_budget=100000)
    final = route_tickets(fdf)
    final.to_csv("outputs/triage_results.csv", index=False)
    print(f"\nFinal triage_results.csv: {len(final)} rows")
    print(final.groupby(["route", "priority"]).size().to_string())


if __name__ == "__main__":
    main()
