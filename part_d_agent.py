"""Part D tasks 5-9: tools, ReAct loop, safe stopping, checkpoint/resume, scenario runner.

Everything here is SIMULATED: no payment service exists; issue_refund / escalate_to_human only append to a local ledger.
All model calls use the offline mock_llm.py.
"""
import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

import mock_llm
from part_d_retrieval import retrieve

DB_PATH = os.path.join("data", "instamart.db")
LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
RUNS_PATH = os.path.join("outputs", "agent_runs.jsonl")
STATE_DIR = "state"
ORDER_ID_RE = re.compile(r"^ORD-\d{6}$")
LATE_MINUTES = 45
AUTO_LIMIT_INR = 1000

AGENT_SYSTEM_PROMPT = (
    "You are a refund AGENT for Instamart support. Use only the provided tools: get_order, search_policy, "
    "calculate_refund, issue_refund, escalate_to_human. Always look up the order and the policy before deciding. "
    "Refunds above INR 1000 must be escalated to a human. Finish with a final answer that starts with one of "
    "REFUND:, ESCALATE:, DENIED: or NOT_FOUND:."
)

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the most relevant passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue (simulate) an automatic refund of up to INR 1000 for an order.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]


class SimulatedCrash(Exception):
    """Raised on purpose to test crash recovery."""


# ---------------------------------------------------------------- tools


def _db(db_path=None):
    return sqlite3.connect(db_path or DB_PATH)


def _num(v):
    return int(v) if isinstance(v, float) and v.is_integer() else v


def _lookup(order_id, db_path=None):
    with _db(db_path) as con:
        return con.execute("SELECT order_id, customer_id, status, order_amount, delivery_minutes "
                           "FROM orders WHERE order_id = ?", (order_id,)).fetchone()


def get_order(order_id, customer_id=None, db_path=None):
    """Return the order, or {"error": "order not found"}. If customer_id is given and does not own the order,
    the answer is the same 'not found' so another customer's order is never revealed."""
    if not isinstance(order_id, str) or not ORDER_ID_RE.match(order_id):
        return {"error": "order not found"}
    row = _lookup(order_id, db_path)
    if row is None or (customer_id is not None and row[1] != customer_id):
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[2], "order_amount": float(row[3]),
            "delivery_minutes": None if row[4] is None or (isinstance(row[4], float) and math.isnan(row[4]))
            else _num(float(row[4]))}


def search_policy(query):
    return {"results": retrieve(query)}


def refund_for(order_amount, delivery_minutes):
    """Pure refund rule: more than 45 minutes -> 100% of the amount, otherwise 0."""
    return round(float(order_amount), 2) if float(delivery_minutes) > LATE_MINUTES else 0.0


def calculate_refund(order_amount, delivery_minutes):
    return {"refund_inr": refund_for(order_amount, delivery_minutes)}


def _ledger_entries(ledger_path):
    if not os.path.exists(ledger_path):
        return []
    with open(ledger_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _record(tool, order_id, refund_inr, ledger_path=None, db_path=None):
    """Re-verify against the database, then append ONE ledger line. Idempotent per (tool, order_id)."""
    ledger_path = ledger_path or LEDGER_PATH
    order = get_order(order_id, db_path=db_path)
    if "error" in order:
        return {"error": "order not found"}
    expected = refund_for(order["order_amount"], order["delivery_minutes"]) if (
        order["status"] == "completed" and order["delivery_minutes"] is not None) else 0.0
    if expected == 0.0 or abs(expected - float(refund_inr)) > 0.01:
        return {"error": "refund does not match policy for this order"}
    if tool == "issue_refund" and expected > AUTO_LIMIT_INR:
        return {"error": "refund above INR 1000 must be escalated"}
    if tool == "escalate_to_human" and expected <= AUTO_LIMIT_INR:
        return {"error": "refund up to INR 1000 does not need escalation"}
    for e in _ledger_entries(ledger_path):
        if e["tool"] == tool and e["order_id"] == order_id:
            return {"refund_inr": e["refund_inr"], "duplicate": True}       # already done: do not write again
    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
             "order_id": order_id, "refund_inr": expected, "simulated": True}
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": expected}


def issue_refund(order_id, refund_inr, ledger_path=None, db_path=None):
    return _record("issue_refund", order_id, refund_inr, ledger_path, db_path)


def escalate_to_human(order_id, refund_inr, ledger_path=None, db_path=None):
    return _record("escalate_to_human", order_id, refund_inr, ledger_path, db_path)


TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
_SPEC = {t["name"]: t for t in TOOLS_SPEC}


def validate_args(name, args):
    """Return an error string, or None. Rejects unknown tools, missing/extra keys and wrong types."""
    if name not in _SPEC:
        return f"unknown tool: {name}"
    if not isinstance(args, dict):
        return "arguments must be an object"
    props, required = _SPEC[name]["parameters"]["properties"], _SPEC[name]["parameters"]["required"]
    if set(args) != set(required) or set(args) - set(props):
        return f"arguments must be exactly {required}"
    for key, val in args.items():
        kind = props[key]["type"]
        if kind == "string" and not isinstance(val, str):
            return f"{key} must be a string"
        if kind == "number" and (isinstance(val, bool) or not isinstance(val, (int, float))):
            return f"{key} must be a number"
    return None


def execute_tool(name, args, customer_id=None):
    """Look the tool up in TOOL_MAP and call it. Errors become observations, never silent successes.
    The customer id comes from the verified session, never from model-generated arguments."""
    err = validate_args(name, args)
    if err:
        return {"error": err}
    try:
        if name == "get_order":
            return TOOL_MAP[name](args["order_id"], customer_id=customer_id)
        return TOOL_MAP[name](**args)
    except Exception as exc:                                    # tool failure -> reported, not hidden
        return {"error": f"tool failed: {type(exc).__name__}"}


# ---------------------------------------------------------------- state / checkpoints


def _checkpoint_path(run_id, state_dir=None):
    return os.path.join(state_dir or STATE_DIR, f"{run_id}.json")


def save_checkpoint(state, state_dir=None):
    path = _checkpoint_path(state["run_id"], state_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, default=str)
    os.replace(tmp, path)                                       # atomic: a crash never leaves half a file


def _handoff(state, reason):
    done = [s["tool"] for s in state["steps"]]
    return (f"HANDOFF to human agent: {reason}. Steps completed: {done or 'none'}. "
            f"Request: {state['messages'][1]['content']}")


def _result(state, status, final):
    refund = 0.0
    for s in state["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            refund = s["observation"]["refund_inr"]
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": final,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "refund_inr": refund, "tools_executed": list(state["tools_executed"])}


def _loop(state, state_dir=None, crash_after_tools=None):
    start = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - start >= state["time_limit_seconds"]:                 # top of each iteration
            return _result(state, "FALLBACK_TIMEOUT", _handoff(state, "time limit reached"))
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:                             # stop BEFORE running the tool
            return _result(state, "FALLBACK_TOKEN_LIMIT", _handoff(state, "token limit exceeded"))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            status = final.split(":", 1)[0].strip() if ":" in final else "UNKNOWN"
            return _result(state, status, final)
        tc = msg.tool_calls[0]
        call = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [call]})
        obs = execute_tool(tc.name, tc.arguments, state.get("customer_id"))
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        save_checkpoint(state, state_dir)                                           # after every observation
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")
    return _result(state, "FALLBACK_MAX_ITERATIONS", _handoff(state, "maximum iterations reached"))


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              customer_id=None, run_id=None, state_dir=None, crash_after_tools=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "customer_id": customer_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [], "tools_executed": [],
             "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(state, state_dir, crash_after_tools)


def resume_agent(run_id, state_dir=None):
    """Load state/<run_id>.json and continue the loop where it stopped (the time budget restarts)."""
    with open(_checkpoint_path(run_id, state_dir), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state, state_dir)


# ---------------------------------------------------------------- scenario runner


def _write_runs(results, path=RUNS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: r[k] for k in ("scenario_id", "status", "final_answer", "llm_calls",
                                                  "tokens_used", "steps")}, default=str) + "\n")


def main():
    os.makedirs("outputs", exist_ok=True)
    for p in (LEDGER_PATH, RUNS_PATH):
        if os.path.exists(p):
            os.remove(p)                                        # reproducible: every run starts from a clean ledger
    if os.path.isdir(STATE_DIR):
        for fn in os.listdir(STATE_DIR):
            os.remove(os.path.join(STATE_DIR, fn))

    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    results, table = [], []
    for r in sc.itertuples():
        res = run_agent(r.message, r.scenario_id, customer_id=r.customer_id)
        results.append(res)
        ok = res["status"] == r.expected_outcome or (r.expected_outcome == "FALLBACK"
                                                    and res["status"] == "FALLBACK_MAX_ITERATIONS")
        ok = ok and abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01
        table.append((r.scenario_id, res["status"], f"{res['refund_inr']:.2f}", res["llm_calls"],
                      res["tokens_used"], "OK" if ok else "MISMATCH"))
    _write_runs(results)
    print("scenario | status | refund_inr | model calls | tokens | vs expected")
    for row in table:
        print(" | ".join(str(x) for x in row))

    s1 = sc.iloc[0]
    d1 = run_agent(s1.message, "S1", token_limit=200, customer_id=s1.customer_id, run_id="S1-tokenlimit")
    print(f"\nDemo S1 token_limit=200 -> {d1['status']} (llm_calls={d1['llm_calls']}, tokens={d1['tokens_used']}, "
          f"tools executed={d1['tools_executed']})")
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, customer_id=s1.customer_id, run_id="S1-timeout")
    print(f"Demo S1 time_limit_seconds=0 -> {d2['status']} (llm_calls={d2['llm_calls']})")

    try:
        run_agent(s1.message, "S1", customer_id=s1.customer_id, run_id="S1-crash", crash_after_tools=2)
    except SimulatedCrash as exc:
        print(f"\nCrash demo: {exc}")
    d3 = resume_agent("S1-crash")
    print(f"Resumed S1 -> {d3['status']} refund={d3['refund_inr']:.2f} llm_calls={d3['llm_calls']}")
    print("Tools executed across crash + resume, in order:", d3["tools_executed"])
    print("Ledger lines:", len(_ledger_entries(LEDGER_PATH)), "| checkpoints in state/:", sorted(os.listdir(STATE_DIR)))


if __name__ == "__main__":
    main()
