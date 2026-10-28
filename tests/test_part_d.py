"""Tests for Part D: retrieval, tools, agent safety, idempotency and crash recovery (temporary DB and ledger)."""
import json
import sqlite3

import pytest

import part_d_agent as ag
import part_d_retrieval as rt


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE orders (order_id TEXT, customer_id TEXT, status TEXT, order_amount REAL, "
                "delivery_minutes REAL)")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
        ("ORD-900001", "C1", "completed", 200.0, 60.0),     # late, small -> refund 200
        ("ORD-900002", "C1", "completed", 1500.0, 70.0),    # late, big   -> escalate
        ("ORD-900003", "C1", "completed", 300.0, 30.0),     # on time     -> denied
        ("ORD-900004", "C1", "cancelled", 100.0, None),     # cancelled   -> denied
        ("ORD-900005", "C2", "completed", 250.0, 55.0)])    # other customer
    con.commit(); con.close()
    monkeypatch.setattr(ag, "DB_PATH", str(db))
    monkeypatch.setattr(ag, "LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    return tmp_path


def ledger(env):
    p = env / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def run(env, msg, cid="C1", **kw):
    return ag.run_agent(msg, kw.pop("scenario_id", "T"), customer_id=cid, state_dir=str(env / "state"), **kw)


# ---------------- retrieval ----------------

def test_chunk_counts():
    idx = rt.PolicyIndex()
    counts = {}
    for c in idx.chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    assert len(idx.chunks) == 15
    assert counts == {"refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3,
                      "payment_policy.txt": 3, "privacy_policy.txt": 2}


def test_chunk_boundaries():
    words = [f"w{i}" for i in range(100)]
    ch = rt.chunk_words(" ".join(words))
    assert ch[0].split()[0] == "w0" and ch[0].split()[-1] == "w39"
    assert ch[1].split()[0] == "w30"
    assert ch[-1].split()[-1] == "w99" and len(ch) == 3


def test_retrieval_top3_from_refund_policy():
    hits = rt.retrieve("refund policy late delivery")
    assert [h["source"] for h in hits] == ["refund_policy.txt"] * 3
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_grounded_answer_and_no_evidence_fallback():
    text, sources = rt.answer("Who must approve a refund above INR 1000?")
    assert "human support lead" in text and "refund_policy.txt" in sources
    text, _ = rt.answer("What is the favourite colour of the company mascot?")
    assert text.startswith("I could not find")


def test_evaluation_reference_and_zero_overlap_breaks():
    r = rt.evaluate(overlap=10)
    assert (r["doc_in_top3"], r["fact_in_answer"], r["fallback"]) == (12, 12, 2)
    r0 = rt.evaluate(overlap=0)
    assert r0["fact_in_answer"] < 12


# ---------------- tools ----------------

def test_get_order_and_ownership(env):
    o = ag.get_order("ORD-900001")
    assert o == {"order_id": "ORD-900001", "status": "completed", "order_amount": 200.0, "delivery_minutes": 60}
    assert ag.get_order("ORD-999999") == {"error": "order not found"}
    assert ag.get_order("ORD-900005", customer_id="C1") == {"error": "order not found"}   # another customer's order
    assert ag.get_order("1; DROP TABLE orders") == {"error": "order not found"}           # bad id never reaches SQL
    assert ag.get_order("ORD-900004")["delivery_minutes"] is None


def test_calculate_refund_rules():
    assert ag.calculate_refund(200, 46) == {"refund_inr": 200.0}
    assert ag.calculate_refund(200, 45) == {"refund_inr": 0.0}
    assert ag.calculate_refund(1999, 80) == {"refund_inr": 1999.0}


def test_validate_args_rejects_bad_calls():
    assert ag.validate_args("get_order", {"order_id": "ORD-100001"}) is None
    assert ag.validate_args("drop_database", {}) is not None
    assert ag.validate_args("get_order", {"order_id": 5}) is not None
    assert ag.validate_args("get_order", {"order_id": "x", "extra": 1}) is not None
    assert ag.validate_args("calculate_refund", {"order_amount": True, "delivery_minutes": 5}) is not None


def test_issue_refund_checks_database_and_is_idempotent(env):
    assert "error" in ag.issue_refund("ORD-900003", 300.0)            # on time
    assert "error" in ag.issue_refund("ORD-900004", 100.0)            # cancelled
    assert "error" in ag.issue_refund("ORD-900001", 999.0)            # wrong amount
    assert "error" in ag.issue_refund("ORD-900002", 1500.0)           # above INR 1000 must be escalated
    assert ledger(env) == []
    first = ag.issue_refund("ORD-900001", 200.0)
    second = ag.issue_refund("ORD-900001", 200.0)
    assert first == {"refund_inr": 200.0} and second["duplicate"] is True
    entries = ledger(env)
    assert len(entries) == 1 and entries[0]["simulated"] is True and entries[0]["tool"] == "issue_refund"
    assert "error" in ag.escalate_to_human("ORD-900001", 200.0)       # small refunds are not escalated


# ---------------- agent ----------------

def test_agent_refund_flow(env):
    r = run(env, "My order ORD-900001 reached me very late. I would like a refund.")
    assert r["status"] == "REFUND" and r["refund_inr"] == 200.0 and r["llm_calls"] == 5
    assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert len(ledger(env)) == 1


def test_agent_escalates_big_refund(env):
    r = run(env, "Order ORD-900002 arrived far too late, please refund me.")
    assert r["status"] == "ESCALATE" and ledger(env)[0]["tool"] == "escalate_to_human"


def test_agent_denials_write_nothing(env):
    assert run(env, "Please refund order ORD-900003")["status"] == "DENIED"
    assert run(env, "Money back for ORD-900004")["status"] == "DENIED"
    assert run(env, "Refund ORD-999999 please")["status"] == "NOT_FOUND"
    assert ledger(env) == []


def test_agent_hides_other_customers_order(env):
    r = run(env, "Refund ORD-900005 please", cid="C1")
    assert r["status"] == "NOT_FOUND" and ledger(env) == []


def test_agent_needs_info_without_order_id(env):
    assert run(env, "I want a refund")["status"] == "NEEDS_INFO"


def test_tool_failure_is_never_success(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setitem(ag.TOOL_MAP, "get_order", boom)
    r = run(env, "Refund ORD-900001 please")
    assert r["steps"][0]["observation"] == {"error": "tool failed: RuntimeError"}
    assert r["status"] != "REFUND" and ledger(env) == []


def test_max_iterations_and_other_limits(env):
    r = run(env, "[LOOP] Refund order ORD-900001.")
    assert r["status"] == "FALLBACK_MAX_ITERATIONS" and r["llm_calls"] == 6 and "HANDOFF" in r["final_answer"]
    assert r["tokens_used"] <= 6000                                    # default limit does not stop it early
    t = run(env, "Refund ORD-900001 please", token_limit=200)
    assert t["status"] == "FALLBACK_TOKEN_LIMIT" and "issue_refund" not in t["tools_executed"]
    o = run(env, "Refund ORD-900001 please", time_limit_seconds=0)
    assert o["status"] == "FALLBACK_TIMEOUT" and o["llm_calls"] == 0 and o["steps"] == []
    assert ledger(env) == []


def test_crash_and_resume_executes_each_tool_once(env):
    with pytest.raises(ag.SimulatedCrash):
        run(env, "Refund ORD-900001 please", run_id="crash1", crash_after_tools=2)
    ckpt = json.loads((env / "state" / "crash1.json").read_text())
    assert ckpt["tools_executed"] == ["get_order", "search_policy"] and ckpt["tokens_used"] > 0
    r = ag.resume_agent("crash1", state_dir=str(env / "state"))
    assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert r["status"] == "REFUND" and r["refund_inr"] == 200.0
    assert len(ledger(env)) == 1


def test_crash_between_refund_and_checkpoint_does_not_double_refund(env):
    """Worst case: issue_refund wrote the ledger but the checkpoint was lost; the resumed agent asks again."""
    with pytest.raises(ag.SimulatedCrash):
        run(env, "Refund ORD-900001 please", run_id="crash2", crash_after_tools=3)
    ag.issue_refund("ORD-900001", 200.0)                               # the action that happened before the crash
    r = ag.resume_agent("crash2", state_dir=str(env / "state"))        # replays issue_refund
    assert r["status"] == "REFUND" and r["steps"][-1]["observation"].get("duplicate") is True
    assert len(ledger(env)) == 1


def test_real_output_files_exist_and_have_content():
    runs = [json.loads(x) for x in open("outputs/agent_runs.jsonl")]
    assert len(runs) == 7
    for r in runs:
        assert {"scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"} <= set(r)
    assert [r["status"] for r in runs] == ["REFUND", "REFUND", "ESCALATE", "DENIED", "DENIED", "NOT_FOUND",
                                           "FALLBACK_MAX_ITERATIONS"]
    led = [json.loads(x) for x in open("outputs/refund_ledger.jsonl")]
    assert {e["tool"] for e in led} <= {"issue_refund", "escalate_to_human"} and len(led) >= 3
