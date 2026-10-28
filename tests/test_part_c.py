"""Tests for Part C: guardrails, redaction, validation, retries, budget, routing."""
import pandas as pd
import pytest

import mock_llm
import part_c_triage as pc


def test_input_guardrails():
    assert pc.input_guardrail("") == "skipped_empty"
    assert pc.input_guardrail("   \n ") == "skipped_empty"
    assert pc.input_guardrail("word " * 400) == "flagged_long"       # 520 estimated tokens
    assert pc.input_guardrail("my app crashes") is None


def test_redaction_order_and_counts():
    counts = {}
    out = pc.redact("Mail a.b@x.com about ORD-123456, call 9876543210.", counts)
    assert out == "Mail [EMAIL] about [ORDER_ID], call [PHONE]."
    assert counts == {"email": 1, "phone": 1, "order_id": 1}
    assert not pc.contains_pii(out)


def test_redaction_on_all_tickets():
    t = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    counts = {}
    red = [pc.redact(x, counts) for x in t["text"]]
    assert counts == {"email": 55, "phone": 91, "order_id": 132}
    assert not any(pc.contains_pii(x) for x in red)


@pytest.mark.parametrize("reply,expected", [
    ('{"category": "fraud", "reason": "x"}', ("classified_json", "fraud")),
    ("The customer appears to have a billing-related problem.", ("classified_keyword", "billing")),
    ('{"category": "shipping", "reason": "x"}', ("flagged_invalid_output", "")),
    ("billing and fraud both", ("flagged_invalid_output", "")),         # ambiguous -> never guess
    ("no idea", ("flagged_invalid_output", "")),
    ("[1, 2]", ("flagged_invalid_output", "")),                          # valid JSON, wrong shape
    ("", ("flagged_invalid_output", "")),
])
def test_validate_reply(reply, expected):
    assert pc.validate_reply(reply) == expected


def test_retry_stops_after_three_retries(monkeypatch):
    calls = []

    def always_fail(messages, tools=None):
        calls.append(1)
        raise mock_llm.TransientLLMError("503")
    monkeypatch.setattr(mock_llm, "chat", always_fail)
    log = []
    with pytest.raises(mock_llm.TransientLLMError):
        pc.call_with_retry([{"role": "system", "content": "CLASSIFY"}, {"role": "user", "content": "x"}],
                           "T1", log, sleep=lambda s: None)
    assert len(calls) == 4 and [r["retry"] for r in log] == [0, 1, 2]
    for r in log:                                                         # wait inside the jitter range
        base = pc.BASE_DELAY * 2 ** r["retry"]
        assert base * 0.9 - 1e-4 <= r["wait"] <= base * 1.15 + 1e-4


def test_full_run_reference_counts_and_no_pii_in_prompts():
    t = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    df, st = pc.run_pipeline(t, pc.FEW_SHOT_PROMPT, sleep=lambda s: None)
    assert df["status"].value_counts().to_dict() == {"classified_json": 135, "classified_keyword": 12,
                                                    "flagged_invalid_output": 9, "skipped_empty": 4,
                                                    "flagged_long": 4}
    assert st["retries"] == 19
    assert all(not pc.contains_pii(m[-1]["content"]) for m in st["prompts_sent"])
    assert len(st["prompts_sent"]) == 156                                 # empty and long tickets never sent
    assert pc.accuracy_report(df)["accuracy"] == 1.0


def test_prompts_follow_the_rules():
    assert "CLASSIFY" in pc.ZERO_SHOT_PROMPT and "EXAMPLES:" not in pc.ZERO_SHOT_PROMPT
    assert "CLASSIFY" in pc.FEW_SHOT_PROMPT and "EXAMPLES:" in pc.FEW_SHOT_PROMPT
    assert all(c in pc.FEW_SHOT_PROMPT for c in pc.CATEGORIES)


def test_budget_exhausts_tickets():
    t = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    df, st = pc.run_pipeline(t, pc.ZERO_SHOT_PROMPT, token_budget=6000, sleep=lambda s: None)
    assert (df["status"] == "budget_exhausted").sum() == 51


def test_routing_rules(tmp_path):
    seg = pd.DataFrame({"customer_id": ["P1", "N1"], "segment_name": ["Premium Big Spenders", "Loyal Regulars"],
                        "is_premium": [True, False]})
    seg.to_csv(tmp_path / "seg.csv", index=False)
    df = pd.DataFrame({"ticket_id": list("abcdef"), "customer_id": ["P1", "P1", "N1", "ZZ", "P1", "N1"],
                       "status": ["classified_json", "classified_json", "classified_json", "classified_keyword",
                                  "flagged_invalid_output", "skipped_empty"],
                       "category": ["fraud", "billing", "fraud", "delivery", "", ""], "tokens_used": [1] * 6,
                       "true_category": [""] * 6})
    out = pc.route_tickets(df, str(tmp_path / "seg.csv")).set_index("ticket_id")
    assert out.loc["a", ["route", "priority"]].tolist() == ["security_team", "HIGH"]     # fraud beats premium
    assert out.loc["b", ["route", "priority"]].tolist() == ["premium_desk", "HIGH"]
    assert out.loc["c", ["route", "priority"]].tolist() == ["security_team", "HIGH"]
    assert out.loc["d", ["route", "priority"]].tolist() == ["support_queue", "NORMAL"]   # no segment row
    assert out.loc["e", ["route", "priority"]].tolist() == ["human_review", "NORMAL"]
    assert out.loc["f", ["route", "priority"]].tolist() == ["human_review", "NORMAL"]


def test_output_files_schema():
    cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    for f in ("outputs/triage_results.csv", "outputs/triage_budget_demo.csv"):
        d = pd.read_csv(f, keep_default_na=False)
        assert list(d.columns) == cols and len(d) == 164
