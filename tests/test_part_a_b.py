"""Tests for Parts A and B: they check the artifacts produced by the documented pipeline commands."""
import sqlite3
import subprocess
import sys

import joblib
import pandas as pd
import pytest


def test_validation_script_passes():
    out = subprocess.run([sys.executable, "validate.py"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().endswith("ALL CHECKPOINTS PASSED")


def test_clean_orders_one_row_per_order_and_no_pending():
    o = pd.read_csv("data/clean/orders.csv")
    assert o["order_id"].is_unique and len(o) == 12309
    assert set(o["status"]) == {"completed", "cancelled"}
    assert o["order_amount"].notna().all()


def test_clean_customers_rules():
    c = pd.read_csv("data/clean/customers.csv")
    assert c["city"].nunique() == 6
    assert set(c["gender"]) == {"male", "female", "unknown"}
    assert c["age"].between(18, 60).all()


def test_foreign_keys_hold():
    con = sqlite3.connect("data/instamart.db")
    bad_c = con.execute("SELECT COUNT(*) FROM orders o LEFT JOIN customers c USING(customer_id) "
                        "WHERE c.customer_id IS NULL").fetchone()[0]
    bad_p = con.execute("SELECT COUNT(*) FROM orders o LEFT JOIN products p USING(product_id) "
                        "WHERE p.product_id IS NULL").fetchone()[0]
    assert bad_c == 0 and bad_p == 0


def test_seven_sql_queries_reference_results():
    out = subprocess.run([sys.executable, "run_queries.py"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    s = out.stdout
    for expected in ("Q1 customers: 30", "Q2 rows: 30", "Q3 products above average price: 11",
                     "Q4 customers without completed order: 25", "Q5 customers above average: 550",
                     "Q6 rows with RANK != ROW_NUMBER: 8", "Q7 daily rows: 183"):
        assert expected in s


def test_segments_file_schema_and_premium_flag():
    s = pd.read_csv("outputs/customer_segments.csv")
    assert list(s.columns) == ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name",
                               "is_premium"]
    assert len(s) == 1175
    means = s.groupby("segment_id")["monetary"].mean()
    premium_ids = set(s.loc[s["is_premium"], "segment_id"])
    assert premium_ids == {means.idxmax()}                       # Premium = highest mean monetary only


def test_rfm_matches_orders_for_one_customer():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    s = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    cid = s.index[0]
    mine = o[(o["customer_id"] == cid) & (o["status"] == "completed")]
    assert s.loc[cid, "frequency"] == mine["order_id"].nunique()
    assert s.loc[cid, "monetary"] == pytest.approx(mine["order_amount"].sum(), abs=0.01)
    assert s.loc[cid, "recency"] == (pd.Timestamp("2025-10-01") - mine["order_ts"].max()).days


def test_lapse_model_loads_and_predicts():
    model = joblib.load("outputs/lapse_model.joblib")
    row = pd.DataFrame([{"recency": 10, "frequency": 4, "monetary": 900.0, "avg_discount": 8.0, "cancel_rate": 0.1,
                         "avg_delivery": 25.0, "age": 30, "tenure_days": 200, "city": "Pune", "gender": "male",
                         "fav_category": "Dairy"}])
    p = model.predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0


def test_retention_list_schema():
    r = pd.read_csv("outputs/retention_list.csv")
    assert list(r.columns) == ["customer_id", "lapse_probability", "actually_lapsed"]
    assert 1 <= len(r) <= 10
    assert r["lapse_probability"].is_monotonic_decreasing
