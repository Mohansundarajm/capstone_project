"""Part A task 5: assert the reference checkpoints for the clean tables and the SQLite database."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
comp, canc = o[o["status"] == "completed"], o[o["status"] == "cancelled"]

checks = {
    "orders rows == 12309": len(o) == 12309,
    "order_id unique": o["order_id"].is_unique,
    "completed == 11110": len(comp) == 11110,
    "cancelled == 1199": len(canc) == 1199,
    "no pending rows": int((o["status"] == "pending").sum()) == 0,
    "0 missing order_amount": int(o["order_amount"].isna().sum()) == 0,
    "0 missing delivery (completed)": int(comp["delivery_minutes"].isna().sum()) == 0,
    "1199 missing delivery (cancelled)": int(canc["delivery_minutes"].isna().sum()) == 1199,
    "completed revenue == 2998668.80": round(float(comp["order_amount"].sum()), 2) == 2998668.80,
    "six cities": c["city"].nunique() == 6,
    "city counts": c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                                         "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages, all 18-60": int(c["age"].isna().sum()) == 0 and int(c["age"].min()) >= 18 and int(c["age"].max()) <= 60,
}

con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
checks["db has customers/products/orders"] = {"customers", "products", "orders"} <= tables
checks["db orders rows == 12309"] = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309
con.close()

failed = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
