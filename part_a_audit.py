"""Part A task 2: audit the raw exports (shape, dtypes, missing values, specific audit figures)."""
import pandas as pd

RAW = "data/raw/"
cust = pd.read_csv(RAW + "customers.csv")
prod = pd.read_csv(RAW + "products.csv")
# keep_default_na=False so the literal text "N/A" in order_amount is NOT silently turned into NaN
orders = pd.read_csv(RAW + "orders.csv", keep_default_na=False, na_values=[""])

for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(f"\n=== {name}: shape={df.shape}")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(summary.to_string())

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
unparseable = int((amt_num.isna() & orders["order_amount"].notna()).sum())
age = pd.to_numeric(cust["age"], errors="coerce")
genders = cust["gender"].dropna()

audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": unparseable,
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(age.isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": int(cust["city"].nunique()),
    "distinct non-blank gender spellings": int(genders.nunique()),
    "blank gender": int(cust["gender"].isna().sum()),
}
print("\n=== Audit figures")
for k, v in audit.items():
    print(f"{k}: {v}")
