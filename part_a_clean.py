"""Part A tasks 3-5: clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import os, sqlite3
import pandas as pd

RAW, CLEAN, DB = "data/raw/", "data/clean/", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)

# ---------------- customers ----------------
cust = pd.read_csv(RAW + "customers.csv")
prod = pd.read_csv(RAW + "products.csv")

cust["city"] = cust["city"].str.strip().str.title()
g = cust["gender"].fillna("").str.strip().str.lower()
cust["gender"] = g.map(lambda x: "male" if x in ("m", "male") else "female" if x in ("f", "female") else "unknown")
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
bad_age = ~cust["age"].between(10, 100)           # NaN also counts as "not between"
n_age_outside = int((cust["age"].notna() & bad_age).sum())
cust.loc[bad_age, "age"] = float("nan")
n_age_missing = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median")).round().astype(int)
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
print(f"customers: {n_age_outside} impossible ages set to missing; {n_age_missing} ages filled with city median")

# ---------------- orders ----------------
o = pd.read_csv(RAW + "orders.csv", keep_default_na=False, na_values=[""])   # keep "N/A" as text, then coerce
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])
o = o.drop_duplicates()
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
n2 = len(o)
print(f"orders: {n0} raw -> {n1} after exact-duplicate removal ({n0 - n1} removed) "
      f"-> {n2} after keeping latest record per order_id ({n1 - n2} older records removed)")

o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
n_bad_amt = int(o["order_amount"].isna().sum())
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
fill = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(fill)
print(f"orders: {n_bad_amt} missing/unparseable order_amount filled from product price")

neg = o["delivery_minutes"] < 0
n_neg = int(neg.sum())
o.loc[neg, "delivery_minutes"] = float("nan")
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median")   # median over completed rows only
need = comp & o["delivery_minutes"].isna()
n_fill = int(need.sum())
o.loc[need, "delivery_minutes"] = med[need]
print(f"orders: {n_neg} negative delivery_minutes set to missing; {n_fill} completed orders filled with category median")

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

# ---------------- store ----------------
cust.to_csv(CLEAN + "customers.csv", index=False)
prod.to_csv(CLEAN + "products.csv", index=False)
o.to_csv(CLEAN + "orders.csv", index=False)

if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
c2, o2 = cust.copy(), o.copy()
c2["signup_date"] = c2["signup_date"].dt.strftime("%Y-%m-%d")
o2["order_ts"] = o2["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c2.to_sql("customers", con, index=False)
prod.to_sql("products", con, index=False)
o2.to_sql("orders", con, index=False)
con.close()
print("written: data/clean/*.csv and data/instamart.db")
