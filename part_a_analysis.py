"""Part A task 7: pandas cross-check of Q2 and exploratory statistics of completed order_amount."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(
    p[["product_id", "category"]], on="product_id")

# (a) pivot cross-check against SQL Q2
pv = pd.pivot_table(comp, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True)
print("Pivot grand total:", round(pv.loc["All", "All"], 2))
print("Largest city/category cell:", pv.drop(index="All", columns="All").stack().idxmax(),
      round(pv.drop(index="All", columns="All").stack().max(), 2))

by_city = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                   orders=("order_amount", "count")).round(2)
print("\nPer-city sum / mean / count:\n", by_city.to_string())

# (b) distribution of completed order_amount
a = comp["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean={a.mean():.2f} median={a.median():.2f} skew={a.skew():.2f} Q1={q1:.2f} Q3={q3:.2f}")
print(f"fences: lower={lo:.2f} upper={hi:.2f} | below lower={int((a < lo).sum())} above upper={int((a > hi).sum())}")

corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:\n", corr.round(3).to_string())
pairs = corr.where(~(corr.index.to_numpy()[:, None] == corr.columns.to_numpy()[None, :])).abs().stack()
print("Largest |r| pair:", pairs.idxmax(), round(corr.loc[pairs.idxmax()], 3))
