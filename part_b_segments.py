"""Part B tasks 1-3: RFM table, K-means (K=2..8), segment profile/naming, PCA, outputs/customer_segments.csv."""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o["status"] == "completed"]

# ---------------- Task 1: RFM ----------------
g = comp.groupby("customer_id")
rfm = pd.DataFrame({
    "last_order_ts": g["order_ts"].max(),
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
})
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm.drop(columns="last_order_ts")[["recency", "frequency", "monetary"]].reset_index()
print(f"RFM customers: {len(rfm)}")
print(f"recency median={rfm.recency.median():.0f} min={rfm.recency.min()} max={rfm.recency.max()}")
print(f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}")
print(f"monetary mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}")

# ---------------- Task 2: K-means sweep ----------------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
print("\nK-means sweep:")
print(pd.DataFrame(rows).to_string(index=False))

# ---------------- Task 3: profile, name, project ----------------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print(f"\nSegment profile (K={FINAL_K}):")
print(prof.to_string())

# Names are assigned by rule from the profile, never from the arbitrary cluster number.
premium_id = prof["monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["frequency"].idxmax()
lapsing_id = rest.drop(index=loyal_id)["recency"].idxmax()
names = {premium_id: "Premium Big Spenders"}
names[loyal_id] = "Loyal Regulars"
names[lapsing_id] = "Fading Customers"
for sid in prof.index:
    names.setdefault(sid, "Occasional Shoppers")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print("\nNames:", {int(k): v for k, v in names.items()}, "| premium segment id:", int(premium_id))

pca = PCA(n_components=2).fit(X)
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)
print("written outputs/customer_segments.csv")
