"""Part B tasks 4-8: lapse cohort/features (no leakage), pipeline, tuning, balance experiment, interpretation, retention list."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")
pd.set_option("display.width", 200)

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
o = o.merge(prod[["product_id", "category"]], on="product_id", how="left")

# ---------------- Task 4: cohort, label, features ----------------
before = o[o["order_ts"] < CUTOFF]                       # ONLY orders before the cutoff feed the features
comp_b = before[before["status"] == "completed"]
cohort_ids = comp_b["customer_id"].unique()

sept_active = set(o[(o["status"] == "completed") & (o["order_ts"] >= CUTOFF) & (o["order_ts"] < END)]["customer_id"])
y_all = pd.Series({c: int(c not in sept_active) for c in cohort_ids}, name="lapsed")

g = comp_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax()),
})
final_b = before[before["status"].isin(["completed", "cancelled"])]
canc_rate = final_b.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
feat["cancel_rate"] = canc_rate.reindex(feat.index).fillna(0.0)
feat = feat.join(cust.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
NOM = ["city", "gender", "fav_category"]
X, y = feat[NUM + NOM], y_all.reindex(feat.index)
print(f"cohort={len(X)}  lapsed={int(y.sum())} ({y.mean()*100:.2f}%)")
assert not X.isna().any().any(), "a feature column contains missing values"
print("no feature column contains a missing value: OK")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train={len(X_tr)} test={len(X_te)}")

def prf(yt, yp):
    return (precision_score(yt, yp, zero_division=0), recall_score(yt, yp, zero_division=0), f1_score(yt, yp, zero_division=0))

# ---------------- Task 6a: baseline ----------------
base_pred = np.zeros(len(y_te), dtype=int)
print(f"\nBaseline (always not-lapsed): accuracy={accuracy_score(y_te, base_pred)*100:.1f}% "
      f"lapsed-F1={f1_score(y_te, base_pred, zero_division=0):.3f}")

# ---------------- Task 5: preprocessing, fitted on training rows only (inside Pipeline) ----------------
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOM)])

# ---------------- Task 6b/c: tuning ----------------
lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_tr, y_tr)
print(f"\nLR best params={lr_gs.best_params_} cv-F1={lr_gs.best_score_:.3f}")
print(f"RF best params={rf_gs.best_params_} cv-F1={rf_gs.best_score_:.3f}")

# scaler vs full-cohort mean of frequency (task 5)
scaler = lr_gs.best_estimator_.named_steps["pre"].named_transformers_["num"]
print(f"\nscaler learned mean(frequency) on train = {scaler.mean_[NUM.index('frequency')]:.3f} | full-cohort mean = {X['frequency'].mean():.3f}")

# ---------------- Task 6d: test-set evaluation ----------------
results = {}
for name, gs in [("LogisticRegression", lr_gs), ("RandomForest", rf_gs)]:
    pred = gs.predict(X_te)
    p, r, f = prf(y_te, pred)
    results[name] = (p, r, f)
    print(f"\n{name} test: precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    print("confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_te, pred))

# model selection on CROSS-VALIDATED F1 (not on the test set)
sel_name, sel_gs = max([("LogisticRegression", lr_gs), ("RandomForest", rf_gs)], key=lambda t: t[1].best_score_)
print(f"\nSelected model (best cv-F1): {sel_name}  test F1={results[sel_name][2]:.3f}")
assert results[sel_name][2] >= 0.55, "selected model must reach test F1 >= 0.55"

# ---------------- Task 7: oversample the TRAINING set only ----------------
minority_idx = y_tr[y_tr == 1].index
n_add = int((y_tr == 0).sum() - (y_tr == 1).sum())
extra = pd.Series(minority_idx).sample(n=n_add, replace=True, random_state=42).to_numpy()
X_bal = pd.concat([X_tr, X_tr.loc[extra]])
y_bal = pd.concat([y_tr, y_tr.loc[extra]])
print(f"\nbalanced train: {int((y_bal==0).sum())} not-lapsed / {int((y_bal==1).sum())} lapsed")
lr_best_C = lr_gs.best_params_["clf__C"]
lr_bal = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000, C=lr_best_C))]).fit(X_bal, y_bal)
pb = prf(y_te, lr_bal.predict(X_te))
pu = prf(y_te, lr_gs.predict(X_te))
print("LR un-balanced : precision=%.3f recall=%.3f F1=%.3f" % pu)
print("LR over-sampled: precision=%.3f recall=%.3f F1=%.3f" % pb)

# ---------------- Task 8a: interpretation ----------------
rf_best = rf_gs.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 RF feature importances:\n", imp.head(5).round(3).to_string())
lr_names = lr_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
print("\nLR coefficients (standardised numeric features):")
for f_ in ["num__recency", "num__frequency", "num__monetary", "num__tenure_days", "num__cancel_rate"]:
    print(f"  {f_}: {coef[f_]:+.3f}")

# ---------------- Task 8b: persist and reload ----------------
joblib.dump(sel_gs.best_estimator_, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(reloaded.predict(X_te), sel_gs.best_estimator_.predict(X_te))
print(f"\nreloaded model reproduces test predictions exactly: {same}")
assert same

# ---------------- Task 8c: retention list (illustrative: Premium flag is a whole-window label) ----------------
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
proba = pd.Series(reloaded.predict_proba(X_te)[:, 1], index=X_te.index)
prem_test = [c for c in X_te.index if c in seg.index and bool(seg.loc[c, "is_premium"])]
ret = pd.DataFrame({"customer_id": prem_test,
                    "lapse_probability": proba.loc[prem_test].round(4).to_numpy(),
                    "actually_lapsed": y_te.loc[prem_test].to_numpy()})
ret = ret.sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"\nPremium customers in test set: {len(prem_test)}; retention list rows={len(ret)}; truly lapsed among them={int(ret['actually_lapsed'].sum())}")
print(ret.to_string(index=False))
