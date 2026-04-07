"""
etl_pipeline.py
Fraud / Anomaly Monitoring & Investigation Dashboard
Part B — Data Architecture & ETL Pipeline

Run: python etl/etl_pipeline.py

Outputs (saved to /data/generated/):
  1. fact_orders_enriched.csv
  2. fact_user_risk_weekly.csv
  3. investigation_queue.csv
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "..", "data", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "..", "data", "generated")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================
# STEP 1 — LOAD RAW FILES
# =============================================================
def load_raw():
    print("\n" + "="*60)
    print("STEP 1: Loading raw files")
    print("="*60)

    users       = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
    sessions    = pd.read_csv(os.path.join(RAW_DIR, "sessions.csv"))
    orders      = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    order_items = pd.read_csv(os.path.join(RAW_DIR, "order_items.csv"))
    payments    = pd.read_csv(os.path.join(RAW_DIR, "payments.csv"))
    shipments   = pd.read_csv(os.path.join(RAW_DIR, "shipments.csv"))
    refunds     = pd.read_csv(os.path.join(RAW_DIR, "refunds.csv"))
    coupons     = pd.read_csv(os.path.join(RAW_DIR, "coupons.csv"))

    with open(os.path.join(RAW_DIR, "products.json"), "r") as f:
        raw = json.load(f)
    products = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])

    for name, df in [("users", users), ("sessions", sessions),
                     ("orders", orders), ("order_items", order_items),
                     ("payments", payments), ("shipments", shipments),
                     ("refunds", refunds), ("coupons", coupons),
                     ("products", products)]:
        print(f"  {name:15s}: {df.shape[0]:>7,} rows  x  {df.shape[1]} cols")

    return (users, sessions, orders, order_items,
            payments, shipments, refunds, coupons, products)


# =============================================================
# STEP 2 — CLEAN EACH TABLE
# =============================================================
def clean_df(df, name, id_col=None, date_cols=None):
    """Strip, lowercase strings, parse dates, drop duplicates."""
    print(f"  Cleaning {name} ...")

    # Standardise string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})

    # Parse dates
    for col in (date_cols or []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"    exact dupes removed : {before - len(df)}")

    # Drop duplicate primary keys – keep last
    if id_col and id_col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
        print(f"    pk dupes removed    : {before - len(df)}")

    return df


def clean_all(users, sessions, orders, order_items,
              payments, shipments, refunds, coupons, products):
    print("\n" + "="*60)
    print("STEP 2: Cleaning all tables")
    print("="*60)

    users = clean_df(users, "users", id_col="user_id",
                     date_cols=["signup_date", "created_at", "dob"])
    sessions = clean_df(sessions, "sessions", id_col="session_id",
                        date_cols=["session_start", "session_end", "session_date"])
    orders = clean_df(orders, "orders", id_col="order_id",
                      date_cols=["order_ts", "order_date", "created_at"])
    order_items = clean_df(order_items, "order_items",
                           date_cols=[])
    payments = clean_df(payments, "payments", id_col="payment_id",
                        date_cols=["payment_ts", "created_at"])
    shipments = clean_df(shipments, "shipments", id_col="shipment_id",
                         date_cols=["shipped_at", "delivered_at", "rto_at"])
    refunds = clean_df(refunds, "refunds", id_col="refund_id",
                       date_cols=["refund_date", "created_at"])
    coupons = clean_df(coupons, "coupons", id_col="coupon_id",
                       date_cols=["expiry_date"])
    products = clean_df(products, "products", id_col="product_id")

    # Coerce numerics
    for col in ["gross_amount", "discount_amount", "net_amount"]:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce")

    for col in ["unit_price", "quantity"]:
        if col in order_items.columns:
            order_items[col] = pd.to_numeric(order_items[col], errors="coerce")

    if "refund_amount" in refunds.columns:
        refunds["refund_amount"] = pd.to_numeric(refunds["refund_amount"], errors="coerce")

    return (users, sessions, orders, order_items,
            payments, shipments, refunds, coupons, products)


# =============================================================
# STEP 3 — DERIVED FEATURE FUNCTIONS
# =============================================================

def feat_payment(payments):
    """Count payment failures before first success per order."""
    p = payments.copy()
    p["status"] = p["status"].fillna("unknown")
    ts_col = "payment_ts" if "payment_ts" in p.columns else None

    if ts_col:
        p = p.sort_values(["order_id", ts_col])
    else:
        p = p.sort_values("order_id")

    def failures_before_success(grp):
        s = list(grp["status"])
        if "success" not in s:
            return len(s)
        return s.index("success")

    result = (p.groupby("order_id")
               .apply(failures_before_success)
               .reset_index())
    result.columns = ["order_id", "payment_fail_count_before_success"]
    return result


def feat_device_reuse(sessions):
    """How many distinct user_ids share the same device_id."""
    if "device_id" not in sessions.columns or "user_id" not in sessions.columns:
        return pd.DataFrame(columns=["session_id", "device_reuse_count",
                                     "channel", "device_type"])
    dev_users = (sessions.groupby("device_id")["user_id"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"user_id": "device_reuse_count"}))
    cols = ["session_id", "device_id"]
    for c in ["channel", "device_type"]:
        if c in sessions.columns:
            cols.append(c)
    s = sessions[cols].drop_duplicates("session_id")
    s = s.merge(dev_users, on="device_id", how="left")
    return s.drop(columns=["device_id"])


def feat_pincode_reuse(orders):
    """How many distinct user_ids share the same shipping pincode."""
    if "shipping_pincode" not in orders.columns or "user_id" not in orders.columns:
        return pd.DataFrame(columns=["shipping_pincode", "pincode_reuse_count"])
    pin = (orders.groupby("shipping_pincode")["user_id"]
           .nunique()
           .reset_index()
           .rename(columns={"user_id": "pincode_reuse_count"}))
    return pin


def feat_order_items(order_items):
    """item_count, total_qty, top_category per order."""
    agg = {}
    if "product_id" in order_items.columns:
        agg["item_count"] = ("product_id", "count")
    if "quantity" in order_items.columns:
        agg["total_qty"] = ("quantity", "sum")

    if agg:
        feat = order_items.groupby("order_id").agg(**agg).reset_index()
    else:
        feat = order_items[["order_id"]].drop_duplicates()

    if "category" in order_items.columns and "quantity" in order_items.columns:
        top = (order_items.groupby(["order_id", "category"])["quantity"]
               .sum()
               .reset_index()
               .sort_values("quantity", ascending=False)
               .drop_duplicates("order_id")[["order_id", "category"]]
               .rename(columns={"category": "top_category"}))
        feat = feat.merge(top, on="order_id", how="left")

    return feat


def feat_refund(refunds):
    """refund_amount and refund_approved per order."""
    r = refunds.copy()
    if "status" in r.columns:
        r = r[r["status"].str.contains("approved|success|done", na=False)]
    feat = (r.groupby("order_id")
             .agg(refund_amount=("refund_amount", "sum"))
             .reset_index())
    feat["refund_approved"] = 1
    return feat


def feat_rto(shipments):
    """RTO flag per order."""
    s = shipments.copy()
    if "status" in s.columns:
        s["rto_flag"] = s["status"].str.contains("rto", na=False).astype(int)
    elif "rto_at" in s.columns:
        s["rto_flag"] = s["rto_at"].notna().astype(int)
    else:
        s["rto_flag"] = 0
    return s.groupby("order_id")["rto_flag"].max().reset_index()


def feat_coupon(orders):
    """coupon_discount_pct, multi_coupon_user_flag."""
    o = orders[["order_id", "user_id", "gross_amount",
                "discount_amount", "coupon_id"]].copy() \
        if "coupon_id" in orders.columns \
        else orders[["order_id", "user_id", "gross_amount", "discount_amount"]].copy()

    # discount pct
    if "gross_amount" in o.columns and "discount_amount" in o.columns:
        o["coupon_discount_pct"] = np.where(
            o["gross_amount"] > 0,
            (o["discount_amount"].fillna(0) / o["gross_amount"] * 100).round(2),
            0
        )
    else:
        o["coupon_discount_pct"] = 0

    # multi coupon user flag
    if "coupon_id" in o.columns:
        coupon_orders = o[o["coupon_id"].notna()]
        user_cnt = (coupon_orders.groupby("user_id")["order_id"]
                    .count()
                    .reset_index()
                    .rename(columns={"order_id": "user_coupon_count"}))
        o = o.merge(user_cnt, on="user_id", how="left")
        o["user_coupon_count"] = o["user_coupon_count"].fillna(0)
        o["multi_coupon_user_flag"] = (o["user_coupon_count"] > 1).astype(int)
    else:
        o["multi_coupon_user_flag"] = 0

    return o[["order_id", "coupon_discount_pct", "multi_coupon_user_flag"]]


# =============================================================
# STEP 4 — RISK SCORING (10 signals, score 0-100)
# =============================================================
def compute_risk_score(df):
    """
    Weighted rule-based risk score per order (0–100).
    Signal weights:
      high_discount_flag            → 12
      multi_coupon_user_flag        → 10
      payment_fail_count (>=2)      → 15
      device_reuse_count (>=3)      → 12
      pincode_reuse_count (>=4)     → 10
      order_value_zscore (>2.5)     → 10
      qty_outlier_flag              →  8
      new_user_flag                 →  8
      cod_flag                      →  8
      rto_flag                      →  7
                                 TOTAL = 100
    """
    s = pd.Series(0.0, index=df.index)

    if "coupon_discount_pct" in df.columns:
        df["high_discount_flag"] = (df["coupon_discount_pct"] > 40).astype(int)
        s += df["high_discount_flag"] * 12

    if "multi_coupon_user_flag" in df.columns:
        s += df["multi_coupon_user_flag"] * 10

    if "payment_fail_count_before_success" in df.columns:
        s += (df["payment_fail_count_before_success"] >= 2).astype(int) * 15

    if "device_reuse_count" in df.columns:
        s += (df["device_reuse_count"] >= 3).astype(int) * 12

    if "pincode_reuse_count" in df.columns:
        s += (df["pincode_reuse_count"] >= 4).astype(int) * 10

    if "order_value_zscore_by_category" in df.columns:
        s += (df["order_value_zscore_by_category"].abs() > 2.5).astype(int) * 10

    if "qty_outlier_flag" in df.columns:
        s += df["qty_outlier_flag"] * 8

    if "new_user_flag" in df.columns:
        s += df["new_user_flag"] * 8

    if "cod_flag" in df.columns:
        s += df["cod_flag"] * 8

    if "rto_flag" in df.columns:
        s
