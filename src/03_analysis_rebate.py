import os
import numpy as np
import pandas as pd
from pathlib import Path

from src._02_metrics import apply_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_TABLES = PROJECT_ROOT / "reports" / "tables"
REPORTS_TABLES.mkdir(parents=True, exist_ok=True)

def pick_latest_year_with_oct_nov_dec(base_month: pd.Series) -> int | None:
    bm = base_month.dropna().astype(str)
    years = sorted({int(x.split("-")[0]) for x in bm if "-" in x}, reverse=True)
    for y in years:
        needed = {f"{y}-10", f"{y}-11", f"{y}-12"}
        if needed.issubset(set(bm)):
            return y
    return years[0] if years else None

def group_apply(gb, func):
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)

def build_paid_p1_flag(master_installments: pd.DataFrame) -> pd.DataFrame:
    mi = master_installments.copy()
    mi["proposal_id"] = mi["proposal_id"].astype("string")

    if "parcela_n" not in mi.columns:
        return pd.DataFrame({"proposal_id": mi["proposal_id"].unique(), "paid_p1": False})

    if "Status" in mi.columns:
        paid_status = mi["Status"].astype("string").str.strip().str.lower().eq("pago")
    else:
        paid_status = pd.Series([False]*len(mi))

    p1 = (
        mi[(mi["parcela_n"] == 1) & paid_status]
        .groupby("proposal_id")
        .size()
        .rename("paid_p1_count")
        .reset_index()
    )
    p1["paid_p1"] = p1["paid_p1_count"] > 0
    return p1[["proposal_id", "paid_p1"]]

def safe_num(df, col):
    return pd.to_numeric(df.get(col, pd.Series([np.nan]*len(df))), errors="coerce")

def agg_rebate_kpis(df: pd.DataFrame) -> pd.Series:
    val = safe_num(df, "[PAGO] Valor Emprestado")
    taxa = safe_num(df, "Taxa de Juros")
    taxa_rb = safe_num(df, "Taxa de Juros Rebate")
    spread = safe_num(df, "rebate_spread")

    return pd.Series({
        "n_propostas": df["proposal_id"].nunique(),
        "valor_emprestado_total": float(val.sum(skipna=True)),
        "rebate_share_pct": float(df["has_rebate"].mean() * 100),
        "rebate_spread_medio": float(spread.mean(skipna=True)),
        "taxa_media": float(taxa.mean(skipna=True)),
        "taxa_rebate_media": float(taxa_rb.mean(skipna=True)),
        "paid_p1_rate_pct": float(df["paid_p1"].mean() * 100),
        "incident_rate_pct": float(df["is_incident"].mean() * 100),
    })

def main():
    mp = pd.read_parquet(DATA_PROCESSED / "master_proposals.parquet")
    mi = pd.read_parquet(DATA_PROCESSED / "master_installments.parquet")

    mp = apply_all(mp)
    mp = mp[mp["base_month"].notna()].copy()

    year_override = os.getenv("YEAR_OVERRIDE")
    if year_override:
        year = int(year_override)
    else:
        year = pick_latest_year_with_oct_nov_dec(mp["base_month"])
        if year is None:
            raise ValueError("Não encontrei ano válido com Out/Nov/Dez em base_month.")

    months = [f"{year}-10", f"{year}-11", f"{year}-12"]
    df = mp[mp["base_month"].isin(months)].copy()
    if df.empty:
        raise ValueError(f"Nenhuma linha encontrada para {months}.")

    # paid_p1
    p1 = build_paid_p1_flag(mi)
    df = df.merge(p1, on="proposal_id", how="left")
    df["paid_p1"] = df["paid_p1"].fillna(False)

    # --------------------------
    # 1) Rebate por mês (Out/Nov/Dez)
    # --------------------------
    by_month = (
        group_apply(df.groupby("base_month", dropna=False), agg_rebate_kpis)
        .reset_index()
        .rename(columns={"base_month": "mes"})
        .sort_values("mes")
    )
    by_month.to_csv(REPORTS_TABLES / f"rebate_{year}_out_nov_dez_mes.csv", index=False)

    # --------------------------
    # 2) Rebate por dimensões (período)
    # --------------------------
    dims = ["Política", "Política Rebate", "Clinica", "score_bucket", "taxa_bucket"]
    for dim in dims:
        if dim not in df.columns:
            continue
        tmp = (
            group_apply(df.groupby(dim, dropna=False), agg_rebate_kpis)
            .reset_index()
            .sort_values(["n_propostas", "rebate_share_pct"], ascending=[False, False])
        )
        tmp.to_csv(REPORTS_TABLES / f"rebate_{year}_out_nov_dez_por_{dim.replace(' ', '_')}.csv", index=False)

    # --------------------------
    # 3) “Efeito” rebate vs sem rebate controlando por grupos (quase-causal)
    # grupo = Política x score_bucket (fallback: só score_bucket)
    # --------------------------
    group_cols = []
    if "Política" in df.columns:
        group_cols.append("Política")
    if "score_bucket" in df.columns:
        group_cols.append("score_bucket")

    if not group_cols:
        group_cols = ["base_month"]  # fallback mínimo

    # taxa de paid_p1 por has_rebate dentro de cada grupo
    effect = (
        df.groupby(group_cols + ["has_rebate"], dropna=False)
          .agg(
              n_propostas=("proposal_id", "nunique"),
              paid_p1_rate=("paid_p1", "mean"),
              incident_rate=("is_incident", "mean"),
              taxa_media=("Taxa de Juros", lambda s: pd.to_numeric(s, errors="coerce").mean()),
              spread_medio=("rebate_spread", lambda s: pd.to_numeric(s, errors="coerce").mean()),
          )
          .reset_index()
    )
    # para facilitar leitura em %
    effect["paid_p1_rate_pct"] = (effect["paid_p1_rate"] * 100).round(2)
    effect["incident_rate_pct"] = (effect["incident_rate"] * 100).round(2)

    effect.to_csv(REPORTS_TABLES / f"rebate_{year}_out_nov_dez_effect_by_{'_'.join(group_cols)}.csv", index=False)

    print(f"✅ Rebate concluído para {year} (Out/Nov/Dez).")
    print("Tabelas em:", REPORTS_TABLES)

if __name__ == "__main__":
    main()