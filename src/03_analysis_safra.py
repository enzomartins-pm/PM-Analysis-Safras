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

def agg_kpis(df: pd.DataFrame) -> pd.Series:
    val = pd.to_numeric(df.get("[PAGO] Valor Emprestado", pd.Series([np.nan]*len(df))), errors="coerce")
    taxa = pd.to_numeric(df.get("Taxa de Juros", pd.Series([np.nan]*len(df))), errors="coerce")
    score = pd.to_numeric(df.get("Score Serasa", pd.Series([np.nan]*len(df))), errors="coerce")
    rebate_spread = pd.to_numeric(df.get("rebate_spread", pd.Series([np.nan]*len(df))), errors="coerce")

    return pd.Series({
        "n_propostas": df["proposal_id"].nunique(),
        "valor_emprestado_total": float(val.sum(skipna=True)),
        "valor_emprestado_medio": float(val.mean(skipna=True)),
        "taxa_media": float(taxa.mean(skipna=True)),
        "score_medio": float(score.mean(skipna=True)),
        "incident_rate_pct": float(df["is_incident"].mean() * 100),
        "rebate_share_pct": float(df["has_rebate"].mean() * 100),
        "rebate_spread_medio": float(rebate_spread.mean(skipna=True)),
    })

def main():
    mp = pd.read_parquet(DATA_PROCESSED / "master_proposals.parquet")
    mp = apply_all(mp)

    # safra só faz sentido com base_month (base = [PAGO] Data do Pagamento)
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
        raise ValueError(f"Nenhuma linha encontrada para {months}. Use YEAR_OVERRIDE ou verifique base_month.")

    # 1) KPIs por mês (Out/Nov/Dez)
    kpi_by_month = (
        group_apply(df.groupby("base_month", dropna=False), agg_kpis)
        .reset_index()
        .rename(columns={"base_month": "mes"})
        .sort_values("mes")
    )
    kpi_by_month.to_csv(REPORTS_TABLES / f"safra_kpis_{year}_out_nov_dez.csv", index=False)

    # 2) Quebras principais
    dims = []
    for dim in ["Política", "Política Rebate", "Clinica", "Estado Empregatício Informado", "Classe de Risco b2e", "score_bucket", "taxa_bucket"]:
        if dim in df.columns:
            dims.append(dim)

    for dim in dims:
        tmp = (
            group_apply(df.groupby(["base_month", dim], dropna=False), agg_kpis)
            .reset_index()
            .rename(columns={"base_month": "mes", dim: "dim_value"})
            .sort_values(["mes", "n_propostas"], ascending=[True, False])
        )
        tmp.to_csv(REPORTS_TABLES / f"safra_{year}_out_nov_dez_by_{dim.replace(' ', '_')}.csv", index=False)

    print(f"✅ Safra Out/Nov/Dez concluída para o ano {year}.")
    print("Tabela:", REPORTS_TABLES / f"safra_kpis_{year}_out_nov_dez.csv")

if __name__ == "__main__":
    main()