import os
import numpy as np
import pandas as pd
from pathlib import Path

from src._02_metrics import apply_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_TABLES = PROJECT_ROOT / "reports" / "tables"

REPORTS_TABLES.mkdir(parents=True, exist_ok=True)

def group_apply_kpis(gb, func):
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)

def pick_latest_year_with_oct_nov_dec(base_month: pd.Series) -> int | None:
    bm = base_month.dropna().astype(str)
    years = sorted({int(x.split("-")[0]) for x in bm if "-" in x}, reverse=True)
    for y in years:
        needed = {f"{y}-10", f"{y}-11", f"{y}-12"}
        if needed.issubset(set(bm)):
            return y
    return years[0] if years else None

def agg_kpis(df: pd.DataFrame) -> pd.Series:
    val = pd.to_numeric(df.get("[PAGO] Valor Emprestado", pd.Series([np.nan]*len(df))), errors="coerce")
    taxa = pd.to_numeric(df.get("Taxa de Juros", pd.Series([np.nan]*len(df))), errors="coerce")
    score = pd.to_numeric(df.get("Score Serasa", pd.Series([np.nan]*len(df))), errors="coerce")

    scr_err = df.get("Consulta scr erro?", pd.Series([pd.NA]*len(df)))
    srm_err = df.get("consulta srm filme erro?", pd.Series([pd.NA]*len(df)))

    def _norm_bool(x):
        if pd.api.types.is_bool_dtype(x):
            return x
        return x.astype("string").str.strip().str.lower().map({"sim": True, "não": False, "nao": False})

    scr_b = _norm_bool(scr_err).fillna(False) if scr_err is not None else pd.Series([False]*len(df))
    srm_b = _norm_bool(srm_err).fillna(False) if srm_err is not None else pd.Series([False]*len(df))

    return pd.Series({
        "n_propostas": df["proposal_id"].nunique(),
        "valor_emprestado_total": float(val.sum(skipna=True)),
        "valor_emprestado_medio": float(val.mean(skipna=True)),
        "taxa_media": float(taxa.mean(skipna=True)),
        "score_medio": float(score.mean(skipna=True)),
        "incident_rate_pct": float(df["is_incident"].mean() * 100),
        "scr_erro_rate_pct": float(scr_b.mean() * 100),
        "srm_erro_rate_pct": float(srm_b.mean() * 100),
        "rebate_share_pct": float(df["has_rebate"].mean() * 100),
    })

def main():
    mp = pd.read_parquet(DATA_PROCESSED / "master_proposals.parquet")
    mp = apply_all(mp)

    mp = mp[mp["base_month"].notna()].copy()

    year_override = os.getenv("YEAR_OVERRIDE")
    if year_override:
        year = int(year_override)
    else:
        year = pick_latest_year_with_oct_nov_dec(mp["base_month"])
        if year is None:
            raise ValueError("Não encontrei base_month válido.")

    months = [f"{year}-10", f"{year}-11", f"{year}-12"]
    df = mp[mp["base_month"].isin(months)].copy()
    if df.empty:
        raise ValueError(f"Nenhuma linha encontrada para {months}. Use YEAR_OVERRIDE ou verifique base_month.")

    if "Clinica" not in df.columns:
        raise ValueError("Coluna 'Clinica' não encontrada em master_proposals.")

    # 1) KPIs gerais por mês
    kpi_by_month = (
        group_apply_kpis(df.groupby("base_month", dropna=False), agg_kpis)
          .reset_index()
          .rename(columns={"base_month": "mes"})
          .sort_values("mes")
    )
    kpi_by_month.to_csv(REPORTS_TABLES / f"clinicas_kpis_{year}_out_nov_dez_geral.csv", index=False)

    # 2) KPIs por clínica (período inteiro)
    kpi_by_clinic = (
        group_apply_kpis(df.groupby("Clinica", dropna=False), agg_kpis)
          .reset_index()
          .sort_values(["n_propostas", "incident_rate_pct"], ascending=[False, False])
    )
    kpi_by_clinic.to_csv(REPORTS_TABLES / f"clinicas_kpis_{year}_out_nov_dez_por_clinica.csv", index=False)

    # 3) KPIs por clínica por mês
    kpi_by_clinic_month = (
        group_apply_kpis(df.groupby(["base_month", "Clinica"], dropna=False), agg_kpis)
          .reset_index()
          .rename(columns={"base_month": "mes"})
          .sort_values(["mes", "n_propostas"], ascending=[True, False])
    )
    kpi_by_clinic_month.to_csv(REPORTS_TABLES / f"clinicas_kpis_{year}_out_nov_dez_por_clinica_mes.csv", index=False)

    # 4) Quadrantes/priorização (volume x incident_rate)
    quad = kpi_by_clinic.copy()
    quad["priority_score"] = quad["n_propostas"] * (quad["incident_rate_pct"] / 100.0)
    quad = quad.sort_values("priority_score", ascending=False)
    quad.to_csv(REPORTS_TABLES / f"clinicas_quadrantes_{year}_out_nov_dez.csv", index=False)

    print(f"✅ Clínicas + Incidentes concluído para {year} (Out/Nov/Dez).")
    print("Tabelas salvas em:", REPORTS_TABLES)

if __name__ == "__main__":
    main()