import os
import re
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

def normalize_employment(x: str) -> str:
    """Normaliza Estado Empregatício Informado em categorias macro."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    s = str(x).strip().lower()

    if s == "" or s in {"nan", "none"}:
        return "NA"

    # padrões comuns
    if re.search(r"aposent", s):
        return "APOSENTADO"
    if re.search(r"aut[ôo]nom|mei|microempre", s):
        return "AUTONOMO/MEI"
    if re.search(r"func.*pub|servidor|estatut|prefeit|govern|municip", s):
        return "FUNC_PUBLICO"
    if re.search(r"clt|assalariad|registrad", s):
        return "CLT"
    if re.search(r"desempreg|sem emprego|sem renda", s):
        return "DESEMPREGADO"
    if re.search(r"estud", s):
        return "ESTUDANTE"
    if re.search(r"empres|s[oó]cio|pj", s):
        return "EMPRESARIO/PJ"

    return "OUTROS"

def agg_kpis(df: pd.DataFrame) -> pd.Series:
    val = pd.to_numeric(df.get("[PAGO] Valor Emprestado", pd.Series([np.nan]*len(df))), errors="coerce")
    taxa = pd.to_numeric(df.get("Taxa de Juros", pd.Series([np.nan]*len(df))), errors="coerce")
    score = pd.to_numeric(df.get("Score Serasa", pd.Series([np.nan]*len(df))), errors="coerce")

    # Preencher PH3A - Empregado? se existir
    ph_emp = df.get("PH3A - Empregado?", pd.Series([pd.NA]*len(df)))
    if pd.api.types.is_bool_dtype(ph_emp):
        ph_emp_b = ph_emp.fillna(False)
    else:
        ph_emp_b = ph_emp.astype("string").str.strip().str.lower().map({"sim": True, "não": False, "nao": False}).fillna(False)

    return pd.Series({
        "n_propostas": df["proposal_id"].nunique(),
        "valor_emprestado_total": float(val.sum(skipna=True)),
        "valor_emprestado_medio": float(val.mean(skipna=True)),
        "taxa_media": float(taxa.mean(skipna=True)),
        "score_medio": float(score.mean(skipna=True)),
        "incident_rate_pct": float(df["is_incident"].mean() * 100),
        "rebate_share_pct": float(df["has_rebate"].mean() * 100),
        "ph3a_empregado_share_pct": float(ph_emp_b.mean() * 100),
    })

def main():
    mp = pd.read_parquet(DATA_PROCESSED / "master_proposals.parquet")
    mp = apply_all(mp)

    # Apenas safra definida
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

    # Normalizações de emprego
    if "Estado Empregatício Informado" in df.columns:
        df["employment_macro"] = df["Estado Empregatício Informado"].map(normalize_employment)
    else:
        df["employment_macro"] = "NA"

    # 1) KPIs gerais por mês
    by_month = (
        group_apply_kpis(df.groupby("base_month", dropna=False), agg_kpis)
        .reset_index()
        .rename(columns={"base_month": "mes"})
        .sort_values("mes")
    )
    by_month.to_csv(REPORTS_TABLES / f"emprego_kpis_{year}_out_nov_dez_geral.csv", index=False)

    # 2) KPIs por employment_macro (período inteiro)
    by_emp = (
        group_apply_kpis(df.groupby("employment_macro", dropna=False), agg_kpis)
        .reset_index()
        .sort_values(["n_propostas", "incident_rate_pct"], ascending=[False, False])
    )
    by_emp.to_csv(REPORTS_TABLES / f"emprego_kpis_{year}_out_nov_dez_por_macro.csv", index=False)

    # 3) KPIs por mês x employment_macro
    by_emp_month = (
        group_apply_kpis(df.groupby(["base_month", "employment_macro"], dropna=False), agg_kpis)
        .reset_index()
        .rename(columns={"base_month": "mes"})
        .sort_values(["mes", "n_propostas"], ascending=[True, False])
    )
    by_emp_month.to_csv(REPORTS_TABLES / f"emprego_kpis_{year}_out_nov_dez_por_macro_mes.csv", index=False)

    # 4) Drill-down por Política (macro emprego x política) — só se Política existir
    if "Política" in df.columns:
        by_emp_pol = (
            group_apply_kpis(df.groupby(["employment_macro", "Política"], dropna=False), agg_kpis)
            .reset_index()
            .sort_values(["employment_macro", "n_propostas"], ascending=[True, False])
        )
        by_emp_pol.to_csv(REPORTS_TABLES / f"emprego_kpis_{year}_out_nov_dez_por_macro_politica.csv", index=False)

    # 5) Sinal de “prioridade” (volume x incidentes)
    pri = by_emp.copy()
    pri["priority_score"] = pri["n_propostas"] * (pri["incident_rate_pct"] / 100.0)
    pri = pri.sort_values("priority_score", ascending=False)
    pri.to_csv(REPORTS_TABLES / f"emprego_prioridade_{year}_out_nov_dez.csv", index=False)

    print(f"✅ Empregos concluído para {year} (Out/Nov/Dez).")
    print("Tabelas salvas em:", REPORTS_TABLES)

if __name__ == "__main__":
    main()