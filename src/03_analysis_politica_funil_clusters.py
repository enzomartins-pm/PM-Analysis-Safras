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

def safe_num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan]*len(df))

def build_paid_p1_flag(master_installments: pd.DataFrame) -> pd.DataFrame:
    """1 linha por proposal_id com flag de parcela 1 paga (Status == Pago)."""
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

def agg_policy_kpis(df: pd.DataFrame) -> pd.Series:
    val = safe_num(df, "[PAGO] Valor Emprestado")
    taxa = safe_num(df, "Taxa de Juros")
    score = safe_num(df, "Score Serasa")
    rebate_spread = safe_num(df, "rebate_spread")

    return pd.Series({
        "n_propostas": df["proposal_id"].nunique(),
        "valor_emprestado_total": float(val.sum(skipna=True)),
        "valor_emprestado_medio": float(val.mean(skipna=True)),
        "taxa_media": float(taxa.mean(skipna=True)),
        "score_medio": float(score.mean(skipna=True)),
        "incident_rate_pct": float(df["is_incident"].mean() * 100),
        "rebate_share_pct": float(df["has_rebate"].mean() * 100),
        "rebate_spread_medio": float(rebate_spread.mean(skipna=True)),
        "paid_p1_rate_pct": float(df["paid_p1"].mean() * 100),
    })

def group_apply(gb, func):
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)

def pick_cluster_features(df: pd.DataFrame) -> list[str]:
    """Escolhe features numéricas com boa disponibilidade (sem depender de colunas específicas)."""
    candidates = [
        "Score Serasa",
        "Taxa de Juros",
        "[PAGO] Valor Emprestado",
        "Idade Cliente",
        "PH3A - Renda Familiar",
        "PH3A - Renda Presumida",
        "SCR - Total Vencido",
        "SCR - Total Prejuízo",
        "SCR - Qtd Insituições",
        "SCR - Qtd. Operações",
        "SCR - Carteira de Crédito",
        "SCR - Creditos a Liberar",
    ]
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        # fallback mínimo: pega qualquer numérico disponível
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return num_cols[:8]
    return existing

def _safe_qcut_labels(series: pd.Series, q: int, prefix: str) -> pd.Series:
    """
    Faz qcut de forma segura:
    - remove NaNs
    - se poucos valores únicos, reduz q automaticamente
    - cria labels dinamicamente conforme o nº real de bins
    - se não der pra cortar, retorna 'NA'
    """
    s = pd.to_numeric(series, errors="coerce")

    # se não tem dado suficiente
    if s.notna().sum() < 10:
        return pd.Series(["NA"] * len(s), index=s.index, dtype="string")

    # rank para lidar com empates
    r = s.rank(method="first")

    # quantos bins é possível? (não faz sentido pedir 3 bins com 2 valores únicos)
    nunique = s.dropna().nunique()
    q_eff = int(min(q, nunique))
    if q_eff < 2:
        return pd.Series(["NA"] * len(s), index=s.index, dtype="string")

    # tenta qcut sem labels primeiro pra ver quantos bins sobram após duplicates drop
    try:
        cats = pd.qcut(r, q=q_eff, duplicates="drop")
    except Exception:
        return pd.Series(["NA"] * len(s), index=s.index, dtype="string")

    # número real de bins
    n_bins = len(cats.cat.categories)
    if n_bins < 2:
        return pd.Series(["NA"] * len(s), index=s.index, dtype="string")

    labels = [f"{prefix}{i+1}" for i in range(n_bins)]
    # refaz com labels corretos
    try:
        return pd.qcut(r, q=q_eff, labels=labels, duplicates="drop").astype("string")
    except Exception:
        # fallback: sem labels
        return cats.astype("string")


def run_clusters(df: pd.DataFrame, n_clusters: int = 6) -> tuple[pd.DataFrame, str]:
    """
    Retorna df com coluna 'cluster' e método usado.
    - tenta sklearn KMeans
    - fallback robusto: clusters por quantis em score/taxa/valor com qcut seguro
    """
    features = pick_cluster_features(df)
    X = df[features].copy()

    # converter para numérico
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # imputação simples (mediana)
    X = X.fillna(X.median(numeric_only=True))

    # padronização simples (z-score)
    X_std = (X - X.mean()) / (X.std(ddof=0).replace(0, 1))

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(X_std.values)
        out = df.copy()
        out["cluster"] = labels.astype(int)
        return out, f"kmeans(n_clusters={n_clusters})"
    except Exception:
        # fallback por quantis: score/taxa/valor (com bins seguros)
        out = df.copy()
        score = pd.to_numeric(out.get("Score Serasa", pd.Series([np.nan]*len(out))), errors="coerce")
        taxa = pd.to_numeric(out.get("Taxa de Juros", pd.Series([np.nan]*len(out))), errors="coerce")
        val = pd.to_numeric(out.get("[PAGO] Valor Emprestado", pd.Series([np.nan]*len(out))), errors="coerce")

        # normaliza taxa se parecer % (ex.: 12.5)
        if taxa.dropna().shape[0] > 0 and (taxa.dropna() > 1).mean() > 0.8:
            taxa = taxa / 100.0

        s_q = _safe_qcut_labels(score, q=3, prefix="S")
        t_q = _safe_qcut_labels(taxa, q=3, prefix="T")
        v_q = _safe_qcut_labels(val, q=3, prefix="V")

        out["cluster"] = (
            s_q.fillna("SNA") + "_" +
            t_q.fillna("TNA") + "_" +
            v_q.fillna("VNA")
        ).astype("string")

        return out, "quantile_clusters_safe(score,taxa,valor)"


def main():
    # Load masters
    mp = pd.read_parquet(DATA_PROCESSED / "master_proposals.parquet")
    mi = pd.read_parquet(DATA_PROCESSED / "master_installments.parquet")

    mp = apply_all(mp)
    mp = mp[mp["base_month"].notna()].copy()

    # Year / months
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

    # Funil MVP: paid_p1 (parcela 1 paga) usando master_installments
    p1 = build_paid_p1_flag(mi)
    df = df.merge(p1, on="proposal_id", how="left")
    df["paid_p1"] = df["paid_p1"].fillna(False).astype(bool)

    # --------------------------
    # 1) Política + Funil (geral por mês)
    # --------------------------
    overall_by_month = (
        group_apply(df.groupby("base_month", dropna=False), agg_policy_kpis)
        .reset_index()
        .rename(columns={"base_month": "mes"})
        .sort_values("mes")
    )
    overall_by_month.to_csv(REPORTS_TABLES / f"politica_funil_{year}_out_nov_dez_geral_mes.csv", index=False)

    # 2) Política + Funil por Política x mês
    if "Política" in df.columns:
        by_policy_month = (
            group_apply(df.groupby(["base_month", "Política"], dropna=False), agg_policy_kpis)
            .reset_index()
            .rename(columns={"base_month": "mes"})
            .sort_values(["mes", "n_propostas"], ascending=[True, False])
        )
        by_policy_month.to_csv(REPORTS_TABLES / f"politica_funil_{year}_out_nov_dez_politica_mes.csv", index=False)

        by_policy = (
            group_apply(df.groupby(["Política"], dropna=False), agg_policy_kpis)
            .reset_index()
            .sort_values(["n_propostas", "paid_p1_rate_pct"], ascending=[False, False])
        )
        by_policy.to_csv(REPORTS_TABLES / f"politica_funil_{year}_out_nov_dez_politica_periodo.csv", index=False)

    # --------------------------
    # 3) Clusters (no período Out/Nov/Dez)
    # --------------------------
    clustered, method = run_clusters(df, n_clusters=int(os.getenv("N_CLUSTERS", "6")))

    # KPIs por cluster
    cluster_kpis = (
        group_apply(clustered.groupby("cluster", dropna=False), agg_policy_kpis)
        .reset_index()
        .sort_values(["n_propostas", "incident_rate_pct"], ascending=[False, False])
    )
    cluster_kpis["cluster_method"] = method
    cluster_kpis.to_csv(REPORTS_TABLES / f"clusters_{year}_out_nov_dez_kpis.csv", index=False)

    # Mix de política por cluster (se existir)
    if "Política" in clustered.columns:
        mix = (
            clustered.groupby(["cluster", "Política"], dropna=False)["proposal_id"]
            .nunique()
            .reset_index(name="n_propostas")
            .sort_values(["cluster", "n_propostas"], ascending=[True, False])
        )
        mix.to_csv(REPORTS_TABLES / f"clusters_{year}_out_nov_dez_mix_politica.csv", index=False)

    # (Opcional) salvar assignments (pode ficar grande)
    save_assign = os.getenv("SAVE_CLUSTER_ASSIGNMENTS", "0") == "1"
    if save_assign:
        clustered[["proposal_id", "base_month", "Política", "Clinica", "cluster", "paid_p1", "is_incident", "has_rebate"]].to_parquet(
            DATA_PROCESSED / f"cluster_assignments_{year}_out_nov_dez.parquet",
            index=False
        )

    print(f"✅ Política + Funil + Clusters concluído para {year} (Out/Nov/Dez).")
    print("Método de cluster:", method)
    print("Tabelas em:", REPORTS_TABLES)

if __name__ == "__main__":
    main()