import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Bootstrap paths (import src)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src._02_metrics import apply_all


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="PM Analytics — Análise por Período",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MASTER_PROPOSALS_PATH = DATA_PROCESSED / "master_proposals.parquet"
MASTER_INSTALLMENTS_PATH = DATA_PROCESSED / "master_installments.parquet"


# -----------------------------
# Helpers
# -----------------------------
def safe_qcut_rank(series: pd.Series, q: int, prefix: str) -> pd.Series:
    """
    qcut robusto: tenta criar q grupos por quantis na série rankeada.
    Se o pandas reduzir bins (duplicates drop) ou faltar variedade, ajusta labels.
    """
    s = pd.to_numeric(series, errors="coerce")
    r = s.rank(method="first").dropna()

    if r.nunique() < 2:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="string")

    try:
        cats = pd.qcut(series.rank(method="first"), q=q, duplicates="drop")
        n_bins = cats.cat.categories.size
        if n_bins < 1:
            return pd.Series([pd.NA] * len(series), index=series.index, dtype="string")

        labels = [f"{prefix}{i+1}" for i in range(n_bins)]
        cats2 = pd.qcut(series.rank(method="first"), q=q, labels=labels, duplicates="drop")
        return cats2.astype("string")
    except Exception:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="string")


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sort_base_months(months: List[str]) -> List[str]:
    """
    Ordena cronologicamente strings YYYY-MM.
    Se houver algum formato inesperado, cai para sort lexicográfico.
    """
    m = pd.Series(months).dropna().astype(str)
    if m.empty:
        return []
    try:
        p = pd.PeriodIndex(m, freq="M")
        return [str(x) for x in p.sort_values().astype(str)]
    except Exception:
        return sorted(m.unique().tolist())


def normalize_employment(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return "NA"
    if "aposent" in s:
        return "APOSENTADO"
    if "autôn" in s or "auton" in s or "mei" in s or "microempre" in s:
        return "AUTONOMO/MEI"
    if ("func" in s and "pub" in s) or "servidor" in s or "estatut" in s:
        return "FUNC_PUBLICO"
    if "clt" in s or "registrad" in s or "assalariad" in s:
        return "CLT"
    if "desempreg" in s or "sem emprego" in s:
        return "DESEMPREGADO"
    if "estud" in s:
        return "ESTUDANTE"
    if "empres" in s or "sócio" in s or "socio" in s or "pj" in s:
        return "EMPRESARIO/PJ"
    return "OUTROS"


def build_paid_p1_flag(mi: pd.DataFrame) -> pd.DataFrame:
    mi = mi.copy()
    mi["proposal_id"] = mi["proposal_id"].astype("string")
    if "parcela_n" not in mi.columns:
        return pd.DataFrame({"proposal_id": mi["proposal_id"].unique(), "paid_p1": False})

    if "Status" in mi.columns:
        paid_status = mi["Status"].astype("string").str.strip().str.lower().eq("pago")
    else:
        paid_status = pd.Series([False] * len(mi))

    p1 = (
        mi[(mi["parcela_n"] == 1) & paid_status]
        .groupby("proposal_id")
        .size()
        .rename("paid_p1_count")
        .reset_index()
    )
    p1["paid_p1"] = p1["paid_p1_count"] > 0
    return p1[["proposal_id", "paid_p1"]]


def ensure_cluster(df: pd.DataFrame, n_clusters: int = 6) -> Tuple[pd.DataFrame, str]:
    """Tenta KMeans, fallback por quantis (score/taxa/valor)."""
    out = df.copy()

    features = [c for c in ["Score Serasa", "Taxa de Juros", "[PAGO] Valor Emprestado", "Idade Cliente"] if c in out.columns]
    if not features:
        out["cluster"] = "NA"
        return out, "none"

    X = out[features].copy()
    for c in features:
        X[c] = safe_num(X[c])

    X = X.fillna(X.median(numeric_only=True))
    X_std = (X - X.mean()) / (X.std(ddof=0).replace(0, 1))

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        out["cluster"] = km.fit_predict(X_std.values).astype(int)
        return out, f"kmeans({n_clusters})"
    except Exception:
        score = safe_num(out.get("Score Serasa", pd.Series([np.nan] * len(out))))
        taxa = safe_num(out.get("Taxa de Juros", pd.Series([np.nan] * len(out))))
        val = safe_num(out.get("[PAGO] Valor Emprestado", pd.Series([np.nan] * len(out))))

        # se taxa parece estar em % (ex. 12.5), converte para decimal
        if taxa.dropna().shape[0] > 0 and (taxa.dropna() > 1).mean() > 0.8:
            taxa = taxa / 100.0

        s_q = safe_qcut_rank(score, q=3, prefix="S")
        t_q = safe_qcut_rank(taxa, q=3, prefix="T")
        v_q = safe_qcut_rank(val, q=3, prefix="V")
        out["cluster"] = (
            s_q.astype("string").fillna("SNA") + "_" +
            t_q.astype("string").fillna("TNA") + "_" +
            v_q.astype("string").fillna("VNA")
        )
        return out, "quantiles(score,taxa,valor)"


def kpi_row(df: pd.DataFrame) -> Dict[str, Any]:
    val = safe_num(df.get("[PAGO] Valor Emprestado", pd.Series([np.nan] * len(df))))
    taxa = safe_num(df.get("Taxa de Juros", pd.Series([np.nan] * len(df))))
    score = safe_num(df.get("Score Serasa", pd.Series([np.nan] * len(df))))
    rebate_spread = safe_num(df.get("rebate_spread", pd.Series([np.nan] * len(df))))

    return {
        "propostas": int(df["proposal_id"].nunique()) if "proposal_id" in df.columns else int(len(df)),
        "valor_total": float(val.sum(skipna=True)),
        "ticket_medio": float(val.mean(skipna=True)),
        "taxa_media": float(taxa.mean(skipna=True)),
        "score_medio": float(score.mean(skipna=True)),
        "incident_rate_pct": float(df.get("is_incident", pd.Series([False] * len(df))).mean() * 100),
        "rebate_share_pct": float(df.get("has_rebate", pd.Series([False] * len(df))).mean() * 100),
        "rebate_spread_medio": float(rebate_spread.mean(skipna=True)),
        "paid_p1_rate_pct": float(df.get("paid_p1", pd.Series([False] * len(df))).mean() * 100),
    }


def money_br(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# -----------------------------
# Loaders (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_master() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not MASTER_PROPOSALS_PATH.exists():
        raise FileNotFoundError(f"Não achei {MASTER_PROPOSALS_PATH}")
    if not MASTER_INSTALLMENTS_PATH.exists():
        raise FileNotFoundError(f"Não achei {MASTER_INSTALLMENTS_PATH}")

    mp = pd.read_parquet(MASTER_PROPOSALS_PATH)
    mi = pd.read_parquet(MASTER_INSTALLMENTS_PATH)

    mp = apply_all(mp)
    mp = mp[mp["base_month"].notna()].copy()

    # emprego macro
    if "Estado Empregatício Informado" in mp.columns:
        mp["employment_macro"] = mp["Estado Empregatício Informado"].map(normalize_employment)
    else:
        mp["employment_macro"] = "NA"

    # paid_p1
    p1 = build_paid_p1_flag(mi)
    mp = mp.merge(p1, on="proposal_id", how="left")
    mp["paid_p1"] = mp["paid_p1"].fillna(False)

    return mp, mi


# -----------------------------
# Chat “tool”: compute table from request
# -----------------------------
ALLOWED_METRICS = {
    "propostas": ("proposal_id", "nunique"),
    "valor_total": ("[PAGO] Valor Emprestado", "sum"),
    "ticket_medio": ("[PAGO] Valor Emprestado", "mean"),
    "taxa_media": ("Taxa de Juros", "mean"),
    "score_medio": ("Score Serasa", "mean"),
    "incident_rate": ("is_incident", "mean"),
    "rebate_share": ("has_rebate", "mean"),
    "paid_p1_rate": ("paid_p1", "mean"),
    "rebate_spread_medio": ("rebate_spread", "mean"),
}

ALLOWED_DIMS = ["base_month", "Política", "Clinica", "employment_macro", "score_bucket", "taxa_bucket", "cluster"]


def compute_table(df: pd.DataFrame, metric: str, groupby: List[str], top_n: int = 15) -> pd.DataFrame:
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Métrica não permitida: {metric}")
    for g in groupby:
        if g not in ALLOWED_DIMS or g not in df.columns:
            raise ValueError(f"Dimensão inválida/indisponível: {g}")

    col, agg = ALLOWED_METRICS[metric]
    d = df.copy()

    if col in d.columns and col not in ["proposal_id", "is_incident", "has_rebate", "paid_p1"]:
        d[col] = safe_num(d[col])

    if agg == "nunique":
        out = d.groupby(groupby, dropna=False)[col].nunique().reset_index(name=metric)
    elif agg == "sum":
        out = d.groupby(groupby, dropna=False)[col].sum().reset_index(name=metric)
    elif agg == "mean":
        out = d.groupby(groupby, dropna=False)[col].mean().reset_index(name=metric)
    else:
        raise ValueError("Agg não suportado")

    if metric in ["incident_rate", "rebate_share", "paid_p1_rate"]:
        out[metric] = (out[metric] * 100).round(2)

    out = out.sort_values(metric, ascending=False).head(top_n)
    return out


# -----------------------------
# Chat OFFLINE helpers
# -----------------------------
MONTH_WORDS = {
    "janeiro": "-01", "jan": "-01",
    "fevereiro": "-02", "fev": "-02",
    "março": "-03", "marco": "-03", "mar": "-03",
    "abril": "-04", "abr": "-04",
    "maio": "-05", "mai": "-05",
    "junho": "-06", "jun": "-06",
    "julho": "-07", "jul": "-07",
    "agosto": "-08", "ago": "-08",
    "setembro": "-09", "set": "-09",
    "outubro": "-10", "out": "-10",
    "novembro": "-11", "nov": "-11",
    "dezembro": "-12", "dez": "-12",
}


def _pick_months_from_text(text: str, available: List[str]) -> list[str]:
    """
    Detecta meses no texto:
    - explícito: YYYY-MM
    - por nome: jan/fev/.../dez => aplica para TODOS os anos disponíveis no recorte atual
    """
    t = (text or "").lower()

    # 1) captura YYYY-MM explícito
    explicit = re.findall(r"\b(20\d{2}-\d{2})\b", t)
    explicit = [m for m in explicit if m in set(available)]

    # 2) captura nomes de meses
    suffixes = []
    for w, suf in MONTH_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            suffixes.append(suf)

    named = []
    for suf in set(suffixes):
        named.extend([m for m in available if str(m).endswith(suf)])

    return sorted(list(set(explicit + named)))


def offline_insights(df: pd.DataFrame) -> str:
    """Resumo executivo offline com base no recorte atual."""
    k = kpi_row(df)
    lines = []
    lines.append("### Snapshot do recorte atual")
    lines.append(
        f"- Propostas: **{k['propostas']:,}** | "
        f"Valor total: **{money_br(k['valor_total'])}** | "
        f"Ticket médio: **{money_br(k['ticket_medio'])}**"
    )
    lines.append(
        f"- Paid P1: **{k['paid_p1_rate_pct']:.2f}%** | "
        f"Incidentes: **{k['incident_rate_pct']:.2f}%** | "
        f"Rebate share: **{k['rebate_share_pct']:.2f}%**"
    )

    if "Política" in df.columns and df["Política"].notna().any():
        pol = (
            df.groupby("Política", dropna=False)
            .agg(
                n=("proposal_id", "nunique"),
                paid=("paid_p1", "mean"),
                inc=("is_incident", "mean"),
                taxa=("Taxa de Juros", lambda s: safe_num(s).mean()),
            )
            .reset_index()
        )
        pol = pol[pol["n"] >= 30].copy()
        if not pol.empty:
            pol["paid_pct"] = pol["paid"] * 100
            pol["inc_pct"] = pol["inc"] * 100
            pol["trade_score"] = (pol["paid_pct"] - pol["inc_pct"]) * np.log1p(pol["n"])
            best = pol.sort_values("trade_score", ascending=False).head(5)

            lines.append("\n### Top políticas (trade-off Paid P1 ↑ vs Incidentes ↓, min 30 props)")
            for _, r in best.iterrows():
                lines.append(
                    f"- **{r['Política']}** | N={int(r['n'])} | "
                    f"PaidP1={r['paid_pct']:.1f}% | Inc={r['inc_pct']:.1f}% | Taxa={r['taxa']:.4f}"
                )

    if "Clinica" in df.columns and df["Clinica"].notna().any():
        cli = (
            df.groupby("Clinica", dropna=False)
            .agg(n=("proposal_id", "nunique"), inc=("is_incident", "mean"))
            .reset_index()
        )
        cli = cli[cli["n"] >= 30].copy()
        if not cli.empty:
            cli["inc_pct"] = cli["inc"] * 100
            cli["priority"] = cli["n"] * cli["inc"]
            top_cli = cli.sort_values("priority", ascending=False).head(5)

            lines.append("\n### Top clínicas para atacar (volume × incidentes, min 30 props)")
            for _, r in top_cli.iterrows():
                lines.append(f"- **{r['Clinica']}** | N={int(r['n'])} | Inc={r['inc_pct']:.1f}%")

    if "has_rebate" in df.columns and "paid_p1" in df.columns:
        rb = (
            df.groupby("has_rebate", dropna=False)
            .agg(
                n=("proposal_id", "nunique"),
                paid=("paid_p1", "mean"),
                inc=("is_incident", "mean"),
                spread=("rebate_spread", lambda s: safe_num(s).mean()),
            )
            .reset_index()
        )
        if not rb.empty:
            lines.append("\n### Rebate: com vs sem (no recorte atual)")
            for _, r in rb.iterrows():
                tag = "Com Rebate" if bool(r["has_rebate"]) else "Sem Rebate"
                lines.append(
                    f"- **{tag}** | N={int(r['n'])} | PaidP1={(r['paid']*100):.1f}% | "
                    f"Inc={(r['inc']*100):.1f}% | Spread={r['spread']:.4f}"
                )

    lines.append("\n### Próximas perguntas úteis")
    lines.append("- Quais políticas explicam a variação de Paid P1 entre meses selecionados?")
    lines.append("- Quais clínicas concentram incidentes e com quais políticas?")
    lines.append("- Rebate melhora Paid P1 em quais score_buckets (e piora incidentes?)")
    lines.append("- Compare um mês específico: escreva algo como **2024-05** no chat.")

    return "\n".join(lines)


def offline_answer(
    user_q: str,
    df_all: pd.DataFrame,
    last_table: Optional[pd.DataFrame] = None,
    last_table_meta: Optional[dict] = None,
) -> str:
    """Responde perguntas comuns usando apenas df_all (já filtrado)."""
    q = (user_q or "").strip()
    ql = q.lower()

    available = sorted(df_all["base_month"].dropna().astype(str).unique().tolist())
    months = _pick_months_from_text(ql, available)

    df = df_all.copy()
    if months:
        df = df[df["base_month"].astype(str).isin(months)].copy()

    if any(w in ql for w in ["resumo", "snapshot", "visão geral", "visao geral", "geral"]):
        return offline_insights(df)

    if last_table is not None and last_table_meta is not None and any(
        w in ql for w in ["tabela", "interpreta", "interprete", "explica", "explique"]
    ):
        meta = last_table_meta
        head = last_table.head(10)
        return (
            f"### Interpretação da tabela (offline)\n"
            f"- Métrica: **{meta.get('metric')}** | Dimensão: **{meta.get('dim')}** | TopN: **{meta.get('topn')}**\n\n"
            f"**Top 10 linhas:**\n\n"
            f"{head.to_markdown(index=False)}\n\n"
            f"Dica: você pode citar um mês específico no chat, ex.: **2024-05**."
        )

    if "política" in ql or "politica" in ql:
        if "Política" not in df.columns:
            return "Não encontrei a coluna **Política** no recorte atual."
        pol = (
            df.groupby("Política", dropna=False)
            .agg(
                n=("proposal_id", "nunique"),
                paid=("paid_p1", "mean"),
                inc=("is_incident", "mean"),
                val=("[PAGO] Valor Emprestado", lambda s: safe_num(s).sum()),
                taxa=("Taxa de Juros", lambda s: safe_num(s).mean()),
            )
            .reset_index()
        )
        pol = pol[pol["n"] >= 30].copy()
        if pol.empty:
            return "Pouco volume por política no recorte (min 30 props). Tente remover filtros."
        pol["paid_pct"] = pol["paid"] * 100
        pol["inc_pct"] = pol["inc"] * 100
        pol["trade_score"] = (pol["paid_pct"] - pol["inc_pct"]) * np.log1p(pol["n"])
        best = pol.sort_values("trade_score", ascending=False).head(10)
        out = best[["Política", "n", "paid_pct", "inc_pct", "taxa", "val"]].rename(
            columns={"n": "N", "paid_pct": "PaidP1%", "inc_pct": "Inc%", "taxa": "Taxa média", "val": "Valor total"}
        )
        return (
            "### Políticas — ranking de trade-off (offline)\n"
            "(trade_score = (PaidP1% - Inc%) * log(1+N), min 30 props)\n\n"
            f"{out.to_markdown(index=False)}\n\n"
            "Se você quiser: eu também monto o **pior ranking** e sugiro hipóteses do porquê."
        )

    if "clínica" in ql or "clinica" in ql:
        if "Clinica" not in df.columns:
            return "Não encontrei a coluna **Clinica** no recorte atual."
        cli = (
            df.groupby("Clinica", dropna=False)
            .agg(n=("proposal_id", "nunique"), inc=("is_incident", "mean"), paid=("paid_p1", "mean"))
            .reset_index()
        )
        cli = cli[cli["n"] >= 30].copy()
        if cli.empty:
            return "Pouco volume por clínica no recorte (min 30 props). Tente remover filtros."
        cli["inc_pct"] = cli["inc"] * 100
        cli["paid_pct"] = cli["paid"] * 100
        cli["priority"] = cli["n"] * cli["inc"]
        top = cli.sort_values("priority", ascending=False).head(10)
        out = top[["Clinica", "n", "inc_pct", "paid_pct"]].rename(columns={"n": "N", "inc_pct": "Inc%", "paid_pct": "PaidP1%"})
        return (
            "### Clínicas — prioridade (offline)\n"
            "(priority = N * Incidente%, min 30 props)\n\n"
            f"{out.to_markdown(index=False)}\n\n"
            "Sugestão: para as top 3 clínicas, ver quais **políticas** dominam o mix."
        )

    if "rebate" in ql:
        if "has_rebate" not in df.columns:
            return "Não encontrei `has_rebate` no recorte atual."
        rb = (
            df.groupby("has_rebate", dropna=False)
            .agg(
                n=("proposal_id", "nunique"),
                paid=("paid_p1", "mean"),
                inc=("is_incident", "mean"),
                spread=("rebate_spread", lambda s: safe_num(s).mean()),
            )
            .reset_index()
        )
        rb["PaidP1%"] = (rb["paid"] * 100).round(2)
        rb["Inc%"] = (rb["inc"] * 100).round(2)
        rb["Spread"] = rb["spread"].round(4)
        rb = rb.rename(columns={"has_rebate": "Com rebate?", "n": "N"})
        return (
            "### Rebate — com vs sem (offline)\n\n"
            f"{rb[['Com rebate?','N','PaidP1%','Inc%','Spread']].to_markdown(index=False)}\n\n"
            "Se você quiser, eu quebro por **Política x score_bucket** para ver onde rebate realmente ajuda."
        )

    if "incidente" in ql or "incidentes" in ql:
        g = (
            df.groupby("base_month", dropna=False)
            .agg(n=("proposal_id", "nunique"), inc=("is_incident", "mean"), paid=("paid_p1", "mean"))
            .reset_index()
            .sort_values("base_month")
        )
        g["Inc%"] = (g["inc"] * 100).round(2)
        g["PaidP1%"] = (g["paid"] * 100).round(2)
        out = g[["base_month", "n", "Inc%", "PaidP1%"]].rename(columns={"base_month": "Mês", "n": "N"})
        return (
            "### Incidentes por mês (offline)\n\n"
            f"{out.to_markdown(index=False)}\n\n"
            "Se quiser, eu aponto **top políticas** e **top clínicas** que mais contribuem para os incidentes."
        )

    if any(w in ql for w in ["paid", "p1", "convers", "pagamento", "pagou"]):
        g = (
            df.groupby("base_month", dropna=False)
            .agg(n=("proposal_id", "nunique"), paid=("paid_p1", "mean"), inc=("is_incident", "mean"))
            .reset_index()
            .sort_values("base_month")
        )
        g["PaidP1%"] = (g["paid"] * 100).round(2)
        g["Inc%"] = (g["inc"] * 100).round(2)
        out = g[["base_month", "n", "PaidP1%", "Inc%"]].rename(columns={"base_month": "Mês", "n": "N"})
        return (
            "### Paid P1 por mês (offline)\n\n"
            f"{out.to_markdown(index=False)}\n\n"
            "Quer que eu mostre as **políticas** com maior Paid P1 e o trade-off com incidentes?"
        )

    return (
        "Posso ajudar com perguntas como:\n"
        "- **resumo** do recorte\n"
        "- **políticas** (ranking trade-off)\n"
        "- **clínicas** (prioridade por incidentes)\n"
        "- **rebate** (com vs sem)\n"
        "- **incidentes** por mês\n"
        "- **paid p1** por mês\n\n"
        "Dicas:\n"
        "- cite um mês no formato **YYYY-MM** (ex.: 2024-05)\n"
        "- ou cite um mês por nome (ex.: 'maio')"
    )


# -----------------------------
# UI
# -----------------------------
st.title("PM Analytics — Análise por Período (base_month)")
st.caption("Base temporal: base_month derivado de [PAGO] Data do Pagamento (Tabela 1). Filtros afetam todas as abas.")

mp, mi = load_master()

# -----------------------------
# Período: todos os meses disponíveis
# -----------------------------
available_months = sort_base_months(
    mp["base_month"].dropna().astype(str).unique().tolist()
)

st.sidebar.header("Filtros (aplicam em tudo)")

preset = st.sidebar.selectbox(
    "Atalho de período",
    ["Últimos 3 meses", "Últimos 6 meses", "Últimos 12 meses", "Tudo", "Custom"],
    index=0,
)

if preset == "Últimos 3 meses":
    default_months = available_months[-3:]
elif preset == "Últimos 6 meses":
    default_months = available_months[-6:]
elif preset == "Últimos 12 meses":
    default_months = available_months[-12:]
elif preset == "Tudo":
    default_months = available_months
else:
    default_months = available_months[-3:]

sel_months = st.sidebar.multiselect(
    "Meses com operação (base_month)",
    options=available_months,
    default=default_months,
)

if not sel_months:
    st.warning("Selecione pelo menos 1 mês para análise.")
    st.stop()

df = mp[mp["base_month"].astype(str).isin(sel_months)].copy()

# ensure cluster
df, cluster_method = ensure_cluster(df, n_clusters=6)

# dynamic filter lists
def ms(col: str, label: str):
    if col not in df.columns:
        return None
    opts = sorted([x for x in df[col].dropna().astype(str).unique()])
    return st.sidebar.multiselect(label, options=opts)

pol_sel = ms("Política", "Política")
cli_sel = ms("Clinica", "Clínica")
emp_sel = ms("employment_macro", "Emprego (macro)")
clu_sel = ms("cluster", f"Cluster ({cluster_method})")
score_sel = ms("score_bucket", "Score bucket")
taxa_sel = ms("taxa_bucket", "Taxa bucket")

flags = st.sidebar.multiselect(
    "Flags",
    options=["Somente Incidentes", "Somente Rebate", "Somente Paid P1"],
)

# apply filters
def apply_ms(d: pd.DataFrame, col: str, selected: Optional[List[str]]):
    if selected and col in d.columns:
        return d[d[col].astype(str).isin(selected)]
    return d

df_f = df.copy()
df_f = apply_ms(df_f, "Política", pol_sel)
df_f = apply_ms(df_f, "Clinica", cli_sel)
df_f = apply_ms(df_f, "employment_macro", emp_sel)
df_f = apply_ms(df_f, "cluster", clu_sel)
df_f = apply_ms(df_f, "score_bucket", score_sel)
df_f = apply_ms(df_f, "taxa_bucket", taxa_sel)

if "Somente Incidentes" in flags and "is_incident" in df_f.columns:
    df_f = df_f[df_f["is_incident"] == True]
if "Somente Rebate" in flags and "has_rebate" in df_f.columns:
    df_f = df_f[df_f["has_rebate"] == True]
if "Somente Paid P1" in flags and "paid_p1" in df_f.columns:
    df_f = df_f[df_f["paid_p1"] == True]

kpis = kpi_row(df_f)

# Tabs
tabs = st.tabs([
    "📌 Visão Geral",
    "📋 Tabelas",
    "📈 Gráficos",
    "🏥 Clínicas",
    "💼 Empregos",
    "🧠 Clusters",
    "💸 Rebate",
    "🧾 Parcelas",
    "🤖 Chat Offline",
])

# -----------------------------
# Tab: Visão Geral
# -----------------------------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Propostas", f"{kpis['propostas']:,}".replace(",", "."))
    c2.metric("Valor total", money_br(kpis["valor_total"]))
    c3.metric("Ticket médio", money_br(kpis["ticket_medio"]))
    c4.metric("Taxa média", f"{kpis['taxa_media']:.4f}")
    c5.metric("Score médio", f"{kpis['score_medio']:.1f}")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Paid P1 (%)", f"{kpis['paid_p1_rate_pct']:.2f}%")
    c7.metric("Incidentes (%)", f"{kpis['incident_rate_pct']:.2f}%")
    c8.metric("Rebate share (%)", f"{kpis['rebate_share_pct']:.2f}%")
    c9.metric("Rebate spread médio", f"{kpis['rebate_spread_medio']:.4f}")

    g = (
        df_f.groupby("base_month", dropna=False)
        .agg(
            propostas=("proposal_id", "nunique"),
            valor_total=("[PAGO] Valor Emprestado", lambda s: safe_num(s).sum()),
            paid_p1=("paid_p1", "mean"),
            incident=("is_incident", "mean"),
            rebate=("has_rebate", "mean"),
        )
        .reset_index()
        .sort_values("base_month")
    )
    g["paid_p1_pct"] = (g["paid_p1"] * 100).round(2)
    g["incident_pct"] = (g["incident"] * 100).round(2)
    g["rebate_pct"] = (g["rebate"] * 100).round(2)

    fig = px.bar(g, x="base_month", y="propostas", title="Propostas por mês (base_month)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=g["base_month"], y=g["paid_p1_pct"], name="Paid P1 (%)", mode="lines+markers"))
    fig2.add_trace(go.Scatter(x=g["base_month"], y=g["incident_pct"], name="Incidentes (%)", mode="lines+markers"))
    fig2.add_trace(go.Scatter(x=g["base_month"], y=g["rebate_pct"], name="Rebate share (%)", mode="lines+markers"))
    fig2.update_layout(title="Rates por mês (Paid P1 / Incidentes / Rebate)", yaxis_title="%")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Tab: Tabelas
# -----------------------------
with tabs[1]:
    st.subheader("Agregações (com base nos filtros atuais)")

    colA, colB = st.columns(2)
    with colA:
        dim = st.selectbox("Dimensão", ["Política", "Clinica", "employment_macro", "cluster", "score_bucket", "taxa_bucket", "base_month"])
    with colB:
        metric = st.selectbox("Métrica", list(ALLOWED_METRICS.keys()), index=list(ALLOWED_METRICS.keys()).index("propostas"))

    tbl = compute_table(df_f, metric=metric, groupby=[dim], top_n=30)
    st.dataframe(tbl, use_container_width=True)

    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", data=csv, file_name=f"table_{dim}_{metric}.csv", mime="text/csv")

# -----------------------------
# Tab: Gráficos
# -----------------------------
with tabs[2]:
    st.subheader("Gráficos principais")

    dim_scatter = st.selectbox("Scatter por", ["Política", "Clinica", "employment_macro", "cluster"], index=0)

    agg = (
        df_f.groupby(dim_scatter, dropna=False)
        .agg(
            propostas=("proposal_id", "nunique"),
            incident=("is_incident", "mean"),
            paid_p1=("paid_p1", "mean"),
            taxa_media=("Taxa de Juros", lambda s: safe_num(s).mean()),
        )
        .reset_index()
    )
    agg["incident_pct"] = agg["incident"] * 100
    agg["paid_p1_pct"] = agg["paid_p1"] * 100

    fig = px.scatter(
        agg,
        x="incident_pct",
        y="paid_p1_pct",
        size="propostas",
        color=dim_scatter,
        hover_data=["propostas", "taxa_media"],
        title=f"Trade-off: Incidentes (%) vs Paid P1 (%) — por {dim_scatter}",
    )
    fig.update_layout(xaxis_title="Incidentes (%)", yaxis_title="Paid P1 (%)")
    st.plotly_chart(fig, use_container_width=True)

    topn = st.slider("Top N (barras)", 5, 30, 12)
    agg2 = agg.sort_values("propostas", ascending=False).head(topn)
    fig2 = px.bar(agg2, x=dim_scatter, y="propostas", title=f"Top {topn} por volume — {dim_scatter}")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Tab: Clínicas
# -----------------------------
with tabs[3]:
    st.subheader("Clínicas — volume, incidentes, paid P1, taxa")

    if "Clinica" not in df_f.columns:
        st.info("Coluna Clinica não disponível.")
    else:
        clinic = (
            df_f.groupby("Clinica", dropna=False)
            .agg(
                propostas=("proposal_id", "nunique"),
                incident=("is_incident", "mean"),
                paid_p1=("paid_p1", "mean"),
                taxa_media=("Taxa de Juros", lambda s: safe_num(s).mean()),
                valor_total=("[PAGO] Valor Emprestado", lambda s: safe_num(s).sum()),
            )
            .reset_index()
        )
        clinic["incident_pct"] = (clinic["incident"] * 100).round(2)
        clinic["paid_p1_pct"] = (clinic["paid_p1"] * 100).round(2)
        clinic["priority_score"] = clinic["propostas"] * (clinic["incident"])

        clinic = clinic.sort_values("priority_score", ascending=False)
        st.dataframe(clinic.head(30), use_container_width=True)

        fig = px.scatter(
            clinic.head(80),
            x="propostas",
            y="incident_pct",
            size="valor_total",
            color="paid_p1_pct",
            hover_name="Clinica",
            title="Clínicas: Volume vs Incidentes (cor=Paid P1, size=Valor total)",
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab: Empregos
# -----------------------------
with tabs[4]:
    st.subheader("Empregos (macro) — comparativos")

    emp = (
        df_f.groupby("employment_macro", dropna=False)
        .agg(
            propostas=("proposal_id", "nunique"),
            incident=("is_incident", "mean"),
            paid_p1=("paid_p1", "mean"),
            taxa_media=("Taxa de Juros", lambda s: safe_num(s).mean()),
            score_medio=("Score Serasa", lambda s: safe_num(s).mean()),
        )
        .reset_index()
        .sort_values("propostas", ascending=False)
    )
    emp["incident_pct"] = (emp["incident"] * 100).round(2)
    emp["paid_p1_pct"] = (emp["paid_p1"] * 100).round(2)

    st.dataframe(emp, use_container_width=True)

    fig = px.bar(emp, x="employment_macro", y="propostas", title="Volume por emprego (macro)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab: Clusters
# -----------------------------
with tabs[5]:
    st.subheader(f"Clusters — método: {cluster_method}")

    clu = (
        df_f.groupby("cluster", dropna=False)
        .agg(
            propostas=("proposal_id", "nunique"),
            incident=("is_incident", "mean"),
            paid_p1=("paid_p1", "mean"),
            taxa_media=("Taxa de Juros", lambda s: safe_num(s).mean()),
            score_medio=("Score Serasa", lambda s: safe_num(s).mean()),
            valor_total=("[PAGO] Valor Emprestado", lambda s: safe_num(s).sum()),
        )
        .reset_index()
        .sort_values("propostas", ascending=False)
    )
    clu["incident_pct"] = (clu["incident"] * 100).round(2)
    clu["paid_p1_pct"] = (clu["paid_p1"] * 100).round(2)

    st.dataframe(clu, use_container_width=True)

    if "Política" in df_f.columns:
        mix = (
            df_f.groupby(["cluster", "Política"], dropna=False)["proposal_id"]
            .nunique()
            .reset_index(name="propostas")
            .sort_values(["cluster", "propostas"], ascending=[True, False])
        )
        st.caption("Mix de políticas por cluster (top por cluster)")
        st.dataframe(mix.groupby("cluster").head(10), use_container_width=True)

# -----------------------------
# Tab: Rebate
# -----------------------------
with tabs[6]:
    st.subheader("Rebate — share, spread, onde está concentrado")

    rbm = (
        df_f.groupby("base_month", dropna=False)
        .agg(
            propostas=("proposal_id", "nunique"),
            rebate=("has_rebate", "mean"),
            spread=("rebate_spread", lambda s: safe_num(s).mean()),
            paid_p1=("paid_p1", "mean"),
            incident=("is_incident", "mean"),
        )
        .reset_index()
        .sort_values("base_month")
    )
    rbm["rebate_pct"] = (rbm["rebate"] * 100).round(2)
    rbm["paid_p1_pct"] = (rbm["paid_p1"] * 100).round(2)
    rbm["incident_pct"] = (rbm["incident"] * 100).round(2)

    st.dataframe(rbm, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=rbm["base_month"], y=rbm["rebate_pct"], name="Rebate share (%)"))
    fig.add_trace(go.Scatter(x=rbm["base_month"], y=rbm["paid_p1_pct"], name="Paid P1 (%)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=rbm["base_month"], y=rbm["incident_pct"], name="Incidentes (%)", mode="lines+markers"))
    fig.update_layout(title="Rebate share vs Paid P1 vs Incidentes (por mês)", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)

    if "rebate_spread" in df_f.columns:
        sp = safe_num(df_f["rebate_spread"])
        fig2 = px.histogram(sp.dropna(), nbins=40, title="Distribuição do rebate_spread")
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Tab: Parcelas
# -----------------------------
with tabs[7]:
    st.subheader("Parcelas — status por parcela_n (baseado em master_installments)")

    ids = set(df_f["proposal_id"].astype(str).unique())
    mi2 = mi.copy()
    if "proposal_id" in mi2.columns:
        mi2 = mi2[mi2["proposal_id"].astype(str).isin(ids)]

    if mi2.empty:
        st.info("Sem parcelas para os filtros atuais.")
    else:
        if "Status" in mi2.columns:
            mi2["is_paid"] = mi2["Status"].astype("string").str.strip().str.lower().eq("pago")
        else:
            mi2["is_paid"] = False

        parc = (
            mi2.groupby("parcela_n", dropna=False)
            .agg(
                linhas=("payment_unique_id", "size") if "payment_unique_id" in mi2.columns else ("unique id", "size"),
                propostas=("proposal_id", "nunique"),
                paid_rate=("is_paid", "mean"),
            )
            .reset_index()
            .sort_values("parcela_n")
        )
        parc["paid_rate_pct"] = (parc["paid_rate"] * 100).round(2)
        st.dataframe(parc, use_container_width=True)

        fig = px.bar(parc, x="parcela_n", y="propostas", title="Propostas com registros por parcela_n")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(parc, x="parcela_n", y="paid_rate_pct", markers=True, title="Paid rate (%) por parcela_n")
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Tab: Chat Offline
# -----------------------------
with tabs[8]:
    st.subheader("Chat Offline (sem OpenAI) — insights a partir dos seus dados")
    st.caption("Esse chat responde usando apenas o recorte atual (filtros + meses selecionados). Você também pode anexar uma tabela para interpretação.")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_table" not in st.session_state:
        st.session_state.last_table = None
    if "last_table_meta" not in st.session_state:
        st.session_state.last_table_meta = None

    st.markdown("### 1) Gerar uma tabela (opcional) e anexar ao contexto")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        chat_metric = st.selectbox(
            "Métrica (para a tabela)",
            list(ALLOWED_METRICS.keys()),
            index=list(ALLOWED_METRICS.keys()).index("paid_p1_rate"),
        )
    with c2:
        chat_dim = st.selectbox(
            "Dimensão (group by)",
            [d for d in ALLOWED_DIMS if d in df_f.columns],
            index=0,
        )
    with c3:
        chat_topn = st.number_input("Top N", min_value=5, max_value=50, value=15, step=1)

    if st.button("📌 Calcular tabela e anexar"):
        try:
            generated_table = compute_table(df_f, metric=chat_metric, groupby=[chat_dim], top_n=int(chat_topn))
            st.success("Tabela calculada e anexada ao chat.")
            st.dataframe(generated_table, use_container_width=True)
            st.session_state.last_table = generated_table
            st.session_state.last_table_meta = {"metric": chat_metric, "dim": chat_dim, "topn": int(chat_topn)}
        except Exception as e:
            st.error(f"Erro ao calcular tabela: {e}")

    st.markdown("### 2) Resumo automático (offline)")
    if st.button("🧾 Gerar resumo do recorte"):
        st.markdown(offline_insights(df_f))

    st.markdown("### 3) Conversa")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Pergunte algo tipo: 'Top políticas por trade-off' ou 'Incidentes por mês' (dica: 2024-05)")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})

        answer = offline_answer(
            user_q=user_q,
            df_all=df_f,
            last_table=st.session_state.last_table,
            last_table_meta=st.session_state.last_table_meta,
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat.append({"role": "assistant", "content": answer})