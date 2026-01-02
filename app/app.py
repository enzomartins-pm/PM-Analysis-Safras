import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# Config
# =========================================================
st.set_page_config(
    page_title="PM Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RUNTIME_DATA_DIR = Path(os.getenv("PM_RUNTIME_DATA_DIR", "/tmp/pm_analytics_data"))
RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# UI (CSS) — alinhamento e visual profissional
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
          }

          div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(120,120,120,.18);
          }

          /* Títulos */
          h1, h2, h3 { letter-spacing: -0.02em; }

          /* Cards de seção */
          .pm-section {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
            background: rgba(255,255,255,0.02);
          }
          .pm-section-title {
            font-size: 0.98rem;
            font-weight: 700;
            margin: 0 0 8px 0;
            opacity: .92;
          }

          /* Métricas com altura consistente */
          div[data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 14px;
            padding: 12px 12px;
            background: rgba(255,255,255,0.02);
            height: 92px;
          }
          div[data-testid="stMetric"] label {
            opacity: .75;
            font-weight: 600;
          }

          /* Dataframes */
          .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
          }

          /* Ajuste fino de espaçamento de headers */
          .stMarkdown h2 { margin-bottom: .4rem; }
          .stMarkdown h3 { margin-bottom: .35rem; }

        </style>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(
        f'<div class="pm-section"><div class="pm-section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def end_section() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Utils
# =========================================================
def parse_br_number(x: Any) -> float:
    """Converte strings BR/US para float. Aceita '1.234,56', '1234,56', '1,234.56', etc."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    s = s.replace("R$", "").replace(" ", "").strip()

    # BR 1.234,56
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def safe_num(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="float64")
    return s.map(parse_br_number)


def normalize_rate(series: pd.Series) -> pd.Series:
    """
    Normaliza taxa:
    - Se parece estar em % (ex.: 12.5), converte para decimal (0.125)
    - Se já está em decimal (0.125), mantém
    """
    x = safe_num(series)
    d = x.dropna()
    if d.shape[0] > 0 and (d > 1).mean() > 0.8:
        x = x / 100.0
    return x


def sort_base_months(months: List[str]) -> List[str]:
    m = pd.Series(months).dropna().astype(str)
    if m.empty:
        return []
    try:
        p = pd.PeriodIndex(m, freq="M")
        return [str(x) for x in p.sort_values().astype(str)]
    except Exception:
        return sorted(m.unique().tolist())


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def money_br(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def clinic_key(x: Any) -> str:
    """Chave normalizada para clínica (remove acentos, baixa caixa, colapsa espaços)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s


def _norm_txt(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s


def pct_fmt(x: float, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{dec}f}%"


# =========================================================
# Data loading (local vs cloud runtime) + private upload
# =========================================================
def resolve_paths() -> Tuple[Path, Path, str]:
    p1_local = DEFAULT_PROCESSED_DIR / "master_proposals.parquet"
    p2_local = DEFAULT_PROCESSED_DIR / "master_installments.parquet"
    if p1_local.exists() and p2_local.exists():
        return p1_local, p2_local, "local"
    return (
        RUNTIME_DATA_DIR / "master_proposals.parquet",
        RUNTIME_DATA_DIR / "master_installments.parquet",
        "runtime",
    )


MASTER_PROPOSALS_PATH, MASTER_INSTALLMENTS_PATH, DATA_MODE = resolve_paths()


def ensure_data_files() -> None:
    """
    Se os parquets não existirem no deploy, permite upload e salva em /tmp (runtime).
    """
    global MASTER_PROPOSALS_PATH, MASTER_INSTALLMENTS_PATH, DATA_MODE

    with st.sidebar:
        st.markdown("### Dados")
        st.caption(f"Fonte detectada: {DATA_MODE}")

        if st.button("Resetar dados (apagar e reenviar)"):
            for p in [MASTER_PROPOSALS_PATH, MASTER_INSTALLMENTS_PATH]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            st.cache_data.clear()
            st.rerun()

    if MASTER_PROPOSALS_PATH.exists() and MASTER_INSTALLMENTS_PATH.exists():
        return

    st.warning(
        "Dados não encontrados no ambiente.\n\n"
        "Envie os 2 arquivos na barra lateral:\n"
        "- master_proposals.parquet\n"
        "- master_installments.parquet\n\n"
        "Eles serão salvos temporariamente em /tmp (não versionado)."
    )

    with st.sidebar:
        up1 = st.file_uploader("Upload: master_proposals.parquet", type=["parquet"])
        up2 = st.file_uploader("Upload: master_installments.parquet", type=["parquet"])

        if up1 is not None and up2 is not None:
            MASTER_PROPOSALS_PATH = RUNTIME_DATA_DIR / "master_proposals.parquet"
            MASTER_INSTALLMENTS_PATH = RUNTIME_DATA_DIR / "master_installments.parquet"
            DATA_MODE = "runtime"

            MASTER_PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
            MASTER_PROPOSALS_PATH.write_bytes(up1.getbuffer())
            MASTER_INSTALLMENTS_PATH.write_bytes(up2.getbuffer())

            st.success("Arquivos recebidos. Recarregando.")
            st.cache_data.clear()
            st.rerun()

    st.stop()


@st.cache_data(show_spinner=True)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    mp = pd.read_parquet(MASTER_PROPOSALS_PATH)
    mi = pd.read_parquet(MASTER_INSTALLMENTS_PATH)

    # proposal_id
    if "proposal_id" not in mp.columns:
        c = pick_first_existing(mp, ["proposal", "unique id", "CPF Proposta"])
        if c:
            mp["proposal_id"] = mp[c].astype("string")
        else:
            raise ValueError(
                "master_proposals não tem proposal_id nem colunas alternativas (proposal/unique id/CPF Proposta)."
            )
    mp["proposal_id"] = mp["proposal_id"].astype("string")

    # base_month
    if "base_month" not in mp.columns:
        if "base_date" in mp.columns:
            dt = pd.to_datetime(mp["base_date"], errors="coerce")
        elif "[PAGO] Data do Pagamento" in mp.columns:
            dt = pd.to_datetime(mp["[PAGO] Data do Pagamento"], errors="coerce")
        else:
            dt = pd.NaT
        mp["base_month"] = pd.to_datetime(dt, errors="coerce").dt.to_period("M").astype("string")

    mp = mp[mp["base_month"].notna()].copy()

    # clinica alias simples
    if "Clinica" not in mp.columns and "Clínica" in mp.columns:
        mp["Clinica"] = mp["Clínica"]

    if "proposal_id" in mi.columns:
        mi["proposal_id"] = mi["proposal_id"].astype("string")

    return mp, mi


# =========================================================
# Flags / Enriquecimentos
# =========================================================
def ensure_incident_flag(mp_f: pd.DataFrame) -> pd.DataFrame:
    """
    Se não existir is_incident, tenta inferir por colunas de erro (SCR/SRM).
    """
    out = mp_f.copy()

    if "is_incident" in out.columns:
        out["is_incident"] = out["is_incident"].fillna(False).astype(bool)
        return out

    scr_col = pick_first_existing(out, ["Consulta scr erro?", "consulta scr erro?", "SCR erro?", "scr_erro"])
    srm_col = pick_first_existing(out, ["consulta srm filme erro?", "Consulta srm filme erro?", "SRM erro?", "srm_erro"])

    scr = (
        out[scr_col].astype("string").str.strip().str.lower().isin(["sim", "s", "true", "1"])
        if scr_col
        else pd.Series([False] * len(out), index=out.index)
    )
    srm = (
        out[srm_col].astype("string").str.strip().str.lower().isin(["sim", "s", "true", "1"])
        if srm_col
        else pd.Series([False] * len(out), index=out.index)
    )

    out["is_incident"] = (scr.fillna(False) | srm.fillna(False)).astype(bool)
    return out


def build_paid_p1_flag(mi: pd.DataFrame) -> pd.DataFrame:
    """
    paid_p1: proposta com parcela_n == 1 marcada como 'Pago' no Status
    """
    if mi is None or mi.empty or "proposal_id" not in mi.columns:
        return pd.DataFrame(columns=["proposal_id", "paid_p1"])

    if "parcela_n" not in mi.columns or "Status" not in mi.columns:
        return pd.DataFrame({"proposal_id": mi["proposal_id"].dropna().unique(), "paid_p1": False})

    paid_status = mi["Status"].astype("string").str.strip().str.lower().eq("pago")
    p1 = mi[(mi["parcela_n"] == 1) & paid_status].groupby("proposal_id").size().reset_index(name="cnt")
    p1["paid_p1"] = p1["cnt"] > 0
    return p1[["proposal_id", "paid_p1"]]


def build_overdue_flags(mi: pd.DataFrame, ref_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Over30/Over90 por proposta:
    - parcela vencida (due_date <= ref_date)
    - não paga (Status != 'Pago')
    - (ref_date - due_date) > 30 / > 90
    """
    if mi is None or mi.empty or "proposal_id" not in mi.columns:
        return pd.DataFrame(columns=["proposal_id", "over30", "over90"])

    ref_date = ref_date or pd.Timestamp.today().normalize()

    due_col = pick_first_existing(mi, ["Data de vencimento", "Data de Vencimento", "Due Date", "Vencimento"])
    if due_col is None:
        return pd.DataFrame({"proposal_id": mi["proposal_id"].dropna().unique(), "over30": False, "over90": False})

    status_col = pick_first_existing(mi, ["Status"])
    due = pd.to_datetime(mi[due_col], errors="coerce")

    if status_col:
        is_paid = mi[status_col].astype("string").str.strip().str.lower().eq("pago")
    else:
        is_paid = pd.Series([False] * len(mi), index=mi.index)

    eligible = due.notna() & (due <= ref_date) & (~is_paid.fillna(False))
    dpd = (ref_date - due).dt.days.where(eligible, 0)

    mi2 = mi[["proposal_id"]].copy()
    mi2["over30_row"] = dpd > 30
    mi2["over90_row"] = dpd > 90

    out = (
        mi2.groupby("proposal_id", dropna=False)
        .agg(over30=("over30_row", "max"), over90=("over90_row", "max"))
        .reset_index()
    )

    out["over30"] = out["over30"].fillna(False).astype(bool)
    out["over90"] = out["over90"].fillna(False).astype(bool)
    return out[["proposal_id", "over30", "over90"]]


def ensure_employment_macro(mp_f: pd.DataFrame) -> pd.DataFrame:
    """
    Cria employment_macro a partir de colunas comuns.
    Se não achar nada, employment_macro = "NA".
    """
    out = mp_f.copy()

    if "employment_macro" in out.columns:
        out["employment_macro"] = out["employment_macro"].astype("string").fillna("NA")
        return out

    emp_col = pick_first_existing(
        out,
        [
            "Estado Empregatício Informado",
            "estado empregatício informado",
            "Estado empregaticio informado",
            "estado empregaticio informado",
            "Emprego",
            "Ocupação",
            "Ocupacao",
            "Profissão",
            "Profissao",
        ],
    )

    if emp_col is None:
        out["employment_macro"] = "NA"
        return out

    raw = out[emp_col].astype("string")

    def _map_emp(x: Any) -> str:
        s = _norm_txt(x)
        if s in {"", "nan", "none", "null"}:
            return "NA"
        if "aposent" in s or "pension" in s:
            return "APOSENTADO/PENSIONISTA"
        if "auton" in s or "autonom" in s or "mei" in s or "microempre" in s:
            return "AUTONOMO/MEI"
        if ("func" in s and "pub" in s) or "servidor" in s or "estatut" in s:
            return "FUNC_PUBLICO"
        if "clt" in s or "registrad" in s or "assalariad" in s:
            return "CLT"
        if "desempreg" in s or "sem emprego" in s:
            return "DESEMPREGADO"
        if "estud" in s:
            return "ESTUDANTE"
        if "empres" in s or "socio" in s or "pj" in s:
            return "EMPRESARIO/PJ"
        if "rural" in s or "agric" in s:
            return "RURAL"
        return "OUTROS"

    out["employment_macro"] = raw.map(_map_emp).astype("string").fillna("NA")
    return out


def ensure_policy_col(mp_f: pd.DataFrame) -> pd.DataFrame:
    """
    Garante coluna canônica 'Política' (mesmo que esteja vazia).
    """
    out = mp_f.copy()
    if "Política" in out.columns:
        out["Política"] = out["Política"].astype("string").str.strip()
        out.loc[out["Política"].isin(["", "nan", "None"]), "Política"] = pd.NA
        return out

    cand = pick_first_existing(out, ["Politica", "POLITICA", "política", "politica", "Política "])
    if cand:
        out["Política"] = out[cand].astype("string").str.strip()
        out.loc[out["Política"].isin(["", "nan", "None"]), "Política"] = pd.NA
    else:
        out["Política"] = pd.NA
    return out


def ensure_score_taxa_buckets(mp_f: pd.DataFrame) -> pd.DataFrame:
    """
    Cria score_bucket e taxa_bucket (mesmo se os dados estiverem faltando).
    """
    out = mp_f.copy()

    score_col = pick_first_existing(out, ["Score Serasa", "[Serasa] Score", "Score"])
    if score_col:
        sc = safe_num(out[score_col])
        out["score_bucket"] = pd.cut(
            sc,
            bins=[-np.inf, 300, 500, 650, 750, np.inf],
            labels=["<=300", "301-500", "501-650", "651-750", "751+"],
        ).astype("string")
    else:
        out["score_bucket"] = pd.NA

    taxa_col = pick_first_existing(out, ["contract_taxa_juros", "[GLOBAL] Taxa de Juros", "Taxa de Juros"])
    if taxa_col:
        tx = normalize_rate(out[taxa_col])
        out["taxa_bucket"] = pd.cut(
            tx,
            bins=[-np.inf, 0.05, 0.08, 0.12, 0.18, np.inf],
            labels=["<=5%", "5-8%", "8-12%", "12-18%", "18%+"],
        ).astype("string")
    else:
        out["taxa_bucket"] = pd.NA

    return out


def ensure_funnel_flags(mp_f: pd.DataFrame) -> pd.DataFrame:
    """
    Cria:
      - is_approved: aprovado (True) / não (False)
      - is_contracted: contrato fechado (True) / não (False)
    """
    out = mp_f.copy()

    if "is_approved" in out.columns:
        out["is_approved"] = out["is_approved"].fillna(False).astype(bool)
    elif "is_approved_contract" in out.columns:
        out["is_approved"] = out["is_approved_contract"].fillna(False).astype(bool)
    else:
        appr_dt = pick_first_existing(out, ["contract_dt_approval", "[GLOBAL] Data da Aprovação", "Data Aprovação", "Data da Aprovação"])
        out["is_approved"] = pd.to_datetime(out[appr_dt], errors="coerce").notna() if appr_dt else False

    if "is_contracted" in out.columns:
        out["is_contracted"] = out["is_contracted"].fillna(False).astype(bool)
    elif "is_contracted_contract" in out.columns:
        out["is_contracted"] = out["is_contracted_contract"].fillna(False).astype(bool)
    else:
        ctr_dt = pick_first_existing(out, ["contract_dt_contract", "[GLOBAL] Data da Contratação", "Data Contratação", "Data da Contratacao"])
        if ctr_dt:
            out["is_contracted"] = pd.to_datetime(out[ctr_dt], errors="coerce").notna()
        else:
            st_col = pick_first_existing(out, ["contract_status_geral", "Status (Geral)"])
            if st_col:
                s = out[st_col].astype("string").str.strip().str.lower()
                out["is_contracted"] = s.isin(["emprestado", "contratado", "ativo"])
            else:
                out["is_contracted"] = False

    return out


# =========================================================
# KPI builders
# =========================================================
def compute_kpis(mp_f: pd.DataFrame) -> Dict[str, Any]:
    n_props = int(mp_f["proposal_id"].nunique())

    val_col = pick_first_existing(mp_f, ["[PAGO] Valor Emprestado", "contract_val_emprestado", "contract_val_contratado", "Valor"])
    val = safe_num(mp_f[val_col]) if val_col else pd.Series([np.nan] * len(mp_f))
    valor_total = float(val.sum(skipna=True))
    ticket_medio = float(val.mean(skipna=True))

    taxa_col = pick_first_existing(mp_f, ["contract_taxa_juros", "[GLOBAL] Taxa de Juros", "Taxa de Juros"])
    taxa = normalize_rate(mp_f[taxa_col]) if taxa_col else pd.Series([np.nan] * len(mp_f))
    taxa_media = float(taxa.mean(skipna=True))

    score_col = pick_first_existing(mp_f, ["Score Serasa", "[Serasa] Score", "Score"])
    score = safe_num(mp_f[score_col]) if score_col else pd.Series([np.nan] * len(mp_f))
    score_medio = float(score.mean(skipna=True))

    incident_pct = float(mp_f["is_incident"].fillna(False).astype(bool).mean() * 100) if "is_incident" in mp_f.columns else np.nan
    paid_p1_pct = float(mp_f["paid_p1"].fillna(False).astype(bool).mean() * 100) if "paid_p1" in mp_f.columns else np.nan
    over30_pct = float(mp_f["over30"].fillna(False).astype(bool).mean() * 100) if "over30" in mp_f.columns else np.nan
    over90_pct = float(mp_f["over90"].fillna(False).astype(bool).mean() * 100) if "over90" in mp_f.columns else np.nan

    return {
        "propostas": n_props,
        "valor_total": valor_total,
        "ticket_medio": ticket_medio,
        "taxa_media": taxa_media,
        "score_medio": score_medio,
        "paid_p1_pct": paid_p1_pct,
        "incident_pct": incident_pct,
        "over30_pct": over30_pct,
        "over90_pct": over90_pct,
        "val_col_used": val_col,
        "taxa_col_used": taxa_col,
        "score_col_used": score_col,
    }


def grouped_bar(df: pd.DataFrame, x: str, value_cols: List[str], title: str) -> None:
    keep = [x] + [c for c in value_cols if c in df.columns]
    if len(keep) <= 1:
        st.info("Sem colunas suficientes para gráfico.")
        return
    d = df[keep].copy()
    long = d.melt(id_vars=[x], var_name="métrica", value_name="valor")
    fig = px.bar(long, x=x, y="valor", color="métrica", barmode="group", title=title)
    fig.update_layout(xaxis_tickangle=-35, legend_title="")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Aggregations (funil)
# =========================================================
def agg_funnel(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    d = df.copy()

    score_col = pick_first_existing(d, ["Score Serasa", "[Serasa] Score", "Score"])
    taxa_col = pick_first_existing(d, ["contract_taxa_juros", "[GLOBAL] Taxa de Juros", "Taxa de Juros"])
    val_col = pick_first_existing(d, ["[PAGO] Valor Emprestado", "contract_val_emprestado", "contract_val_contratado"])

    d["_score"] = safe_num(d[score_col]) if score_col else np.nan
    d["_taxa"] = normalize_rate(d[taxa_col]) if taxa_col else np.nan
    d["_val"] = safe_num(d[val_col]) if val_col else np.nan

    g = (
        d.groupby(dim, dropna=False)
        .agg(
            N=("proposal_id", "nunique"),
            aprov=("is_approved", "mean"),
            contr=("is_contracted", "mean"),
            score_medio=("_score", "mean"),
            taxa_media=("_taxa", "mean"),
            valor_total=("_val", "sum"),
            paid_p1=("paid_p1", "mean"),
            incident=("is_incident", "mean"),
            over90=("over90", "mean"),
        )
        .reset_index()
    )

    g["approval_rate_pct"] = (g["aprov"] * 100).round(2)
    g["conv_e2e_pct"] = (g["contr"] * 100).round(2)

    approved_count = (g["aprov"] * g["N"]).replace(0, np.nan)
    contracted_count = (g["contr"] * g["N"])
    g["conv_approved_pct"] = ((contracted_count / approved_count) * 100).round(2)

    g["paid_p1_pct"] = (g["paid_p1"] * 100).round(2)
    g["incident_pct"] = (g["incident"] * 100).round(2)
    g["over90_pct"] = (g["over90"] * 100).round(2)
    g["ticket_medio"] = g["valor_total"] / g["N"].replace(0, np.nan)

    return g.drop(columns=["aprov", "contr", "paid_p1", "incident", "over90"], errors="ignore")


def agg_monthly_funnel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g = (
        d.groupby("base_month", dropna=False)
        .agg(
            N=("proposal_id", "nunique"),
            aprov=("is_approved", "mean"),
            contr=("is_contracted", "mean"),
        )
        .reset_index()
        .sort_values("base_month")
    )
    g["approval_rate_pct"] = (g["aprov"] * 100).round(2)
    g["conv_e2e_pct"] = (g["contr"] * 100).round(2)

    approved_count = (g["aprov"] * g["N"]).replace(0, np.nan)
    contracted_count = (g["contr"] * g["N"])
    g["conv_approved_pct"] = ((contracted_count / approved_count) * 100).round(2)

    return g.drop(columns=["aprov", "contr"], errors="ignore")


# =========================================================
# Pages
# =========================================================
def page_geral(mp_f: pd.DataFrame, k: Dict[str, Any]) -> None:
    st.title("PM Analytics")
    st.caption("Visão geral do período selecionado.")

    section("Indicadores principais")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Propostas", f"{k['propostas']:,}".replace(",", "."))
    c2.metric("Valor total emprestado", money_br(k["valor_total"]))
    c3.metric("Ticket médio", money_br(k["ticket_medio"]))
    c4.metric("Taxa média", f"{k['taxa_media']:.4f}" if not np.isnan(k["taxa_media"]) else "-")
    c5.metric("Score médio", f"{k['score_medio']:.1f}" if not np.isnan(k["score_medio"]) else "-")
    end_section()

    section("Qualidade e risco (proxies)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Paid P1", pct_fmt(k["paid_p1_pct"]))
    d2.metric("Incidentes", pct_fmt(k["incident_pct"]))
    d3.metric("Over30", pct_fmt(k["over30_pct"]))
    d4.metric("Over90", pct_fmt(k["over90_pct"]))
    end_section()

    section("Volumetria por mês")
    val_col = k["val_col_used"]
    val_series = safe_num(mp_f[val_col]) if val_col else pd.Series([np.nan] * len(mp_f))

    g = (
        mp_f.assign(_val=val_series)
        .groupby("base_month", dropna=False)
        .agg(propostas=("proposal_id", "nunique"), valor_total=("_val", "sum"))
        .reset_index()
        .sort_values("base_month")
    )

    colA, colB = st.columns(2)
    with colA:
        fig1 = px.bar(g, x="base_month", y="propostas", title="Volume de propostas por mês")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.bar(g, x="base_month", y="valor_total", title="Valor total emprestado por mês")
        fig2.update_layout(yaxis_tickprefix="R$ ")
        st.plotly_chart(fig2, use_container_width=True)
    end_section()


def page_clinicas(mp_f: pd.DataFrame) -> None:
    st.title("Clínicas")
    st.caption("Ranking de desempenho e risco, sem repetição entre listas.")

    if "Clinica" not in mp_f.columns:
        st.error("Não encontrei a coluna Clinica no recorte atual.")
        st.stop()

    with st.sidebar:
        st.markdown("### Clínicas (configuração)")
        min_n = st.number_input("Volume mínimo por clínica (N propostas)", min_value=5, max_value=500, value=30, step=5)
        top_n = st.slider("Top N clínicas", min_value=5, max_value=50, value=15, step=5)

        st.markdown("Critério de risco (gatilhos)")
        th_inc = st.slider("Incidentes mínimo (%)", 0.0, 30.0, 2.0, 0.5)
        th_o30 = st.slider("Over30 mínimo (%)", 0.0, 50.0, 2.0, 0.5)
        th_o90 = st.slider("Over90 mínimo (%)", 0.0, 50.0, 1.0, 0.5)

    d = mp_f.copy()
    d["Clinica"] = d["Clinica"].astype("string").str.strip()
    d["clinic_key"] = d["Clinica"].map(clinic_key)

    val_col = pick_first_existing(d, ["[PAGO] Valor Emprestado", "contract_val_emprestado", "contract_val_contratado"])
    taxa_col = pick_first_existing(d, ["contract_taxa_juros", "[GLOBAL] Taxa de Juros", "Taxa de Juros"])
    score_col = pick_first_existing(d, ["Score Serasa", "[Serasa] Score", "Score"])

    d["_val"] = safe_num(d[val_col]) if val_col else np.nan
    d["_taxa"] = normalize_rate(d[taxa_col]) if taxa_col else np.nan
    d["_score"] = safe_num(d[score_col]) if score_col else np.nan

    def _pick_display_name(s: pd.Series) -> str:
        ss = s.dropna().astype(str).str.strip()
        if ss.empty:
            return "NA"
        vc = ss.value_counts()
        return str(vc.index[0])

    tbl = (
        d.groupby("clinic_key", dropna=False)
        .agg(
            Clinica=("Clinica", _pick_display_name),
            N=("proposal_id", "nunique"),
            valor_total=("_val", "sum"),
            taxa_media=("_taxa", "mean"),
            score_medio=("_score", "mean"),
            paid_p1=("paid_p1", "mean"),
            incident=("is_incident", "mean"),
            over30=("over30", "mean"),
            over90=("over90", "mean"),
        )
        .reset_index()
    )

    tbl["ticket_medio"] = tbl["valor_total"] / tbl["N"].replace(0, np.nan)
    tbl["paid_p1_pct"] = (tbl["paid_p1"] * 100).round(2)
    tbl["incident_pct"] = (tbl["incident"] * 100).round(2)
    tbl["over30_pct"] = (tbl["over30"] * 100).round(2)
    tbl["over90_pct"] = (tbl["over90"] * 100).round(2)

    w = np.log1p(tbl["N"].clip(lower=1))
    tbl["performance_score"] = (tbl["paid_p1_pct"] - tbl["incident_pct"] - tbl["over90_pct"])
    tbl["performance_score_final"] = (tbl["performance_score"] * w).round(3)

    tbl["risk_score"] = (tbl["incident_pct"] + tbl["over30_pct"] + tbl["over90_pct"])
    tbl["risk_score_final"] = (tbl["risk_score"] * w).round(3)

    tbl2 = tbl[tbl["N"] >= int(min_n)].copy()
    if tbl2.empty:
        st.warning("Nenhuma clínica passa o volume mínimo com os filtros atuais.")
        st.dataframe(tbl.sort_values("N", ascending=False), use_container_width=True)
        st.stop()

    section("Top clínicas por desempenho")
    top_perf = tbl2.sort_values("performance_score_final", ascending=False).head(int(top_n)).copy()
    cols_show = [
        "Clinica",
        "N",
        "valor_total",
        "ticket_medio",
        "taxa_media",
        "score_medio",
        "paid_p1_pct",
        "incident_pct",
        "over30_pct",
        "over90_pct",
        "performance_score_final",
    ]
    st.dataframe(top_perf[cols_show], use_container_width=True)
    grouped_bar(top_perf, x="Clinica", value_cols=["paid_p1_pct", "incident_pct", "over90_pct"], title="Métricas-chave (%) — desempenho")
    end_section()

    perf_keys = set(top_perf["clinic_key"].astype(str).tolist())
    risk_mask = (
        (tbl2["incident_pct"].fillna(0) >= float(th_inc))
        | (tbl2["over30_pct"].fillna(0) >= float(th_o30))
        | (tbl2["over90_pct"].fillna(0) >= float(th_o90))
    )
    candidates_risk = tbl2[(~tbl2["clinic_key"].astype(str).isin(perf_keys)) & (risk_mask)].copy()

    section("Top clínicas por risco (sem repetir top desempenho)")
    if candidates_risk.empty:
        st.warning("Nenhuma clínica atende ao critério de risco e não está no top desempenho dentro do recorte atual.")
        end_section()
        return

    top_risk = candidates_risk.sort_values("risk_score_final", ascending=False).head(int(top_n)).copy()
    cols_show2 = [
        "Clinica",
        "N",
        "valor_total",
        "ticket_medio",
        "taxa_media",
        "score_medio",
        "paid_p1_pct",
        "incident_pct",
        "over30_pct",
        "over90_pct",
        "risk_score_final",
    ]
    st.dataframe(top_risk[cols_show2], use_container_width=True)
    grouped_bar(top_risk, x="Clinica", value_cols=["incident_pct", "over30_pct", "over90_pct"], title="Métricas-chave (%) — risco")
    end_section()


def page_empregos(mp_f: pd.DataFrame) -> None:
    st.title("Empregos")
    st.caption("Diagnóstico por macro-categoria de emprego (proxy de risco e funil).")

    if "employment_macro" not in mp_f.columns:
        st.error("Não existe employment_macro no dataset atual.")
        st.stop()

    with st.sidebar:
        st.markdown("### Empregos (configuração)")
        min_n = st.number_input("Volume mínimo por emprego (N propostas)", min_value=10, max_value=2000, value=80, step=10)
        top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)

    tbl = agg_funnel(mp_f, dim="employment_macro")
    tbl2 = tbl[tbl["N"] >= int(min_n)].copy()

    if tbl2.empty:
        st.warning("Nenhum emprego passa o volume mínimo com os filtros atuais.")
        st.dataframe(tbl.sort_values("N", ascending=False), use_container_width=True)
        st.stop()

    section("Top empregos por Over90 (proxy de risco)")
    top_risk = tbl2.sort_values("over90_pct", ascending=False).head(int(top_n)).copy()
    st.dataframe(
        top_risk[["employment_macro", "N", "approval_rate_pct", "conv_approved_pct", "conv_e2e_pct", "incident_pct", "over90_pct", "paid_p1_pct"]],
        use_container_width=True,
    )
    grouped_bar(top_risk, x="employment_macro", value_cols=["incident_pct", "over90_pct", "paid_p1_pct"], title="Métricas-chave (%) — empregos")
    end_section()


def page_politica_score_taxa(mp_f: pd.DataFrame) -> None:
    st.title("Score, Taxa e Política")
    st.caption("Aprovação, conversão e risco por Política, buckets de Score e buckets de Taxa.")

    with st.sidebar:
        st.markdown("### Score / Taxa / Política (configuração)")
        min_n = st.number_input("Volume mínimo por grupo (N propostas)", min_value=10, max_value=5000, value=100, step=10)
        top_n = st.slider("Top N (tabelas)", min_value=5, max_value=50, value=15, step=5)
        metric = st.selectbox(
            "Métrica principal",
            ["Approval rate (%)", "Conversão de aprovados (%)", "Conversão ponta a ponta (%)"],
            index=2,
        )

    metric_map = {
        "Approval rate (%)": "approval_rate_pct",
        "Conversão de aprovados (%)": "conv_approved_pct",
        "Conversão ponta a ponta (%)": "conv_e2e_pct",
    }
    metric_col = metric_map[metric]

    n = int(mp_f["proposal_id"].nunique())
    appr = float(mp_f["is_approved"].mean() * 100) if "is_approved" in mp_f.columns else np.nan
    e2e = float(mp_f["is_contracted"].mean() * 100) if "is_contracted" in mp_f.columns else np.nan
    appr_count = float(mp_f["is_approved"].sum()) if "is_approved" in mp_f.columns else 0.0
    contr_count = float(mp_f["is_contracted"].sum()) if "is_contracted" in mp_f.columns else 0.0
    conv_appr = float((contr_count / appr_count) * 100) if appr_count > 0 else np.nan

    section("Resumo do funil no período")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Propostas", f"{n:,}".replace(",", "."))
    a2.metric("Approval rate", pct_fmt(appr))
    a3.metric("Conv. aprovados → contrato", pct_fmt(conv_appr))
    a4.metric("Conv. ponta a ponta", pct_fmt(e2e))
    end_section()

    section("Tendência mensal")
    monthly = agg_monthly_funnel(mp_f)
    if monthly.empty:
        st.info("Sem dados suficientes para tendência mensal.")
    else:
        fig = px.line(
            monthly,
            x="base_month",
            y=["approval_rate_pct", "conv_approved_pct", "conv_e2e_pct"],
            markers=True,
            title="Approval / Conv. aprovados / Conv. ponta a ponta",
        )
        fig.update_layout(yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(monthly, use_container_width=True)
    end_section()

    section("Por Política")
    pol_nonempty = mp_f["Política"].notna().sum() if "Política" in mp_f.columns else 0
    if pol_nonempty == 0:
        st.warning("A coluna Política está vazia no recorte atual. Quando houver preenchimento, esta seção se populá automaticamente.")
        end_section()
    else:
        pol = agg_funnel(mp_f, dim="Política")
        pol2 = pol[pol["N"] >= int(min_n)].copy()
        if pol2.empty:
            st.info("Sem volume mínimo por Política com os filtros atuais.")
        else:
            pol2 = pol2.sort_values(metric_col, ascending=False).head(int(top_n))
            st.dataframe(
                pol2[["Política", "N", "approval_rate_pct", "conv_approved_pct", "conv_e2e_pct", "score_medio", "taxa_media", "paid_p1_pct", "incident_pct", "over90_pct"]],
                use_container_width=True,
            )
            figp = px.bar(pol2, x="Política", y=metric_col, title=f"{metric} por Política")
            figp.update_layout(xaxis_tickangle=-35, yaxis_title="%")
            st.plotly_chart(figp, use_container_width=True)
        end_section()

    section("Por Score bucket")
    sb = agg_funnel(mp_f, dim="score_bucket")
    sb2 = sb[(sb["N"] >= int(min_n)) & (sb["score_bucket"].notna())].copy()
    if sb2.empty:
        st.info("Sem volume mínimo por score_bucket com os filtros atuais (ou score indisponível).")
    else:
        order = ["<=300", "301-500", "501-650", "651-750", "751+"]
        sb2["_ord"] = sb2["score_bucket"].astype(str).map(lambda x: order.index(x) if x in order else 999)
        sb2 = sb2.sort_values("_ord").drop(columns=["_ord"]).head(50)
        st.dataframe(
            sb2[["score_bucket", "N", "approval_rate_pct", "conv_approved_pct", "conv_e2e_pct", "taxa_media", "paid_p1_pct", "incident_pct", "over90_pct"]],
            use_container_width=True,
        )
        figsb = px.line(sb2, x="score_bucket", y=metric_col, markers=True, title=f"{metric} por Score bucket")
        figsb.update_layout(yaxis_title="%")
        st.plotly_chart(figsb, use_container_width=True)
    end_section()

    section("Por Taxa bucket")
    tb = agg_funnel(mp_f, dim="taxa_bucket")
    tb2 = tb[(tb["N"] >= int(min_n)) & (tb["taxa_bucket"].notna())].copy()
    if tb2.empty:
        st.info("Sem volume mínimo por taxa_bucket com os filtros atuais (ou taxa indisponível).")
    else:
        order = ["<=5%", "5-8%", "8-12%", "12-18%", "18%+"]
        tb2["_ord"] = tb2["taxa_bucket"].astype(str).map(lambda x: order.index(x) if x in order else 999)
        tb2 = tb2.sort_values("_ord").drop(columns=["_ord"]).head(50)
        st.dataframe(
            tb2[["taxa_bucket", "N", "approval_rate_pct", "conv_approved_pct", "conv_e2e_pct", "score_medio", "paid_p1_pct", "incident_pct", "over90_pct"]],
            use_container_width=True,
        )
        figtb = px.line(tb2, x="taxa_bucket", y=metric_col, markers=True, title=f"{metric} por Taxa bucket")
        figtb.update_layout(yaxis_title="%")
        st.plotly_chart(figtb, use_container_width=True)
    end_section()

    with st.expander("Diagnóstico de colunas"):
        keys = ["Política", "score_bucket", "taxa_bucket", "is_approved", "is_contracted"]
        rows = []
        for c in keys:
            if c not in mp_f.columns:
                rows.append({"coluna": c, "status": "não existe"})
            else:
                miss = float(mp_f[c].isna().mean() * 100)
                rows.append({"coluna": c, "status": f"ok — missing {miss:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_credit_features_table(mp_f: pd.DataFrame) -> None:
    st.title("Variáveis de Crédito")
    st.caption("Tabela com Score, PH3A e SCR (quando disponíveis).")

    col_candidates: Dict[str, List[str]] = {
        "PH3A - Pre Screening D00": ["PH3A - Pre Screening D00", "PH3A - Pre Screening D0", "PH3A - Pre-Screening D00"],
        "Score Serasa": ["Score Serasa", "[Serasa] Score", "Score Serasa "],
        "SCR - Valor Médio Desembolso (31-90)": ["SCR - Valor Médio Desembolso (31-90)", "SCR - Valor Medio Desembolso (31-90)"],
        "SCR - Total Prejuízo": ["SCR - Total Prejuízo", "SCR - Total Prejuizo"],
        "SCR - Total Vencido": ["SCR - Total Vencido"],
        "SCR - Qtd. Operações": ["SCR - Qtd. Operações", "SCR - Qtd. Operacoes"],
        "SCR - Qtd Insituições": ["SCR - Qtd Insituições", "SCR - Qtd Instituições", "SCR - Qtd. Instituições", "SCR - Qtd. Insituições"],
        "SCR - Carteira de Crédito": ["SCR - Carteira de Crédito", "SCR - Carteira de Credito"],
        "SCR - Créditos a Liberar": ["SCR - Creditos a Liberar", "SCR - Créditos a Liberar", "SCR - Creditos a liberar"],
        "PH3A - Renda Familiar": ["PH3A - Renda Familiar"],
        "PH3A - Renda Presumida": ["PH3A - Renda Presumida"],
    }

    desired_order = list(col_candidates.keys())

    d = pd.DataFrame(index=mp_f.index)
    missing = []
    source_used: Dict[str, Optional[str]] = {}

    for canon, cands in col_candidates.items():
        src = pick_first_existing(mp_f, cands)
        source_used[canon] = src
        if src is None:
            d[canon] = pd.NA
            missing.append(canon)
        else:
            d[canon] = mp_f[src]

    with st.sidebar:
        st.markdown("### Variáveis de crédito (configuração)")
        max_rows = st.slider("Máx. linhas na tabela", min_value=200, max_value=20000, value=3000, step=200)
        show_missing_diag = st.checkbox("Mostrar diagnóstico de colunas", value=True)

    for c in desired_order:
        if c in d.columns:
            d[c] = safe_num(d[c])

    if show_missing_diag:
        with st.expander("Diagnóstico (origem das colunas e missing)"):
            rows = []
            for canon in desired_order:
                rows.append(
                    {
                        "coluna_canônica": canon,
                        "coluna_origem": source_used.get(canon) or "NÃO ENCONTRADA",
                        "missing_%": float(d[canon].isna().mean() * 100) if canon in d.columns else np.nan,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if missing:
        st.warning("Algumas colunas não existem no arquivo atual e foram preenchidas como vazias: " + ", ".join(missing))

    d_out = d[desired_order].copy()

    if "Score Serasa" in d_out.columns and d_out["Score Serasa"].notna().any():
        d_out = d_out.sort_values("Score Serasa", ascending=False)

    if len(d_out) > int(max_rows):
        st.info(f"Mostrando apenas as primeiras {int(max_rows):,} linhas (de {len(d_out):,}).".replace(",", "."))
        d_out = d_out.head(int(max_rows))

    st.dataframe(d_out, use_container_width=True)

    csv = d_out.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV", data=csv, file_name="variaveis_credito.csv", mime="text/csv")


# =========================================================
# Main
# =========================================================
inject_css()
ensure_data_files()
mp, mi = load_data()

# Sidebar: navegação
st.sidebar.markdown("---")
st.sidebar.markdown("### Navegação")
page = st.sidebar.radio(
    "Página",
    ["Geral", "Clínicas", "Empregos", "Score/Taxa/Política", "Variáveis de Crédito"],
    index=0,
)

# Sidebar: filtro de período (global)
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros globais")

available_months = sort_base_months(mp["base_month"].dropna().astype(str).unique().tolist())

preset = st.sidebar.selectbox(
    "Atalho de período",
    ["Últimos 3 meses", "Últimos 6 meses", "Últimos 12 meses", "Tudo", "Custom"],
    index=2,
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
    default_months = available_months[-12:]

sel_months = st.sidebar.multiselect("Meses (base_month)", options=available_months, default=default_months)

if not sel_months:
    st.warning("Selecione pelo menos 1 mês para análise.")
    st.stop()

mp_f = mp[mp["base_month"].astype(str).isin(sel_months)].copy()

# Enriquecimentos base (para todas as páginas)
mp_f = ensure_incident_flag(mp_f)
mp_f = ensure_employment_macro(mp_f)
mp_f = ensure_policy_col(mp_f)
mp_f = ensure_score_taxa_buckets(mp_f)
mp_f = ensure_funnel_flags(mp_f)

# Filtra installments para proposals do recorte
if not mi.empty and "proposal_id" in mi.columns:
    ids = set(mp_f["proposal_id"].astype(str).unique())
    mi_f = mi[mi["proposal_id"].astype(str).isin(ids)].copy()
else:
    mi_f = mi.copy()

# paid_p1 + overdue (para todas)
p1 = build_paid_p1_flag(mi_f)
mp_f = mp_f.merge(p1, on="proposal_id", how="left")
mp_f["paid_p1"] = mp_f["paid_p1"].fillna(False).astype(bool)

od = build_overdue_flags(mi_f, ref_date=pd.Timestamp.today().normalize())
mp_f = mp_f.merge(od, on="proposal_id", how="left")
mp_f["over30"] = mp_f["over30"].fillna(False).astype(bool)
mp_f["over90"] = mp_f["over90"].fillna(False).astype(bool)

# Render
if page == "Geral":
    k = compute_kpis(mp_f)
    page_geral(mp_f, k)
elif page == "Clínicas":
    page_clinicas(mp_f)
elif page == "Empregos":
    page_empregos(mp_f)
elif page == "Score/Taxa/Política":
    page_politica_score_taxa(mp_f)
else:
    page_credit_features_table(mp_f)
