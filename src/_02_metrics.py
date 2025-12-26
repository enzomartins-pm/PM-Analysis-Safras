from __future__ import annotations

import numpy as np
import pandas as pd

_BOOL_MAP = {
    "sim": True, "s": True, "yes": True, "y": True, "true": True, "1": True,
    "não": False, "nao": False, "n": False, "no": False, "false": False, "0": False,
}

def norm_bool_sim_nao(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="boolean")
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    return (
        s.astype("string")
         .str.strip()
         .str.lower()
         .map(_BOOL_MAP)
         .astype("boolean")
    )

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def add_base_fields(master_proposals: pd.DataFrame) -> pd.DataFrame:
    out = master_proposals.copy()

    if "proposal_id" not in out.columns or out["proposal_id"].isna().all():
        for c in ["proposal", "unique id", "CPF Proposta"]:
            if c in out.columns:
                out["proposal_id"] = out[c].astype("string")
                break
    else:
        out["proposal_id"] = out["proposal_id"].astype("string")

    if "base_date" not in out.columns:
        if "[PAGO] Data do Pagamento" in out.columns:
            out["base_date"] = pd.to_datetime(out["[PAGO] Data do Pagamento"], errors="coerce")
        else:
            out["base_date"] = pd.NaT
    else:
        out["base_date"] = pd.to_datetime(out["base_date"], errors="coerce")

    if "base_month" not in out.columns:
        out["base_month"] = out["base_date"].dt.to_period("M").astype("string")
    else:
        out["base_month"] = out["base_month"].astype("string")

    if "is_paid_base" not in out.columns:
        out["is_paid_base"] = out["base_date"].notna()
    else:
        out["is_paid_base"] = out["is_paid_base"].astype("boolean")

    return out

def add_incident_metrics(master_proposals: pd.DataFrame) -> pd.DataFrame:
    out = master_proposals.copy()
    scr_err = norm_bool_sim_nao(out["Consulta scr erro?"]) if "Consulta scr erro?" in out.columns else pd.Series([False]*len(out))
    srm_err = norm_bool_sim_nao(out["consulta srm filme erro?"]) if "consulta srm filme erro?" in out.columns else pd.Series([False]*len(out))
    out["is_incident"] = (scr_err.fillna(False) | srm_err.fillna(False)).astype("boolean")
    return out

def add_rebate_metrics(master_proposals: pd.DataFrame) -> pd.DataFrame:
    out = master_proposals.copy()
    has_pol_rebate = out["Política Rebate"].notna() if "Política Rebate" in out.columns else pd.Series([False]*len(out))
    has_rate_rebate = out["Taxa de Juros Rebate"].notna() if "Taxa de Juros Rebate" in out.columns else pd.Series([False]*len(out))
    out["has_rebate"] = (has_pol_rebate | has_rate_rebate).astype("boolean")

    if "Taxa de Juros" in out.columns and "Taxa de Juros Rebate" in out.columns:
        out["rebate_spread"] = safe_numeric(out["Taxa de Juros"]) - safe_numeric(out["Taxa de Juros Rebate"])
    else:
        out["rebate_spread"] = np.nan
    return out

def add_buckets(master_proposals: pd.DataFrame) -> pd.DataFrame:
    out = master_proposals.copy()

    if "Score Serasa" in out.columns:
        score = safe_numeric(out["Score Serasa"])
        out["score_bucket"] = pd.cut(
            score,
            bins=[-np.inf, 300, 500, 650, 750, np.inf],
            labels=["<=300", "301-500", "501-650", "651-750", "751+"],
        ).astype("string")
    else:
        out["score_bucket"] = pd.NA

    if "Taxa de Juros" in out.columns:
        taxa = safe_numeric(out["Taxa de Juros"])
        # normaliza % vs decimal
        taxa_dec = taxa / 100.0 if (taxa.dropna() > 1).mean() > 0.8 else taxa
        out["taxa_bucket"] = pd.cut(
            taxa_dec,
            bins=[-np.inf, 0.05, 0.08, 0.12, 0.18, np.inf],
            labels=["<=5%", "5-8%", "8-12%", "12-18%", "18%+"],
        ).astype("string")
    else:
        out["taxa_bucket"] = pd.NA

    return out

def apply_all(master_proposals: pd.DataFrame) -> pd.DataFrame:
    out = add_base_fields(master_proposals)
    out = add_incident_metrics(out)
    out = add_rebate_metrics(out)
    out = add_buckets(out)
    return out
