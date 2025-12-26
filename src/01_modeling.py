import re
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_TABLES = PROJECT_ROOT / "reports" / "tables"

REPORTS_TABLES.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def read_table(path_no_ext: Path):
    """
    Lê parquet se existir, senão csv.
    Ex.: read_table(DATA_PROCESSED / "tabela1_clean")
    """
    parquet_path = Path(str(path_no_ext) + ".parquet")
    csv_path = Path(str(path_no_ext) + ".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=str)
    raise FileNotFoundError(f"Não achei {parquet_path} nem {csv_path}")

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def safe_str(s):
    return s.astype("string")

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Load (tabelas já tratadas do Passo 1)
# -------------------------
t1 = read_table(DATA_PROCESSED / "tabela1_clean")
t2 = read_table(DATA_PROCESSED / "tabela2_clean")

# Colunas importantes
base_date_col = "[PAGO] Data do Pagamento"  # regra do projeto
t2_payment_date_col = "[PAGO] Data Pagamento"

# Chaves
proposal_col = pick_first_existing(t1, ["proposal", "unique id", "CPF Proposta"])
if proposal_col is None:
    raise ValueError("Não encontrei coluna chave em t1 (tente proposal/unique id/CPF Proposta).")

t2_key_col = pick_first_existing(t2, ["unique id"])
if t2_key_col is None:
    raise ValueError("Não encontrei 'unique id' na tabela 2.")

# DT|Payment 1..6
dt_cols = [c for c in t1.columns if re.fullmatch(r"DT\|Payment [1-6]", str(c))]
dt_cols = sorted(dt_cols, key=lambda x: int(x.split()[-1]))

if len(dt_cols) == 0:
    raise ValueError("Não encontrei colunas DT|Payment 1..6 na tabela 1.")

# Tipos essenciais (garantir datetime e numéricos se já não vierem)
if base_date_col in t1.columns:
    t1[base_date_col] = safe_to_datetime(t1[base_date_col])

if t2_payment_date_col in t2.columns:
    t2[t2_payment_date_col] = safe_to_datetime(t2[t2_payment_date_col])

# -------------------------
# 1) master_proposals (1 linha por proposta)
# -------------------------
master_proposals = t1.copy()

master_proposals["base_date"] = master_proposals[base_date_col] if base_date_col in master_proposals.columns else pd.NaT
master_proposals["base_month"] = master_proposals["base_date"].dt.to_period("M").astype("string")
master_proposals["is_paid_base"] = master_proposals["base_date"].notna()

# Normaliza chave padrão "proposal_id" para uso no restante
master_proposals["proposal_id"] = safe_str(master_proposals[proposal_col])

# -------------------------
# 2) bridge_payments (explode DT|Payment 1..6 -> long)
# -------------------------
bridge = (
    t1[[proposal_col] + dt_cols].copy()
)
bridge["proposal_id"] = safe_str(bridge[proposal_col])

bridge_long = bridge.melt(
    id_vars=["proposal_id"],
    value_vars=dt_cols,
    var_name="dt_payment_col",
    value_name="payment_unique_id",
)

bridge_long["payment_unique_id"] = safe_str(bridge_long["payment_unique_id"]).str.strip()
bridge_long = bridge_long[bridge_long["payment_unique_id"].notna() & (bridge_long["payment_unique_id"] != "")].copy()

bridge_long["parcela_n"] = bridge_long["dt_payment_col"].str.extract(r"(\d+)").astype(int)

# -------------------------
# 3) master_installments (join com tabela 2)
# -------------------------
t2_join = t2.copy()
t2_join["payment_unique_id"] = safe_str(t2_join[t2_key_col]).str.strip()

master_installments = bridge_long.merge(
    t2_join,
    on="payment_unique_id",
    how="left",
    suffixes=("", "_t2"),
)

# herda base_month da proposta (pra sempre analisar “por safra base”)
master_installments = master_installments.merge(
    master_proposals[["proposal_id", "base_month", "base_date"]],
    on="proposal_id",
    how="left",
)

# Derivadas úteis
if "Status" in master_installments.columns:
    master_installments["is_paid_status"] = safe_str(master_installments["Status"]).str.lower().eq("pago")
else:
    master_installments["is_paid_status"] = pd.NA

if t2_payment_date_col in master_installments.columns:
    master_installments["payment_date_real"] = master_installments[t2_payment_date_col]
    master_installments["payment_month_real"] = master_installments["payment_date_real"].dt.to_period("M").astype("string")
else:
    master_installments["payment_date_real"] = pd.NaT
    master_installments["payment_month_real"] = pd.NA

# -------------------------
# 4) Quality checks do join
# -------------------------
qc = []

# Unicidade do unique id na tabela 2
dup_t2 = t2_join.duplicated(subset=["payment_unique_id"]).sum()
qc.append({"check": "t2_duplicate_unique_id", "value": int(dup_t2)})

# Match rate geral
n_bridge = len(bridge_long)
n_matched = master_installments["payment_unique_id"].notna().sum()  # sempre notna; melhor checar coluna de algum campo da t2
# usamos a própria data/status como proxy de match
match_proxy_col = t2_payment_date_col if t2_payment_date_col in master_installments.columns else "Status"
n_matched_proxy = master_installments[match_proxy_col].notna().sum()
qc.append({"check": "bridge_rows", "value": int(n_bridge)})
qc.append({"check": "matched_rows_proxy", "value": int(n_matched_proxy)})
qc.append({"check": "match_rate_proxy_pct", "value": float(round((n_matched_proxy / n_bridge) * 100, 2)) if n_bridge else np.nan})

# Match rate por parcela_n
mr_by_parcela = (
    master_installments.groupby("parcela_n")[match_proxy_col]
    .apply(lambda s: s.notna().mean() * 100)
    .reset_index(name="match_rate_pct")
)
mr_by_parcela["match_rate_pct"] = mr_by_parcela["match_rate_pct"].round(2)

# Propostas distintas nas duas tabelas finais
qc.append({"check": "unique_proposals_master_proposals", "value": int(master_proposals["proposal_id"].nunique())})
qc.append({"check": "unique_proposals_master_installments", "value": int(master_installments["proposal_id"].nunique())})

qc_df = pd.DataFrame(qc)

# -------------------------
# 5) Save outputs
# -------------------------
bridge_long.to_parquet(DATA_PROCESSED / "bridge_payments.parquet", index=False)
master_proposals.to_parquet(DATA_PROCESSED / "master_proposals.parquet", index=False)
master_installments.to_parquet(DATA_PROCESSED / "master_installments.parquet", index=False)

# (opcional CSV)
# bridge_long.to_csv(DATA_PROCESSED / "bridge_payments.csv", index=False)
# master_proposals.to_csv(DATA_PROCESSED / "master_proposals.csv", index=False)
# master_installments.to_csv(DATA_PROCESSED / "master_installments.csv", index=False)

qc_df.to_csv(REPORTS_TABLES / "modeling_qc_summary.csv", index=False)
mr_by_parcela.to_csv(REPORTS_TABLES / "modeling_match_rate_by_parcela.csv", index=False)

print("✅ Modelagem concluída.")
print(qc_df)
print("\nMatch rate por parcela:")
print(mr_by_parcela)
