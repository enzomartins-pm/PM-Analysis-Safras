import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

PT_MONTHS = {
    "jan": "Jan", "fev": "Feb", "mar": "Mar", "abr": "Apr", "mai": "May", "jun": "Jun",
    "jul": "Jul", "ago": "Aug", "set": "Sep", "out": "Oct", "nov": "Nov", "dez": "Dec",
}

def _normalize_months_pt_en(s: pd.Series) -> pd.Series:
    if s.dtype != "object":
        s = s.astype("object")
    out = s.fillna("").astype(str).str.strip()
    out = out.str.replace(
        r"\b(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\.\b",
        r"\1",
        regex=True,
        flags=re.IGNORECASE
    )
    for pt, en in PT_MONTHS.items():
        out = out.str.replace(rf"\b{pt}\b", en, regex=True, flags=re.IGNORECASE)
    out = out.replace({"": np.nan})
    return out

def parse_datetime_mixed(series: pd.Series) -> pd.Series:
    s = _normalize_months_pt_en(series)
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def to_number_pt(series: pd.Series) -> pd.Series:
    s = series.astype("object").where(series.notna(), None)
    s = pd.Series(s)

    def _conv(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        txt = str(x).strip()
        if txt == "" or txt.lower() in {"nan", "none"}:
            return np.nan
        txt = txt.replace(" ", "")
        if "," in txt and "." in txt:
            txt = txt.replace(".", "").replace(",", ".")
        elif "," in txt:
            txt = txt.replace(",", ".")
        try:
            return float(txt)
        except:
            return np.nan

    return s.map(_conv)

def to_bool_sim_nao(series: pd.Series) -> pd.Series:
    mapping = {
        "sim": True, "s": True, "yes": True, "y": True, "true": True, "1": True,
        "não": False, "nao": False, "n": False, "no": False, "false": False, "0": False
    }
    s = series.astype("object")
    s = s.where(series.notna(), None)

    def _map(x):
        if x is None:
            return np.nan
        txt = str(x).strip().lower()
        if txt == "":
            return np.nan
        return mapping.get(txt, np.nan)

    return s.map(_map)

def main():
    # <<< ajuste aqui se quiser fixar nomes, mas dá pra deixar hardcoded igual seu notebook >>>
    t1_path = DATA_RAW / "export_-JO-O--An-lise-Full_2025-12-16_16-57-32.csv"
    t2_path = DATA_RAW / "export_View---Vencimentos-CESKGPF_2025-12-16_11-36-56 - export_View---Vencimentos-CESKGPF_2025-12-16_11-36-56.csv"

    t1 = pd.read_csv(t1_path, dtype=str, low_memory=False)
    t2 = pd.read_csv(t2_path, dtype=str, low_memory=False)

    # ---- Datas principais
    for c in ["[PAGO] Data do Pagamento", "Data Analise/Proposta", "Creation Date", "Modified Date"]:
        if c in t1.columns:
            t1[c] = parse_datetime_mixed(t1[c])

    for c in ["[PAGO] Data Pagamento", "Creation Date", "Data de vencimento"]:
        if c in t2.columns:
            t2[c] = parse_datetime_mixed(t2[c])

    # ---- Números principais
    for c in ["[PAGO] Valor Emprestado", "Taxa de Juros", "Taxa de Juros Rebate", "Score Serasa"]:
        if c in t1.columns:
            t1[c] = to_number_pt(t1[c])

    for c in ["[PAGO] Valor Pago", "Valor Parcela", "Valor total"]:
        if c in t2.columns:
            t2[c] = to_number_pt(t2[c])

    # ---- Booleans (se existirem)
    for c in ["Antecipada?", "Em espera?"]:
        if c in t2.columns:
            t2[c] = to_bool_sim_nao(t2[c])

    for c in ["[OTIMA] is_subscriber?", "PH3A - Empregado?", "Consulta scr erro?", "consulta srm filme erro?"]:
        if c in t1.columns:
            t1[c] = to_bool_sim_nao(t1[c])

    # ---- Trim em IDs críticos
    for c in [col for col in t1.columns if str(col).startswith("DT|Payment ")]:
        t1[c] = t1[c].astype("string").str.strip()

    if "unique id" in t2.columns:
        t2["unique id"] = t2["unique id"].astype("string").str.strip()

    # ---- Salvar CANÔNICO (isso destrava o 01_modeling)
    out1 = DATA_PROCESSED / "tabela1_clean.parquet"
    out2 = DATA_PROCESSED / "tabela2_clean.parquet"
    t1.to_parquet(out1, index=False)
    t2.to_parquet(out2, index=False)

    print("✅ Criados:")
    print("-", out1)
    print("-", out2)
    print("Linhas T1:", len(t1), "| Linhas T2:", len(t2))

if __name__ == "__main__":
    main()
