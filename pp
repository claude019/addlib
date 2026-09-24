# =============================================================================
# ALERT -> POLICY  **ROUTED** LINKAGE   (RSM cascade, cost-tracked, resumable)
#
# THE STORY THIS PIPELINE TELLS:
#   Each alert is judged against the policies that sit under its LINKED
#   Regulation Summaries. Same shape as the Library-Risk routed lane; only the
#   parentage source and the object being judged change.
#
#     LR lane   : record -> LINKED RSM -> (Extract3)              -> Library Risk
#     THIS lane : record -> LINKED RSM -> (IA_RSM_POLICY_Linkage) -> Policy
#
#   Golden truth is built the same way as the LR golden: the golden dataset is
#   record -> REGULATION, so composing it with the RSM->Policy table at
#   REGULATION level reproduces the familiar structural over-mapping. It is
#   labelled as such in [B2] and quantified in [E]/[F] -- do not present the
#   naive recall number without that carve-out.
#
# CARRIED OVER UNCHANGED from the LR routed lane:
#   _read_any / _find / _norm_id / load_alerts, judge machinery (chunk,
#   semaphore, 401 refresh, ERROR rows), scorecard framing, [E], [F] audit,
#   [F1] honest recall, provenance check, performance pack.
#
# NEW HERE:
#   [0b] COST LEDGER -- every LLM + embedding call, EVERY attempt, on disk.
#        Judge AND audit both route through chat_with_cost, so the team cost
#        file now covers the whole run, not just embeddings.
#   [A]  load_policies (PUBL only, newest transfer per policy, HTML entities
#        decoded, long body truncated) + load_policy_links (RSM -> Policy).
#   [C]  checkpointed execution: plan.json + fsync'd JSONL + parquet snapshots,
#        so a VDI disconnect costs nothing already paid for.
#
# SECTIONS
#   0.  Imports + paths + run id        0b. Cost ledger
#   A.  Data load                       A2. Validation probe
#   B1. Config + model params           P.  Provenance fingerprint
#   B2. Parentage + funnel + reachability + cost projection
#   C.  Work plan + checkpointed model execution
#   D.  Scorecard (recall / data gap / discoveries) + ranking
#   E.  Diagnostic                      F.  LLM-as-judge audit
#   F1. Honest recall                   G.  Cost report -> the cost file
#   H.  Performance pack (xlsx)         P-after. Second provenance check
# =============================================================================


# %% ==========================================================================
# [0] IMPORTS + PATHS
# =============================================================================
import os, re, json, time, html, math, hashlib, asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from tqdm.auto import tqdm

import nest_asyncio                 # embed_with_cache calls run_until_complete
nest_asyncio.apply()                # -> must patch the Jupyter loop first

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

# --- Gateway objects already defined upstream (do NOT recreate) --------------
oai         = openai_client
tok         = trust_token
MODEL       = CHAT_MODEL
clean       = clean_text
embed_cache = embed_with_cache       # (ids, texts, cache_tag) -> L2-normalised matrix
USE_CASE    = globals().get("USE_CASE", "alert_policy_linkage")
EMBED_MODEL = globals().get("EMBED_MODEL", "gemini-embedding-001")   # cost attribution only

# --- Paths ------------------------------------------------------------------
DATA_DIR   = Path("/home/jovyan/data")
REGMAP_DIR = DATA_DIR / "Tactical_Regmap_Data"

# Everything this lane writes lives under PROJECT_OUTPUT_DIR (rename the leaf to
# policy_linkage_routed if you prefer -- the lane is routed, not direct).
PROJECT_OUTPUT_DIR   = Path("../output/execution_output/policy_linkage_direct")
EXECUTION_OUTPUT_DIR = PROJECT_OUTPUT_DIR / "Iteration_1"

POLICY_FILE      = DATA_DIR / "alert_gpps_cleaned.parquet"          # policy text, PUBL filter
POLICY_LINK_FILE = DATA_DIR / "IA_RSM_POLICY_Linkage.xlsx"          # RSM -> Policy  <- PARENTAGE
GOLDEN_FILE      = DATA_DIR / "golden_dataset_US_exploded_with_control_mapping_03sep26.parquet"
RAPID2_FILE      = REGMAP_DIR / "Rapid2_REGDEV_US_2020_2025.parquet"
RWT_FILE         = DATA_DIR / ("results_with_truth_gpt-5.6-luna_Prompt_V0_EMB_161_Cases_"
                               "RSM_to_RGL_AGG_Control_integration_20260905T133537Z.parquet")

POLICY_PROMPT_PATH = Path("/home/jovyan/prompt/policy_linkage/prompt_policy_linkage_v0.txt")

OUTPUT_DIR = Path(EXECUTION_OUTPUT_DIR); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR   = OUTPUT_DIR / "checkpoints";  CKPT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID      = "POL_routed_iter1_v0"
RUN_COMMENT = "Alert -> Policy ROUTED via LINKED RSMs, PUBL policies, prompt v0, top_k=None"

# --- Team cost file (same format as earlier runs) ----------------------------
# cost_log_<COST_RUN_ID>.parquet  cols: kind, in, out, ts, use_case, staff_id,
#                                       run_id, run_cost_usd
COST_DIR = PROJECT_OUTPUT_DIR / "cost"; COST_DIR.mkdir(parents=True, exist_ok=True)
STAFF_ID = globals().get("STAFF_ID", "<staff_id>")      # bind upstream or set here

# COST_RUN_ID is minted ONCE per execution folder and pinned in the checkpoint
# dir, so a resume after a disconnect keeps writing to the SAME cost file.
_COST_RUN_ID_FILE = CKPT_DIR / "cost_run_id.txt"
if _COST_RUN_ID_FILE.exists():
    COST_RUN_ID = _COST_RUN_ID_FILE.read_text().strip()
else:
    COST_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _COST_RUN_ID_FILE.write_text(COST_RUN_ID)
COST_PARQUET = COST_DIR / f"cost_log_{COST_RUN_ID}.parquet"

# --- Path sanity check ------------------------------------------------------
for _name, _p in [("POLICY", POLICY_FILE), ("POLICY_LINK", POLICY_LINK_FILE),
                  ("GOLDEN", GOLDEN_FILE), ("RAPID2", RAPID2_FILE),
                  ("RWT", RWT_FILE), ("PROMPT", POLICY_PROMPT_PATH)]:
    print(f"{_name:12s} {'OK     ' if _p.exists() else 'MISSING'} {_p}")
print(f"{'OUTPUT':12s} {OUTPUT_DIR.resolve()}")
print(f"{'COST':12s} {COST_PARQUET}")


# %% ==========================================================================
# [0b] COST LEDGER
#
# ONE row per model call ATTEMPT -- a retried call costs money on every attempt,
# and a response that fails to parse was still billed. Nothing that calls the
# gateway should bypass chat_with_cost / embed_with_cost.
# =============================================================================

# USD per 1M tokens. FILL FROM YOUR GATEWAY RATE CARD. Unpriced models are still
# logged (tokens); cost is recomputed from THIS table in [G], so a price fixed
# later re-prices every past call without re-running anything.
#   in        : fresh (uncached) prompt tokens
#   cached_in : cached prompt tokens (None -> billed at `in`)
#   out       : completion tokens (reasoning tokens are INSIDE completion_tokens)
PRICING = {
    MODEL:       {"in": None, "cached_in": None, "out": None},
    EMBED_MODEL: {"in": None, "cached_in": None, "out": 0.0},
}

# If the upstream embed_with_cache ALREADY writes its own 'emb' rows to the team
# cost file, set True -- otherwise embeddings are counted twice.
EMBED_COST_LOGGED_UPSTREAM = False
COST_KIND_MAP = {"embedding": "emb", "chat": "chat"}

COST_LEDGER       = OUTPUT_DIR / "cost_ledger.jsonl"            # append-only, crash-safe
COST_FILE         = OUTPUT_DIR / f"cost_log_{MODEL}.csv"        # ledger, re-priced
COST_SUMMARY_FILE = OUTPUT_DIR / f"cost_summary_{MODEL}.csv"    # by run / step / model

_COST_SESSION = {"calls": 0, "usd": 0.0, "unpriced_calls": 0}   # this kernel only

for _m, _p in PRICING.items():
    if _p.get("in") is None:
        print(f"[0b] WARNING no price for '{_m}' -- tokens will be logged, cost left blank.")


def _append_jsonl(path, obj):
    """Append one JSON line and force it to disk. A hard kill can leave the file
    without a trailing newline, which would glue the next append onto the torn
    line and lose a SECOND record, so terminate it first."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_nl = fh.read(1) != b"\n"
        if needs_nl:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def _read_jsonl(path):
    """Tolerant reader: a torn final line (hard kill mid-write) is dropped."""
    out = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"    [jsonl] dropping one torn line in {Path(path).name}")
    return out


# ---- token counting (projection + embedding estimates only) -----------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
    TOKEN_ESTIMATOR = "tiktoken:o200k_base"
except Exception:                    # not installed / encoding download blocked
    _ENC = None
    TOKEN_ESTIMATOR = "chars/4"


def _count_tokens(text) -> int:
    if not isinstance(text, str) or not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _usd(model, prompt_tokens=0, cached_tokens=0, completion_tokens=0):
    """Cost of one call under the CURRENT PRICING table; None if unpriced."""
    p = PRICING.get(model)
    if not p or p.get("in") is None:
        return None
    cached = int(cached_tokens or 0)
    fresh  = max(int(prompt_tokens or 0) - cached, 0)
    c_in   = p["in"] if p.get("cached_in") is None else p["cached_in"]
    out    = p.get("out") or 0.0
    return (fresh * p["in"] + cached * c_in + int(completion_tokens or 0) * out) / 1e6


def log_cost(step, model, kind="chat", prompt_tokens=0, completion_tokens=0,
             reasoning_tokens=0, cached_tokens=0, status="ok", estimated=False,
             record_id=None, unit=None, attempt=None, n_items=None,
             finish_reason=None, latency_s=None, error=None, note=None):
    usd = _usd(model, prompt_tokens, cached_tokens, completion_tokens)
    row = {"run_id": RUN_ID, "cost_run_id": COST_RUN_ID, "ts_epoch": time.time(),
           "ts": datetime.now(timezone.utc).isoformat(),
           "step": step, "kind": kind, "model": model, "status": status,
           "attempt": attempt, "record_id": record_id, "unit": unit,
           "n_items": n_items,
           "prompt_tokens": int(prompt_tokens or 0),
           "cached_tokens": int(cached_tokens or 0),
           "completion_tokens": int(completion_tokens or 0),
           "reasoning_tokens": int(reasoning_tokens or 0),
           "finish_reason": finish_reason,
           "latency_s": None if latency_s is None else round(latency_s, 3),
           "estimated": bool(estimated), "cost_usd": usd,
           "error": None if error is None else str(error)[:500], "note": note}
    _append_jsonl(COST_LEDGER, row)
    _COST_SESSION["calls"] += 1
    if usd is None:
        _COST_SESSION["unpriced_calls"] += 1
    else:
        _COST_SESSION["usd"] += usd
    return row


def _usage_fields(u):
    """Normalise an OpenAI-style usage object; missing detail blocks -> 0."""
    if u is None:
        return dict(prompt_tokens=0, completion_tokens=0,
                    reasoning_tokens=0, cached_tokens=0)
    ctd = getattr(u, "completion_tokens_details", None)
    ptd = getattr(u, "prompt_tokens_details", None)
    return dict(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        reasoning_tokens=(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0,
        cached_tokens=(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0)


async def chat_with_cost(step, messages, call_kwargs, parse=None, *,
                         record_id=None, unit=None, attempt=1, n_items=None):
    """The ONLY way this notebook talks to the chat endpoint (judge AND audit).
    Usage is logged the moment the response lands -- BEFORE parsing -- so a
    billed but unparseable response is still costed. Raises on call or parse
    failure so the caller's retry loop behaves exactly as before.
    Returns (parsed_or_raw_response, usage_dict)."""
    model = call_kwargs.get("model", MODEL)
    t0 = time.time()
    try:
        token = await tok.get()
        r = await oai.chat.completions.create(
            messages=messages,
            extra_headers={"Authorization": f"Bearer {token}"},   # gateway v2 auth
            **call_kwargs)
    except Exception as e:
        log_cost(step, model, status="call_error", record_id=record_id, unit=unit,
                 attempt=attempt, n_items=n_items, latency_s=time.time() - t0,
                 error=f"{type(e).__name__}: {e}")
        raise

    u  = _usage_fields(getattr(r, "usage", None))
    fr = r.choices[0].finish_reason if getattr(r, "choices", None) else None
    parsed, status, err = r, "ok", None
    if parse is not None:
        try:
            parsed = parse(r)
        except Exception as e:
            status, err = "parse_error", f"{type(e).__name__}: {e}"
    log_cost(step, model, status=status, record_id=record_id, unit=unit,
             attempt=attempt, n_items=n_items, finish_reason=fr,
             latency_s=time.time() - t0, error=err, **u)
    if err:
        raise ValueError(f"[{step}] unparseable response (finish_reason={fr}): {err}")
    return parsed, u


def embed_with_cost(step, ids, texts, cache_tag):
    """Wrap the upstream embedding cache. The cache does not expose usage, so
    tokens are ESTIMATED locally and flagged; treat as an UPPER BOUND because
    cache hits are not billed."""
    t0 = time.time()
    try:
        vecs = embed_cache(ids, texts, cache_tag)
    except Exception as e:
        log_cost(step, EMBED_MODEL, kind="embedding", status="call_error",
                 n_items=len(texts), latency_s=time.time() - t0,
                 error=f"{type(e).__name__}: {e}", note=f"cache_tag={cache_tag}")
        raise
    if EMBED_COST_LOGGED_UPSTREAM:
        return vecs                   # upstream already costed it -- don't double count
    n_tok = int(sum(_count_tokens(t) for t in texts))
    log_cost(step, EMBED_MODEL, kind="embedding", prompt_tokens=n_tok,
             estimated=True, n_items=len(texts), latency_s=time.time() - t0,
             note=f"cache_tag={cache_tag}; {TOKEN_ESTIMATOR}; upper bound (cache hits free)")
    return vecs


def write_team_cost_file(verbose=False):
    """Rebuild cost_log_<COST_RUN_ID>.parquet from the ledger, in the team format:
    kind, in, out, ts, use_case, staff_id, run_id, run_cost_usd.
    Every chat AND embedding call of this run lands here (every attempt).
    Rebuilt from the fsync'd ledger each time, so a disconnect loses nothing and
    repeated calls never duplicate rows. Written atomically (tmp + rename)."""
    rows = [r for r in _read_jsonl(COST_LEDGER) if r.get("cost_run_id") == COST_RUN_ID]
    df = pd.DataFrame(rows)
    if not len(df):
        return None
    df = df[(df.prompt_tokens > 0) | (df.completion_tokens > 0)]   # drop zero-token errors
    usd = [_usd(m, p, c, o) for m, p, c, o in zip(df.model, df.prompt_tokens,
                                                   df.cached_tokens, df.completion_tokens)]
    run_total = None if any(u is None for u in usd) else float(sum(usd))
    out = pd.DataFrame({
        "kind":         df["kind"].map(COST_KIND_MAP).fillna(df["kind"]).values,
        "in":           df["prompt_tokens"].astype("int64").values,
        "out":          df["completion_tokens"].astype("int64").values,
        "ts":           df["ts_epoch"].astype("float64").values,
        "use_case":     USE_CASE,
        "staff_id":     STAFF_ID,
        "run_id":       COST_RUN_ID,
        "run_cost_usd": run_total,
    })
    tmp = COST_PARQUET.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, COST_PARQUET)
    if verbose:
        print(f"    [cost] {len(out):,} rows {out.kind.value_counts().to_dict()} "
              f"run_cost_usd={'unpriced' if run_total is None else f'{run_total:.6f}'}"
              f" -> {COST_PARQUET.name}")
    return out


def _spend_line():
    s = _COST_SESSION
    tail = f" (+{s['unpriced_calls']} unpriced)" if s["unpriced_calls"] else ""
    return f"session spend ${s['usd']:.4f} over {s['calls']:,} calls{tail}"


# %% ==========================================================================
# [A] DATA LOAD
# =============================================================================
POLICY_RENAME = {
    "policy_core_id":                    "POLICY_ID",
    "transfer_date":                     "TRANSFER_DATE",
    "country":                           "COUNTRY",
    "policyTitle":                       "POL_TITLE",
    "policyRiskTypePrimary":             "POL_RISK_TYPE",
    "policyText.sgmntText":              "POL_TEXT",
    "policyPurpose.sgmntText":           "POL_PURPOSE",
    "policyApplication.riskTxnmyL2Text": "POL_TAX_L2",
    "versn.versStatCde":                 "POL_STATUS",
}
PUBLISHED_CODE        = "PUBL"
POLICY_TEXT_MAX_CHARS = 12_000      # body budget per policy; the probe reports truncation
POLICY_TEXT_COLS      = ["POL_TITLE", "POL_RISK_TYPE", "POL_TAX_L2",
                         "COUNTRY", "POL_PURPOSE", "POL_TEXT"]

# IA_RSM_POLICY_Linkage headers -> short codes. This is the Extract3 analogue.
POLICY_LINK_RENAME = {
    "RegMap Regulation ID":  "REG_ID",
    "Jurisdiction_Regmap":   "JUR",
    "RSM ID":                "SUM_ID",
    "Policy ID":             "POLICY_ID",
    "Last Published Version": "POL_LINK_VER",
    "Status":                "POL_LINK_STATUS",
    "Type":                  "POL_LINK_TYPE",
    "L1 RT":                 "TAX_L1",
    "L2 RT":                 "TAX_L2",
    "L3 RT":                 "TAX_L3",
}
# The linkage file is multi-jurisdiction; the policy pool and golden are US.
# Set to None to keep every jurisdiction.
JUR_FILTER = {"US", "UNITED STATES", "UNITED STATES OF AMERICA", "USA"}

_BLANK = {"", "NONE", "NAN", "-", "NAT", "<NA>"}


def _read_parquet(path, columns=None) -> pd.DataFrame:
    """Parquet written from a warehouse client can carry pandas metadata for
    extension dtypes ('dbdate' / 'dbtime' from db-dtypes). Without that package
    pandas raises "data type 'dbdate' not understood". Try the normal read, then
    db_dtypes, then fall back to ignoring the pandas metadata (dates come back
    as datetime.date; pd.to_datetime handles them)."""
    try:
        return pd.read_parquet(path, columns=columns)
    except TypeError as e:
        if "not understood" not in str(e):
            raise
    try:
        import db_dtypes  # noqa: F401  -- registers dbdate / dbtime
        return pd.read_parquet(path, columns=columns)
    except (ImportError, TypeError):
        pass
    import pyarrow.parquet as pq
    df = pq.read_table(path, columns=columns).to_pandas(ignore_metadata=True)
    return df.drop(columns=[c for c in df.columns if str(c).startswith("__index_level_")])


def _read_any(path: Path, usecols=None) -> pd.DataFrame:
    """Read a parquet / xlsx / csv into a DataFrame, tolerant of the format."""
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        df = _read_parquet(path, columns=usecols)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if usecols:
            df = df[[c for c in usecols if c in df.columns]]
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find(directory: Path, stem: str) -> Path:
    """Find the first file in `directory` whose name contains `stem`."""
    hits = [p for p in Path(directory).iterdir()
            if p.suffix.lower() in (".parquet", ".xlsx", ".xls", ".csv")
            and stem.lower() in p.name.lower()]
    if not hits:
        raise FileNotFoundError(f"no file matching '{stem}' in {directory}")
    return sorted(hits)[0]


def _norm_id(series) -> pd.Series:
    """String / stripped / upper-cased, and a float-read id ('394.0') collapses
    to '394' so integer policy ids join to ids read with NaNs present."""
    return (pd.Series(series).astype(str).str.strip().str.upper()
              .str.replace(r"\.0+$", "", regex=True))


def _nid(x) -> str:
    """Scalar form of _norm_id, for ids echoed back by the model."""
    return re.sub(r"\.0+$", "", str(x).strip().upper())


def _as_text(v) -> str:
    """Parquet can hand back arrays for repeated fields -> join them."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return "; ".join(t for t in (_as_text(x) for x in v) if t)
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def _clean_field(v) -> str:
    """The policy extract carries HTML entities (&ndash; &ldquo; &gt; &amp;),
    some double-encoded -> unescape twice, drop tags, collapse whitespace."""
    s = html.unescape(html.unescape(_as_text(v)))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _truncate(s: str, n: int = None) -> str:
    n = n or POLICY_TEXT_MAX_CHARS
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + " [... policy text truncated]"


def load_policies():
    """One row per PUBLISHED policy with a judgeable `policy_text`.

    The extract holds several PUBL rows per policy_core_id (repeated snapshots
    with different transfer dates) -> newest transfer wins. If a version number
    column becomes available, sort on it first.

    Returns (policies_df, ids_any_status). The second set lets [B2] split the
    data gap into NOT_PUBLISHED (exists, wrong status) vs ABSENT (not in extract).
    """
    raw = _read_any(POLICY_FILE).rename(columns=POLICY_RENAME)
    missing = [c for c in POLICY_RENAME.values() if c not in raw.columns]
    if missing:
        raise KeyError(f"policy extract missing {missing}; columns are {list(raw.columns)}")

    raw["POLICY_ID"] = _norm_id(raw["POLICY_ID"])
    raw = raw[~raw.POLICY_ID.isin(_BLANK)].copy()
    raw["POL_STATUS"]    = raw["POL_STATUS"].astype(str).str.strip().str.upper()
    raw["TRANSFER_DATE"] = pd.to_datetime(raw["TRANSFER_DATE"], errors="coerce")
    ids_any_status = set(raw.POLICY_ID)

    pub = raw[raw.POL_STATUS.eq(PUBLISHED_CODE)]
    dup_rows = int(pub.POLICY_ID.duplicated().sum())
    pol = (pub.sort_values("TRANSFER_DATE", ascending=False, na_position="last")
              .drop_duplicates("POLICY_ID").copy())

    for c in POLICY_TEXT_COLS:
        pol[c] = pol[c].map(_clean_field)

    pol["body_chars"]     = pol["POL_TEXT"].str.len()
    pol["text_truncated"] = pol["body_chars"] > POLICY_TEXT_MAX_CHARS

    # Title / risk type / taxonomy / country / purpose are part of the policy's
    # MEANING -- the body alone is long and generic in places.
    pol["policy_text"] = (
        "Policy Title:- "          + pol.POL_TITLE
      + "\n\nPrimary Risk Type:- " + pol.POL_RISK_TYPE
      + "\n\nRisk Taxonomy L2:- "  + pol.POL_TAX_L2
      + "\n\nCountry:- "           + pol.COUNTRY
      + "\n\nPurpose:- "           + pol.POL_PURPOSE
      + "\n\nPolicy Text:- "       + pol.POL_TEXT.map(_truncate)
    ).map(clean)

    print(f"[A] policy extract: rows={len(raw):,}  {PUBLISHED_CODE} rows={len(pub):,}  "
          f"duplicate {PUBLISHED_CODE} rows collapsed={dup_rows:,}  "
          f"-> policies={len(pol):,}  (ids any status={len(ids_any_status):,})")

    keep = ["POLICY_ID", "POL_TITLE", "POL_RISK_TYPE", "POL_TAX_L2", "COUNTRY",
            "TRANSFER_DATE", "body_chars", "text_truncated", "policy_text"]
    return pol[keep].reset_index(drop=True), ids_any_status


def load_policy_links() -> pd.DataFrame:
    """IA_RSM_POLICY_Linkage: the RSM -> Policy edge. THIS is our parentage
    source -- the direct analogue of Extract3's RSM -> Library Risk.

    Rows whose taxonomy is 'Placeholder Risk ...' carry no Policy ID; those are
    MISSING DATA in RegMap, not evidence of no policy, and are dropped here and
    counted in the probe.
    Output columns: REG_ID, SUM_ID, POLICY_ID (+ JUR, TAX_L2 for reporting)."""
    df = _read_any(POLICY_LINK_FILE).rename(columns=POLICY_LINK_RENAME)
    for c in ["REG_ID", "SUM_ID", "POLICY_ID"]:
        if c not in df.columns:
            raise KeyError(f"policy linkage missing {c}; align POLICY_LINK_RENAME. "
                           f"Columns: {list(df.columns)}")
        df[c] = _norm_id(df[c])
    df["JUR"] = df["JUR"].map(_as_text).str.strip().str.upper() if "JUR" in df.columns else ""

    n_all = len(df)
    placeholder = df["TAX_L2"].map(_as_text).str.contains("placeholder", case=False, na=False) \
                  if "TAX_L2" in df.columns else pd.Series(False, index=df.index)
    df = df[~df.POLICY_ID.isin(_BLANK) & ~df.SUM_ID.isin(_BLANK)]
    n_with_policy = len(df)

    if JUR_FILTER:
        before = len(df)
        df = df[df.JUR.isin(JUR_FILTER)] if "JUR" in df.columns else df
        print(f"[A] policy links: jurisdiction filter kept {len(df):,} of {before:,} rows")

    print(f"[A] policy links: rows={n_all:,}  with a Policy ID={n_with_policy:,}  "
          f"placeholder-taxonomy rows={int(placeholder.sum()):,} (missing data, dropped)")

    cols = [c for c in ["REG_ID", "SUM_ID", "POLICY_ID", "JUR", "TAX_L2"] if c in df.columns]
    return df[cols].drop_duplicates(subset=["SUM_ID", "POLICY_ID"]).reset_index(drop=True)


def load_golden() -> pd.DataFrame:
    """Golden control mapping: known-true Record -> RGL links (its RSM column is
    blank, exactly as in the LR lane). VALIDATION ONLY -- never used to build
    candidates. The record -> POLICY truth is DERIVED from this in
    derive_golden_policy()."""
    g = _read_any(GOLDEN_FILE).rename(columns={
        "Record ID":     "RECORD_ID",
        "Regulation ID": "REG_ID",
    })
    for c in ["RECORD_ID", "REG_ID"]:
        if c not in g.columns:
            raise KeyError(f"golden missing {c}. Columns: {list(g.columns)}")
        g[c] = _norm_id(g[c])
    return g[~g.RECORD_ID.isin(_BLANK) & ~g.REG_ID.isin(_BLANK)].reset_index(drop=True)


def load_rwt() -> pd.DataFrame:
    """Upstream Alert -> RSM/RGL run. Prefer the in-kernel `results_with_truth`
    if it is already loaded; otherwise read the parquet."""
    if "results_with_truth" in globals():
        return globals()["results_with_truth"]
    return _read_parquet(RWT_FILE)


def load_alerts(record_ids) -> pd.DataFrame:
    """Build one cleaned alert text per record, restricted to the records we run.
    (Verbatim from the RSM/RGL and LR lanes so alert text is enriched IDENTICALLY.)"""
    df = _read_parquet(RAPID2_FILE)

    for col in ["TITLE", "SUMM_CONTENT", "SUMM_UPDATE",
                "RECORD_JURISDICTION_CODE", "RECORD_JURISDICTION_NAME", "REGULATOR_CODE"]:
        if col in df.columns:                      # REGULATOR_CODE may not exist
            df[col] = df[col].fillna("").map(clean)
        else:
            df[col] = ""

    # When the running summary equals the original content, don't repeat it.
    content_equals_update = df["SUMM_CONTENT"] == df["SUMM_UPDATE"]
    df["alert_text"] = np.where(
        content_equals_update,
        (df.TITLE + "\n\n" + df.SUMM_CONTENT).str.strip(),
        ("Title:- "                   + df.TITLE
         + "\n\nContent:- "           + df.SUMM_CONTENT
         + "\n\nUpdate:- "            + df.SUMM_UPDATE
         + "\n\nJurisdiction Code:- " + df.RECORD_JURISDICTION_CODE
         + "\n\nJurisdiction Name:- " + df.RECORD_JURISDICTION_NAME
         + "\n\nRegulator Code:- "    + df.REGULATOR_CODE).str.strip())

    df["RECORD_ID"] = _norm_id(df["RECORD_ID"])
    df = df[df.RECORD_ID.isin(record_ids) & (df.alert_text.str.len() > 0)]
    return df[["RECORD_ID", "alert_text"]].drop_duplicates("RECORD_ID").reset_index(drop=True)


# ---- Execute the loads ------------------------------------------------------
policies, policy_ids_any_status = load_policies()   # PUBL policies + text
policy_links = load_policy_links()                  # RSM -> Policy  <- PARENTAGE
golden_reg   = load_golden()                        # record -> RGL (validation only)
rwt          = load_rwt()                           # upstream alert -> RSM/RGL

print("[A] loaded:",
      f"policies={len(policies):,}  policy_links={len(policy_links):,}  "
      f"golden_reg={len(golden_reg):,}  rwt={len(rwt):,}")


# %% ==========================================================================
# [A2] VALIDATION PROBE -- run this BEFORE building anything.
# Five numbers decide top_k, chunk size, and whether the chain is even wired up.
# =============================================================================
print("=" * 78); print("[A2] VALIDATION PROBE -- POLICY LANE"); print("=" * 78)

# --- 1. policy pool
_ln = policies.policy_text.str.len()
print(f"  published policies with text : {len(policies):,}")
print(f"  policy_text chars med/min/max: {int(_ln.median()):,} / {int(_ln.min()):,} / {int(_ln.max()):,}")
print(f"  bodies truncated at {POLICY_TEXT_MAX_CHARS:,}    : {int(policies.text_truncated.sum()):,}")
print(f"  policies with EMPTY body     : {int((policies.body_chars == 0).sum()):,}")
print(f"  countries                    : {policies.COUNTRY.value_counts().to_dict()}")

# --- 2. RSM -> Policy fanout   *** THE CRITICAL ONE ***
_fan = policy_links.groupby("SUM_ID").POLICY_ID.nunique()
print(f"\n  RSM -> Policy fanout         : median={_fan.median():.0f}  "
      f"mean={_fan.mean():.1f}  max={_fan.max()}")
print(f"  RSMs with a policy           : {policy_links.SUM_ID.nunique():,}")
print(f"  distinct policies in linkage : {policy_links.POLICY_ID.nunique():,}")
print(f"  ... present in the PUBL pool : "
      f"{len(set(policy_links.POLICY_ID) & set(policies.POLICY_ID)):,}")

# --- 3. rwt -> RSM chain (this is what returned 0 candidates in the LR lane)
print(f"\n  rwt cols with 'rsm'          : {[c for c in rwt.columns if 'rsm' in c.lower()]}")
_linked = rwt[rwt["final_decision"].astype(str).str.upper() == "LINKED"].copy()
_linked["RECORD_ID"] = _norm_id(_linked["RECORD_ID"])
print(f"  rwt LINKED rows              : {len(_linked):,}")
_rbr = {}
for _, _row in _linked.iterrows():
    _ids = str(_row.get("linked_rsm_ids", "") or "")
    _rs = {_nid(x) for x in re.split(r"[|,]", _ids) if x.strip()}
    _rbr.setdefault(_row["RECORD_ID"], set()).update(_rs)
_all_rsms = set().union(*_rbr.values()) if _rbr else set()
print(f"  records with RSMs            : {len(_rbr):,}   distinct RSMs: {len(_all_rsms):,}")
print(f"  RSM overlap with linkage file: {len(_all_rsms & set(policy_links.SUM_ID)):,}")
_pols = set(policy_links[policy_links.SUM_ID.isin(_all_rsms)].POLICY_ID)
print(f"  policies reachable via those : {len(_pols):,}   "
      f"with text: {len(_pols & set(policies.POLICY_ID)):,}")

# --- 4. golden record coverage
print(f"\n  golden records               : {golden_reg.RECORD_ID.nunique():,}")
print(f"  golden RGLs                  : {golden_reg.REG_ID.nunique():,}")
print(f"  golden RGL overlap w/ linkage: "
      f"{len(set(golden_reg.REG_ID) & set(policy_links.REG_ID)):,}")

print("\n  READ: fanout median 1 -> mapping is deterministic, the judge only filters.")
print("        RSM overlap 0     -> linked_rsm_ids column name / delimiter wrong.")
print("        pool overlap low  -> the linkage file points at policies outside the")
print("                             US PUBL extract: a DATA GAP, not a model problem.")
print("=" * 78)


# %% ==========================================================================
# [B1] CONFIG + MODEL PARAMS + CHECKPOINT CONFIG
# Same model as the RSM/RGL and LR judges; only the prompt + schema change.
# chunk is smaller than the LR lane's 40 because policy text is far longer --
# check the worst-case per-call size printed in [B2].
# =============================================================================
with open(POLICY_PROMPT_PATH, "r") as fh:
    POLICY_SYSTEM_PROMPT = fh.read()
print(POLICY_SYSTEM_PROMPT[:600])
PROMPT_SHA = hashlib.sha256(POLICY_SYSTEM_PROMPT.encode()).hexdigest()[:12]

LANE = {
    "top_k":              None,   # None -> judge ALL policies under the linked RSMs.
                                  # The cascade already bounds the set, so retrieval is
                                  # removed as a confound: NOT_JUDGED can then only mean
                                  # a data gap or an upstream RSM-recall gap.
    "max_tokens":         50000,  # per judge call
    "concurrency":        30,     # parallel judge calls in flight   <-- speed lever
    "chunk":              10,     # policies per judge call (long text)
    "max_retries":        3,
    "reasoning":          "low",
    "order_by_embedding": True,   # ordering only; False skips the embedding spend
}
JUDGE_LOGPROBS = False            # newer gateway models reject logprobs -> must stay False

# --- cost projection assumptions (used in [B2] only) -------------------------
EST_OUT_TOKENS_PER_POLICY = 80    # verdict + one-line reasoning + low-effort reasoning
PER_POLICY_JSON_OVERHEAD  = 15
PER_CALL_MSG_OVERHEAD     = 30

# --- checkpointing -----------------------------------------------------------
SNAPSHOT_EVERY  = 25              # parquet snapshot cadence, in completed units
MAX_PAIRS_GUARD = 400_000         # refuse to start a run bigger than this by accident

PLAN_FILE      = CKPT_DIR / "plan.json"
VERDICTS_JSONL = CKPT_DIR / "verdicts.jsonl"
STATE_FILE     = CKPT_DIR / "run_state.json"
PARTIAL_FILE   = OUTPUT_DIR / f"policy_verdicts_partial_{MODEL}.parquet"
FINAL_FILE     = OUTPUT_DIR / f"policy_verdicts_{MODEL}.parquet"

DECISION_RANK   = {"LINKED": 4, "INSUFFICIENT_EVIDENCE": 3, "NOT_LINKED": 2,
                   "MISSING": 1, "ERROR": 0}
RETRIEVAL_STATS = {}


# %% ==========================================================================
# [P] DATA PROVENANCE FINGERPRINT -- called before [C] and again at the end.
# Diff the two printouts by eye: mtime, prompt hash, plan.json built_at.
# =============================================================================
def _fp(path):
    p = Path(path)
    if not p.exists():
        return f"MISSING: {p}"
    st = p.stat()
    return (f"{p.name:55s} modified "
            f"{datetime.fromtimestamp(st.st_mtime, tz=timezone.utc):%Y-%m-%d %H:%M UTC}"
            f"   size {st.st_size:,} B")


def _hash_ids(series):
    """Stable hash of a distinct-value set -- order-independent, catches content
    changes even when the row COUNT stays the same."""
    vals = sorted(set(str(v) for v in series if v is not None))
    return hashlib.sha256("|".join(vals).encode()).hexdigest()[:12]


def run_provenance_check(label):
    print("=" * 78); print(f"[P] DATA PROVENANCE -- {label}"); print("=" * 78)
    for lbl, path in [("POLICY_FILE", POLICY_FILE), ("POLICY_LINK_FILE", POLICY_LINK_FILE),
                      ("GOLDEN_FILE", GOLDEN_FILE), ("RAPID2_FILE (alerts)", RAPID2_FILE),
                      ("POLICY_PROMPT_PATH", POLICY_PROMPT_PATH)]:
        print(f"  {lbl:24s} {_fp(path)}")
    print("-" * 78)
    print(f"  policies      : {len(policies):,} rows | hash {_hash_ids(policies.POLICY_ID)}")
    print(f"  policy_links  : {len(policy_links):,} rows | "
          f"hash {_hash_ids(policy_links.SUM_ID + '|' + policy_links.POLICY_ID)}")
    print(f"  golden_reg    : {len(golden_reg):,} rows | "
          f"{golden_reg.RECORD_ID.nunique():,} records | hash {_hash_ids(golden_reg.REG_ID)}")
    print(f"  MODEL         : {MODEL}   EMBED: {EMBED_MODEL}")
    print(f"  prompt hash   : {PROMPT_SHA}   ({len(POLICY_SYSTEM_PROMPT):,} chars)")
    if PLAN_FILE.exists():
        _plan = json.loads(PLAN_FILE.read_text())
        print(f"  plan built_at : {_plan.get('built_at')}   prompt_sha {_plan.get('prompt_sha')}")
        print("                  (if plan prompt_sha differs from the hash above, finished")
        print("                   units used a DIFFERENT prompt than is loaded now)")
    else:
        print("  plan.json     : not yet built")
    print(f"  {'verdicts parquet':14s}: {_fp(FINAL_FILE)}")
    print(f"  {'cost file':14s}: {_fp(COST_PARQUET)}")
    print("=" * 78)


run_provenance_check("BEFORE [C] -- baseline, nothing spent yet")


# %% ==========================================================================
# [B2] PARENTAGE + FUNNEL + REACHABILITY + COST PROJECTION
# No model call is made in this cell.
# =============================================================================
def build_parentage():
    """Parentage maps used everywhere downstream.
      rsm_to_policy : RSM -> [POLICY_ID, ...]      (candidate build)
      policy_to_rsm : POLICY_ID -> set(RSM)        (reachability)
      reg_to_policy : RGL -> set(POLICY_ID)        (golden derivation ONLY)
    """
    rsm_to_policy = (policy_links.groupby("SUM_ID")["POLICY_ID"]
                       .apply(lambda s: list(dict.fromkeys(s))).to_dict())
    policy_to_rsm = policy_links.groupby("POLICY_ID")["SUM_ID"].apply(set).to_dict()
    reg_to_policy = policy_links.groupby("REG_ID")["POLICY_ID"].apply(set).to_dict()
    return {"rsm_to_policy": rsm_to_policy, "policy_to_rsm": policy_to_rsm,
            "reg_to_policy": reg_to_policy}


def derive_golden_policy(parentage):
    """Record -> POLICY truth, derived as  golden(Record -> RGL)  x  (RGL -> Policy).

    THE CAVEAT, same as the LR lane: the golden carries no RSM, so every policy
    under a LINKED REGULATION becomes 'golden-true' for that record, including
    policies the specific alert does not touch. That inflates the denominator
    and is the main reason naive recall looks low. Quantified in [E]/[F] as
    STRUCTURAL_OVERMAPPING -- never present the naive number without it."""
    reg_to_policy = parentage["reg_to_policy"]
    gp = golden_reg[["RECORD_ID", "REG_ID"]].drop_duplicates()
    rows = []
    for rid, reg in zip(gp.RECORD_ID, gp.REG_ID):
        for pol in reg_to_policy.get(reg, ()):
            rows.append((rid, reg, pol))
    g = pd.DataFrame(rows, columns=["RECORD_ID", "REG_ID", "POLICY_ID"]).drop_duplicates()
    print(f"[B2] derived golden: {len(gp):,} record-RGL pairs -> {len(g):,} record-policy "
          f"pairs over {g.RECORD_ID.nunique():,} records, {g.POLICY_ID.nunique():,} policies")
    return g


def print_funnel_report(parentage, golden):
    """Counts + overlaps that give a complete picture of the data. This is the
    cell you show to prove the pipeline sees what it should."""
    bar = "=" * 78
    pool         = set(policies.POLICY_ID)
    link_pols    = set(policy_links.POLICY_ID)
    golden_recs  = set(golden.RECORD_ID)
    golden_pols  = set(golden.POLICY_ID)

    print(bar); print("[B2] DATA FUNNEL - LEVEL SIZES PER SOURCE"); print(bar)
    print(f"  Linkage : RGL={policy_links.REG_ID.nunique():,}  "
          f"RSM={policy_links.SUM_ID.nunique():,}  Policy={len(link_pols):,}")
    print(f"  Policy  : {PUBLISHED_CODE} policies with text={len(pool):,}  "
          f"(ids any status={len(policy_ids_any_status):,})")
    print(f"  Golden  : records={len(golden_recs):,}  policies={len(golden_pols):,}  "
          f"pairs={len(golden):,}")
    print(f"  RWT     : records={rwt['RECORD_ID'].nunique():,}  linked rows="
          f"{int((rwt['final_decision'].astype(str).str.upper()=='LINKED').sum()):,}")

    print("\n" + bar); print("[B2] OVERLAPS - do the sources agree on ids?"); print(bar)
    def _ov(name, a, b):
        print(f"  {name:44s}: {len(a & b):6,} / {len(a):6,}  "
              f"({len(a & b)/max(len(a),1):5.1%} of first)")
    _ov("linkage policies present in PUBL pool", link_pols, pool)
    _ov("golden policies present in PUBL pool",  golden_pols, pool)
    _ov("golden records present in RWT",         golden_recs,
        set(_norm_id(rwt["RECORD_ID"])))

    # --- Reachability: can we even build candidates for the golden policies? ---
    policy_to_rsm = parentage["policy_to_rsm"]
    reachable = {p for p in golden_pols if p in policy_to_rsm and p in pool}
    not_pub   = (golden_pols - pool) & policy_ids_any_status
    absent    = golden_pols - policy_ids_any_status - pool
    print("\n" + bar); print("[B2] REACHABILITY - can the pipeline judge each golden policy?"); print(bar)
    print(f"  golden policies total                  : {len(golden_pols):,}")
    print(f"  ... with an RSM parent in the linkage   : {len(golden_pols & set(policy_to_rsm)):,}")
    print(f"  ... published with text                 : {len(golden_pols & pool):,}")
    print(f"  ... REACHABLE (both -> can be judged)   : {len(reachable):,} "
          f"({len(reachable)/max(len(golden_pols),1):.1%})")
    print(f"  DATA GAP: in extract but not {PUBLISHED_CODE}       : {len(not_pub):,}")
    print(f"  DATA GAP: absent from the extract       : {len(absent):,}")
    print(f"  ==> upper bound on policy recall        : "
          f"{len(reachable)/max(len(golden_pols),1):.1%}  (no prompt can beat this)")

    # --- LINKAGES + COVERAGE (pairs, not just ids) ---
    print("\n" + bar); print("[B2] LINKAGES + COVERAGE (golden record->policy pairs)"); print(bar)
    gp = golden[["RECORD_ID", "REG_ID", "POLICY_ID"]].drop_duplicates(
        subset=["RECORD_ID", "POLICY_ID"])

    def _coverage(frame, label):
        rsm_set = set()
        for p in frame["POLICY_ID"].unique():
            rsm_set |= policy_to_rsm.get(p, set())
        print(f"  {label:34s}: {len(frame):6,} linkages | "
              f"alerts={frame['RECORD_ID'].nunique():4,}  RGL={frame['REG_ID'].nunique():4,}  "
              f"policies={frame['POLICY_ID'].nunique():4,}  RSM(parents)={len(rsm_set):4,}")

    _coverage(gp, "ALL golden linkages")
    _coverage(gp[gp.POLICY_ID.isin(reachable)], "... REACHABLE (judgeable) subset")
    lost = len(gp) - len(gp[gp.POLICY_ID.isin(reachable)])
    print(f"  {'... LOST (no published policy text)':34s}: {lost:6,} linkages "
          f"({lost/max(len(gp),1):.1%}) -> data gap, cannot be judged")

    print("\n" + bar); print("[B2] TWO PATHS TO A POLICY (why numbers can look bad)"); print(bar)
    print("  GOLDEN path   : Record -> RGL -> Policy        (RSM is BLANK in golden)")
    print("  PIPELINE path : Record -> LINKED RSM -> Policy  (from the RSM->Policy linkage)")
    print("  Because golden has no RSM, every policy under a linked REGULATION counts as")
    print("  'golden-true', even policies the specific alert does not touch. That inflates")
    print("  the denominator and is the main reason apparent recall looks low. (E/F.)")
    return reachable


def project_cost(records):
    """Rough spend BEFORE the run. Input tokens are counted exactly; output is an
    assumption -- compare against the real ledger in [G] after a 10-record smoke
    run and tune EST_OUT_TOKENS_PER_POLICY."""
    alerts_df = load_alerts(set(records))
    n_alert, chunk = len(alerts_df), LANE["chunk"]
    p_tok   = policies.set_index("POLICY_ID").policy_text.map(_count_tokens)
    a_tok   = alerts_df.alert_text.map(_count_tokens)
    sys_tok = _count_tokens(POLICY_SYSTEM_PROMPT)

    # candidates per record, from the cascade (not the whole pool)
    rsm_to_policy = parentage["rsm_to_policy"]
    per_rec = []
    for rid in alerts_df.RECORD_ID:
        cands = set()
        for rsm in rsms_by_record.get(rid, set()):
            cands |= {p for p in rsm_to_policy.get(rsm, []) if p in p_tok.index}
        per_rec.append(cands)
    n_cand   = int(sum(len(c) for c in per_rec))
    n_calls  = int(sum(math.ceil(len(c) / chunk) for c in per_rec))
    cand_tok = int(sum(p_tok[list(c)].sum() for c in per_rec if c))

    judge_in  = (n_calls * (sys_tok + PER_CALL_MSG_OVERHEAD) + cand_tok
                 + n_cand * PER_POLICY_JSON_OVERHEAD
                 + int(sum(t * math.ceil(len(c) / chunk)
                           for t, c in zip(a_tok, per_rec))))
    judge_out = n_cand * EST_OUT_TOKENS_PER_POLICY
    worst_chunk_in = (sys_tok + int(a_tok.max() if len(a_tok) else 0)
                      + int(p_tok.nlargest(chunk).sum()))
    embed_tok = int(p_tok.sum() + a_tok.sum()) if LANE["order_by_embedding"] else 0

    usd_judge = _usd(MODEL, judge_in, 0, judge_out)
    usd_embed = _usd(EMBED_MODEL, embed_tok, 0, 0)
    fmt = lambda x: "unpriced" if x is None else f"${x:,.2f}"

    bar = "=" * 78
    print(bar); print(f"[B2] COST PROJECTION  (tokens: {TOKEN_ESTIMATOR})"); print(bar)
    print(f"  alerts in scope                  : {n_alert:,}")
    print(f"  alert-policy pairs (cascade)     : {n_cand:,}   avg/alert "
          f"{n_cand/max(n_alert,1):.1f}")
    print(f"  judge calls (chunk={chunk})           : {n_calls:,}")
    print(f"  policy tokens  median / max      : "
          f"{int(p_tok.median()):,} / {int(p_tok.max()):,}")
    print(f"  WORST-CASE input per call        : {worst_chunk_in:,} tokens"
          "   <- lower chunk if this nears the context limit")
    print(f"  judge  in / out tokens           : {judge_in:,} / {judge_out:,}  -> {fmt(usd_judge)}")
    print(f"  embeddings (upper bound)         : {embed_tok:,}  -> {fmt(usd_embed)}")
    if usd_judge is not None:
        print(f"  judge cost per alert             : ${usd_judge/max(n_alert,1):.4f}")
    return {"judge_in": judge_in, "judge_out": judge_out, "calls": n_calls,
            "usd_judge": usd_judge, "usd_embed": usd_embed, "pairs": n_cand}


# ---- Execute -----------------------------------------------------------------
parentage = build_parentage()
golden    = derive_golden_policy(parentage)          # record -> policy truth (derived)

# record -> its LINKED RSMs, straight from the upstream run (the cascade)
_linked_rows = rwt[rwt["final_decision"].astype(str).str.upper() == "LINKED"].copy()
_linked_rows["RECORD_ID"] = _norm_id(_linked_rows["RECORD_ID"])
rsms_by_record = {}
for _, _row in _linked_rows.iterrows():
    _ids = str(_row.get("linked_rsm_ids", "") or "")
    rsms_by_record.setdefault(_row["RECORD_ID"], set()).update(
        {_nid(x) for x in re.split(r"[|,]", _ids) if x.strip()})

reachable_policies = print_funnel_report(parentage, golden)

# Scope from the PREVIOUS RUN, not from golden. Golden is validation only.
scope_records = set(rsms_by_record)

# >>> First pass: run on a handful of records before scaling up. <<<
# (delete checkpoints/plan.json when you widen the scope again)
# scope_records = set(sorted(scope_records)[:10])

projection = project_cost(scope_records)


# %% ==========================================================================
# [C] WORK PLAN + CHECKPOINTED MODEL EXECUTION (concurrent, resumable)
#
# Unit of work = (RECORD_ID, chunk_index) -> a deterministic key.
# plan.json       built ONCE and never rebuilt on resume.
# verdicts.jsonl  one fsync'd line per COMPLETED unit.
# cost_ledger     one fsync'd line per ATTEMPT (see [0b]).
# Resume          re-run cells [0] -> [C]; completed unit keys are skipped.
# =============================================================================
POLICY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "policy_linkage", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["policy_id", "decision", "relevance_score", "reasoning"],
                "properties": {
                    "policy_id":       {"type": "string"},
                    "decision":        {"type": "string",
                                        "enum": ["LINKED", "NOT_LINKED", "INSUFFICIENT_EVIDENCE"]},
                    "relevance_score": {"type": "integer"},
                    "reasoning":       {"type": ["string", "null"]},
                }}}}}
    }
}


def load_completed():
    """(done_unit_keys, rows_already_on_disk)"""
    done, rows = set(), []
    for rec in _read_jsonl(VERDICTS_JSONL):
        key = rec.get("unit")
        if key is None or key in done:          # ignore a duplicated unit
            continue
        done.add(key)
        rows.extend(rec.get("rows", []))
    return done, rows


def _snapshot(rows, n_done, n_total):
    df = pd.DataFrame(rows).drop_duplicates(["RECORD_ID", "POLICY_ID"])
    df.to_parquet(PARTIAL_FILE, index=False)
    STATE_FILE.write_text(json.dumps({
        "run_id": RUN_ID, "units_done": n_done, "units_total": n_total,
        "rows": len(df), "session_usd": round(_COST_SESSION["usd"], 6),
        "updated": datetime.now(timezone.utc).isoformat()}))
    write_team_cost_file()
    print(f"    [ckpt] {n_done}/{n_total} units, {len(df):,} rows -> "
          f"{PARTIAL_FILE.name} | {_spend_line()}")


def _finalise(rows):
    df = pd.DataFrame(rows)
    if len(df):
        # a retried unit can duplicate a pair; keep the strongest decision
        df["_r"] = df.decision.astype(str).str.upper().map(DECISION_RANK).fillna(-1)
        df = (df.sort_values("_r", ascending=False)
                .drop_duplicates(["RECORD_ID", "POLICY_ID"])
                .drop(columns="_r").reset_index(drop=True))
    df.to_parquet(FINAL_FILE, index=False)
    n_linked = int((df.decision.astype(str).str.upper() == "LINKED").sum()) if len(df) else 0
    print(f"[C] verdicts: {len(df):,}  linked: {n_linked:,}  -> {FINAL_FILE}")
    write_team_cost_file(verbose=True)
    print(f"[C] {_spend_line()}")
    return df


def build_plan(records, top_k=None, chunk=None, force=False):
    """Build (or load) the deterministic unit plan.
    Candidates = the policies under the record's LINKED RSMs. Embeddings (if
    enabled) order them; with top_k=None nothing is dropped."""
    chunk = chunk or LANE["chunk"]

    if PLAN_FILE.exists() and not force:
        plan = json.loads(PLAN_FILE.read_text())
        print(f"[C] plan loaded from checkpoint: {len(plan['units']):,} units, "
              f"{len(plan['records']):,} records, chunk={plan['chunk']}")
        if plan.get("prompt_sha") != PROMPT_SHA:
            print(f"[C] WARNING prompt changed since the plan was built "
                  f"({plan.get('prompt_sha')} -> {PROMPT_SHA}). Finished units used the "
                  f"old prompt; start a new EXECUTION_OUTPUT_DIR for a clean run.")
        return plan

    pol_text_by_id = {p: t for p, t in zip(policies.POLICY_ID, policies.policy_text)
                      if isinstance(t, str) and t.strip()}
    alerts_df = load_alerts(set(records))
    alert_text_by_id = dict(zip(alerts_df.RECORD_ID, alerts_df.alert_text))

    dropped = set(records) - set(alert_text_by_id)
    if dropped:
        print(f"[C] WARNING {len(dropped)} scoped records have no alert text and are "
              f"dropped: {sorted(dropped)[:5]}")

    rsm_to_policy = parentage["rsm_to_policy"]
    rec_list, cand_by_rec = [], {}
    for rid in sorted(records):
        if rid not in alert_text_by_id:
            continue
        child = []
        for rsm in rsms_by_record.get(rid, set()):
            child.extend(rsm_to_policy.get(rsm, []))
        child = [p for p in dict.fromkeys(child) if p in pol_text_by_id]  # dedup + has text
        if child:
            rec_list.append(rid); cand_by_rec[rid] = child

    if not cand_by_rec:
        raise ValueError(
            f"No candidates built. records with RSMs={len(rsms_by_record)}, "
            f"RSM overlap with linkage="
            f"{len(set().union(*rsms_by_record.values()) & set(rsm_to_policy)) if rsms_by_record else 0}, "
            f"policies in pool={len(pol_text_by_id)}, alerts={len(alert_text_by_id)}")

    # embeddings: ordering only
    used_embeddings = False
    if LANE["order_by_embedding"]:
        try:
            pool = sorted(pol_text_by_id)
            pol_vecs = embed_with_cost("embed_policies", pool,
                                       [pol_text_by_id[p] for p in pool],
                                       "gpps_policies_v1")        # NEW tag for this lane
            alert_vecs = embed_with_cost("embed_alerts", rec_list,
                                         [alert_text_by_id[r] for r in rec_list],
                                         "alerts_policy_lane")    # NEW tag -- do NOT reuse
            P, A = np.asarray(pol_vecs), np.asarray(alert_vecs)
            ppos = {p: i for i, p in enumerate(pool)}
            for i, rid in enumerate(rec_list):
                child = cand_by_rec[rid]
                sims  = P[[ppos[p] for p in child]] @ A[i]
                order = np.argsort(-sims) if top_k is None else np.argsort(-sims)[:top_k]
                cand_by_rec[rid] = [child[j] for j in order]
            used_embeddings = True
        except Exception as e:
            print(f"[C] embedding ordering skipped ({type(e).__name__}: {e}); linkage order.")
    if not used_embeddings and top_k:
        cand_by_rec = {r: c[:top_k] for r, c in cand_by_rec.items()}

    units = []
    for rid in rec_list:
        cands = cand_by_rec[rid]
        for i in range(0, len(cands), chunk):
            units.append([rid, i // chunk, cands[i:i + chunk]])

    total_pairs = sum(len(u[2]) for u in units)
    if total_pairs > MAX_PAIRS_GUARD:
        raise ValueError(f"plan would judge {total_pairs:,} pairs (> MAX_PAIRS_GUARD="
                         f"{MAX_PAIRS_GUARD:,}). Set LANE['top_k'] for a smoke run, or "
                         f"raise the guard deliberately.")

    # --- upstream reachability: golden policies reachable via the LINKED RSMs.
    # With top_k=None there is no truncation, so a miss here is an UPSTREAM gap:
    # the policy's parent RSM was not LINKED, or the linkage has no row for it.
    golden_by_rec = golden.groupby("RECORD_ID")["POLICY_ID"].apply(set).to_dict()
    g_total = g_reached = 0
    for rid in rec_list:
        gp = {p for p in golden_by_rec.get(rid, set()) if p in reachable_policies}
        g_total   += len(gp)
        g_reached += len(gp & set(cand_by_rec[rid]))
    RETRIEVAL_STATS["upstream_recall"] = g_reached / max(g_total, 1)
    print(f"[C] upstream recall (golden policy reachable via LINKED RSMs): "
          f"{g_reached:,}/{g_total:,} = {RETRIEVAL_STATS['upstream_recall']:.1%}")

    plan = {"run_id": RUN_ID, "comment": RUN_COMMENT, "model": MODEL,
            "chunk": chunk, "top_k": top_k,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "prompt_sha": PROMPT_SHA, "upstream_recall": RETRIEVAL_STATS["upstream_recall"],
            "records": cand_by_rec, "units": units}
    PLAN_FILE.write_text(json.dumps(plan))
    write_team_cost_file()          # embedding spend lands before any judge call
    print(f"[C] plan built: {len(units):,} units over {len(rec_list):,} records, "
          f"{total_pairs:,} pairs, chunk={chunk}, "
          f"ordering={'cosine' if used_embeddings else 'linkage order'}")
    return plan


def _judge_call_kwargs():
    """Per-model-family kwargs (GPT vs Gemini) for the policy judge call."""
    kw = {"model": MODEL, "user": USE_CASE, "response_format": POLICY_SCHEMA}
    if JUDGE_LOGPROBS:
        kw["logprobs"] = True; kw["top_logprobs"] = 5
    if "gemini" in MODEL.lower():
        kw.update(temperature=0.0, seed=42,
                  reasoning_effort=LANE["reasoning"], max_tokens=LANE["max_tokens"])
    else:  # gpt-family
        kw.update(reasoning_effort=LANE["reasoning"],
                  max_completion_tokens=LANE["max_tokens"])
    return kw


def _parse_items(r):
    return json.loads(r.choices[0].message.content)["items"]


async def _judge_one_chunk(alert_text, policy_chunk, rid=None, unit=None):
    """Judge one alert against a chunk of policies. Returns (items, last_error).
    policy_chunk: list of (policy_id, policy_text). Every attempt is costed in
    chat_with_cost; retries + token refresh on 401 as before."""
    payload = {"alert_text": alert_text,
               "policies": [{"policy_id": p, "policy_text": t} for p, t in policy_chunk]}
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)}]
    last_err = None
    for attempt in range(1, LANE["max_retries"] + 1):
        try:
            items, _ = await chat_with_cost(
                "judge", messages, _judge_call_kwargs(), parse=_parse_items,
                record_id=rid, unit=unit, attempt=attempt, n_items=len(policy_chunk))
            return items, None
        except Exception as e:
            last_err = e
            # a 401 means the token went stale mid-flight -> force refresh and retry
            if "401" in str(e) or "unauth" in str(e).lower():
                try:
                    await tok.invalidate()
                except Exception:
                    pass
            await asyncio.sleep(1.5 ** attempt)

    # all retries failed -> emit ERROR rows so the batch never crashes
    return ([{"policy_id": p, "decision": "ERROR", "relevance_score": 0,
              "reasoning": f"n/a: {last_err}"} for p, _ in policy_chunk], last_err)


async def run_policy_linkage(plan, resume=True):
    """Execute the plan with checkpointing. Safe to call repeatedly."""
    pol_text_by_id   = dict(zip(policies.POLICY_ID, policies.policy_text))
    alerts_df        = load_alerts(set(plan["records"]))
    alert_text_by_id = dict(zip(alerts_df.RECORD_ID, alerts_df.alert_text))

    done, rows_on_disk = (load_completed() if resume else (set(), []))
    pending = [(rid, idx, cands) for rid, idx, cands in plan["units"]
               if f"{rid}#{idx}" not in done]

    print(f"[C] units total={len(plan['units']):,}  done={len(done):,}  "
          f"pending={len(pending):,}  (rows already on disk: {len(rows_on_disk):,})")
    if not pending:
        print("[C] nothing pending -- assembling from checkpoint.")
        return _finalise(rows_on_disk)

    sem       = asyncio.Semaphore(LANE["concurrency"])
    write_lok = asyncio.Lock()
    counter   = {"n": len(done), "rows": list(rows_on_disk)}

    async def _one(rid, idx, pol_ids):
        key = f"{rid}#{idx}"
        async with sem:                                   # cap concurrency
            pairs = [(p, pol_text_by_id.get(p, "")) for p in pol_ids]
            items, _err = await _judge_one_chunk(alert_text_by_id.get(rid, ""), pairs,
                                                 rid=rid, unit=key)
            got = {_nid(it.get("policy_id", "")): it for it in items}
            rows = []
            # iterate the ORIGINAL chunk so we always emit a row per candidate,
            # even if the model skipped / reordered / hallucinated an id
            for pid, _ in pairs:
                it = got.get(pid)
                rows.append({"RECORD_ID": rid, "POLICY_ID": pid,
                             "decision": (it or {}).get("decision", "MISSING"),
                             "relevance_score": (it or {}).get("relevance_score", 0),
                             "reasoning": (it or {}).get("reasoning")})
        # ---- checkpoint: one fsync'd line per completed unit -----------------
        async with write_lok:
            _append_jsonl(VERDICTS_JSONL, {"unit": key, "ts": time.time(), "rows": rows})
            counter["n"] += 1
            counter["rows"].extend(rows)
            if counter["n"] % SNAPSHOT_EVERY == 0:
                _snapshot(counter["rows"], counter["n"], len(plan["units"]))
        return len(rows)

    tasks = [asyncio.create_task(_one(rid, idx, c)) for rid, idx, c in pending]
    try:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="judge units"):
            await fut
    except BaseException:
        # kernel interrupt / disconnect: stop the rest cleanly. Everything
        # already written -- verdicts AND cost ledger -- survives.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _snapshot(counter["rows"], counter["n"], len(plan["units"]))
        print(f"[C] INTERRUPTED at {counter['n']}/{len(plan['units'])} units. "
              f"Checkpoint intact -- re-run cells [0] -> [C] to finish.")
        print("    If you interrupted with Ctrl-C in a LIVE kernel, restart the kernel")
        print("    first: the aborted coroutine can linger on the event loop and re-raise")
        print("    into the next run. After a VDI disconnect just re-run the cells.")
        raise
    return _finalise(counter["rows"])


# ---- Execute -----------------------------------------------------------------
plan = build_plan(scope_records, top_k=LANE["top_k"])

verdicts = await run_policy_linkage(plan)

# sanity: truncated responses / failures, straight from the cost ledger
_led = pd.DataFrame([r for r in _read_jsonl(COST_LEDGER)
                     if r.get("run_id") == RUN_ID and r.get("step") == "judge"])
if len(_led):
    print("[C] judge attempts by status:", _led.status.value_counts().to_dict(),
          "| finish_reason:", _led.finish_reason.value_counts(dropna=False).to_dict())


# %% ==========================================================================
# [D] PERFORMANCE -- GOLDEN ENTERS HERE, FOR VALIDATION ONLY
#
# The golden is HAND-PICKED and NON-EXHAUSTIVE, and its record->policy pairs are
# derived at REGULATION level. Two consequences for reporting:
#   1. A golden policy link is verified truth -> we CAN measure RECALL.
#   2. A predicted link NOT in golden is NOT a false positive -> DISCOVERY.
#
# THREE honest numbers:
#   A) ADDRESSABLE RECALL = validated / golden links whose policy is published
#   B) DATA GAP           = golden links to unpublished / absent policies
#   C) DISCOVERY VOLUME   = predicted links not in golden
#   + RANKING             = hit@1 / hit@3 / MRR per alert
# =============================================================================
def _best_verdicts(verdicts_df):
    v = verdicts_df.copy()
    v["REC"], v["POL"] = _norm_id(v.RECORD_ID), _norm_id(v.POLICY_ID)
    v["DEC"] = v.decision.astype(str).str.strip().str.upper()
    v["_r"]  = v.DEC.map(DECISION_RANK).fillna(-1)
    v["_s"]  = pd.to_numeric(v.relevance_score, errors="coerce").fillna(0)
    return (v.sort_values(["REC", "_r", "_s"], ascending=[True, False, False])
              .drop_duplicates(["REC", "POL"]))


def policy_scorecard(golden_df, verdicts_df, reachable):
    """Print the honest scorecard + return the per-golden-pair labelled frame."""
    g = (golden_df.rename(columns={"RECORD_ID": "REC", "POLICY_ID": "POL"})
                  [["REC", "POL"]].drop_duplicates())
    g = g[~g.POL.isin(_BLANK)]
    v = _best_verdicts(verdicts_df)[["REC", "POL", "DEC"]]

    g = g[g.REC.isin(set(v.REC))].copy()           # scope to records we actually ran
    g["reachable"]     = g.POL.isin(reachable)
    golden_total       = len(g)
    golden_addressable = int(g.reachable.sum())
    data_gap           = golden_total - golden_addressable

    m = g.merge(v, on=["REC", "POL"], how="left")
    m["gate"] = m.DEC.where(m.DEC.notna(), "NOT_JUDGED").map(
        {"NOT_JUDGED": "NOT_JUDGED", "LINKED": "CAUGHT", "NOT_LINKED": "JUDGED_NOT_LINKED",
         "INSUFFICIENT_EVIDENCE": "JUDGED_INSUFFICIENT", "ERROR": "JUDGED_ERROR",
         "MISSING": "JUDGED_ERROR"}).fillna("NOT_JUDGED")

    addr               = m[m.reachable]
    validated          = int((addr.gate == "CAUGHT").sum())
    addressable_recall = validated / max(golden_addressable, 1)

    golden_keys = set(g.REC + "|" + g.POL)
    linked      = v[v.DEC == "LINKED"]
    discoveries = len(set(linked.REC + "|" + linked.POL) - golden_keys)

    bar = "=" * 74
    print(bar); print("[D] POLICY SCORECARD (golden = derived at regulation level)"); print(bar)
    print(f"  Golden policy links (verified truth)      : {golden_total:,}")
    print(f"    ADDRESSABLE (policy published)          : {golden_addressable:,}   <- recall denominator")
    print(f"    DATA GAP    (not published / absent)    : {data_gap:,}")
    print("-" * 60)
    print("  A) RECALL (on addressable golden):")
    print(f"       VALIDATED (model linked)       : {validated:,}")
    for gate in ["JUDGED_NOT_LINKED", "JUDGED_INSUFFICIENT", "NOT_JUDGED", "JUDGED_ERROR"]:
        c = int((addr.gate == gate).sum())
        if c: print(f"       {gate:22s}: {c:,}")
    print(f"       >>> ADDRESSABLE RECALL = {validated:,}/{golden_addressable:,} "
          f"= {addressable_recall:.1%}")
    print("-" * 60)
    print(f"  B) DATA GAP (carve-out, not model fault) : {data_gap:,} "
          f"({data_gap/max(golden_total,1):.1%} of golden)")
    print("-" * 60)
    print(f"  C) DISCOVERIES (predicted, not in golden): {discoveries:,}")
    print("       -> NOT false positives (golden non-exhaustive); candidate NEW links")
    print("          for SME review. Audit them via [F] slice 'false_positives'.")
    print()
    print("  CAVEAT TO CARRY INTO EVERY DECK: the golden has no RSM, so its pairs are")
    print("  regulation-level. Section [F] splits the misses into model fault vs")
    print("  STRUCTURAL_OVERMAPPING before any headline recall number is quoted.")

    globals()["SCORECARD_COUNTS"] = {
        "addressable":       golden_addressable,
        "validated":         validated,
        "judged_not_linked": int((addr.gate == "JUDGED_NOT_LINKED").sum()),
        "insufficient":      int((addr.gate == "JUDGED_INSUFFICIENT").sum()),
        "not_judged":        int((addr.gate == "NOT_JUDGED").sum()),
        "golden_total":      golden_total,
        "data_gap":          data_gap,
        "discoveries":       discoveries,
    }
    return m


def ranking_metrics(verdicts_df, ks=(1, 3, 5)):
    """Per alert: rank policies by (decision, relevance_score); where does the
    best golden policy land? The 'did it find ITS policy' number."""
    v = _best_verdicts(verdicts_df)
    v["pos"] = v.groupby("REC").cumcount() + 1
    g = golden[golden.RECORD_ID.isin(set(v.REC)) & golden.POLICY_ID.isin(reachable_policies)]
    m = g.merge(v[["REC", "POL", "pos"]], left_on=["RECORD_ID", "POLICY_ID"],
                right_on=["REC", "POL"], how="left")
    best = m.groupby("RECORD_ID").pos.min()             # NaN -> never judged
    n = len(best)
    out = {f"hit@{k}": float((best <= k).mean()) if n else 0.0 for k in ks}
    out["MRR"] = float((1 / best).fillna(0).mean()) if n else 0.0
    out["records"] = n
    print("-" * 60)
    print(f"  RANKING over {n:,} records with an addressable golden policy:")
    print("    " + "   ".join(f"{k} {val:.1%}" for k, val in out.items()
                              if k not in ("records", "MRR")) + f"   MRR {out['MRR']:.3f}")
    globals()["RANKING_METRICS"] = out
    return out


perf    = policy_scorecard(golden, verdicts, reachable_policies)
ranking = ranking_metrics(verdicts)


# %% ==========================================================================
# [E] DIAGNOSTIC -- where do the addressable-recall misses go?
# =============================================================================
def diagnose(perf_df):
    addr  = perf_df[perf_df.reachable].copy()
    total = len(addr)
    parts = {k: int((addr.gate == k).sum()) for k in
             ["CAUGHT", "NOT_JUDGED", "JUDGED_NOT_LINKED",
              "JUDGED_INSUFFICIENT", "JUDGED_ERROR"]}

    bar = "=" * 70
    print(bar); print("[E] DIAGNOSIS -- POLICY LANE"); print(bar)
    print(f"  addressable golden pairs   : {total:,}")
    for k, val in parts.items():
        print(f"    {k:22s}: {val:,}  ({val/max(total,1):.1%})")
    print("-" * 60)
    if parts["NOT_JUDGED"] > 0.05 * max(total, 1):
        print(f"  NOTE {parts['NOT_JUDGED']:,} reachable golden policies were NOT judged.")
        print("       With no top-k cap this means their parent RSM was not LINKED")
        print("       upstream, or the linkage file has no RSM->Policy row for them.")
        print(f"       CANDIDATE-BUILD / upstream issue, not the prompt. (upstream recall = "
              f"{RETRIEVAL_STATS.get('upstream_recall', float('nan')):.1%})")
    else:
        print("  NOT_JUDGED is small, as designed -- the misses below are MODEL decisions")
        print("  or structural over-mapping. Send them to [F].")
    if parts["JUDGED_ERROR"]:
        print(f"  {parts['JUDGED_ERROR']:,} pairs errored. Delete their units from "
              f"{VERDICTS_JSONL.name} and re-run to redo them.")
    return addr


perf_addr = diagnose(perf)


# %% ==========================================================================
# [F] LLM-AS-JUDGE AUDIT LAYER (the MRM story)
# Sub-reasons are POLICY-shaped -- the LR lane's procedure-shaped reasons
# misclassify policy misses. Every audit call goes through chat_with_cost, so
# audit spend lands in the same cost file under step="audit".
# =============================================================================
AUDIT_BUCKETS = ["TRUE_LINK_MISSED", "CORRECT_REJECTION", "SME_JUDGEMENT_REQUIRED",
                 "STRUCTURAL_OVERMAPPING", "AMBIGUOUS"]

SUB_REASONS = {
    "TRUE_LINK_MISSED": [
        "MODEL_ERROR_TEXT_SUPPORTS_LINK",   # the policy text does support the link
        "POLICY_TEXT_TOO_THIN",             # title/purpose only, body empty or boilerplate
        "TRUNCATED_POLICY_BODY",            # the relevant clause fell outside the char budget
        "IMPLICIT_DOMAIN_KNOWLEDGE",        # needs knowledge not present in either text
    ],
    "CORRECT_REJECTION": [
        "DIFFERENT_SUBJECT_MATTER",
        "JURISDICTION_MISMATCH",            # country-specific policy, different jurisdiction
        "WRONG_RISK_TYPE",                  # policy governs another primary risk type
    ],
    "SME_JUDGEMENT_REQUIRED": [
        "REQUIRES_POLICY_OWNER_VIEW",
        "INDIRECT_DOWNSTREAM_IMPACT",
    ],
    "STRUCTURAL_OVERMAPPING": [
        "REGULATION_LEVEL_SWEEP_IN",        # golden has no RSM -> whole-RGL policy set
        "HORIZONTAL_POLICY_BLANKET",        # generic compliance policy attached to everything
        "POLICY_FAMILY_LEVEL_MAPPING",      # mapped to a family, not this specific policy
    ],
    "AMBIGUOUS": ["INSUFFICIENT_INFORMATION"],
}
ALL_SUB_REASONS = [s for subs in SUB_REASONS.values() for s in subs]

AUDIT_SYSTEM_PROMPT = f"""ROLE:
You are an independent model-risk reviewer. A linkage model judged whether a
regulatory alert bears on an internal POLICY. A hand-built golden mapping
disagrees with the model. Decide WHY, from the two texts alone.

Return exactly one bucket and one sub-reason.

BUCKETS: {", ".join(AUDIT_BUCKETS)}
SUB_REASONS: {json.dumps(SUB_REASONS)}

GUIDANCE:
- The golden mapping is recorded at REGULATION level and carries no summary id,
  so a policy can appear 'golden-true' merely because it sits under the same
  regulation. If the alert itself gives no reason to touch this policy, that is
  STRUCTURAL_OVERMAPPING, not a model error.
- Judge only on the supplied texts. If the link needs knowledge that is in
  neither text, that is IMPLICIT_DOMAIN_KNOWLEDGE or SME_JUDGEMENT_REQUIRED.
- Policy text may be truncated; if the decisive clause is plainly missing, say
  TRUNCATED_POLICY_BODY.
Return JSON only."""

AUDIT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "audit_verdict", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["audit_category", "sub_reason", "explanation"],
            "properties": {
                "audit_category": {"type": "string", "enum": AUDIT_BUCKETS},
                "sub_reason":     {"type": "string", "enum": ALL_SUB_REASONS},
                "explanation":    {"type": ["string", "null"]},
            }}}
}


def _audit_call_kwargs():
    kw = {"model": MODEL, "user": USE_CASE, "response_format": AUDIT_SCHEMA}
    if "gemini" in MODEL.lower():
        kw.update(temperature=0.0, seed=42, reasoning_effort="low", max_tokens=4000)
    else:
        kw.update(reasoning_effort="low", max_completion_tokens=4000)
    return kw


async def _audit_one(alert_text, policy_text, golden_says_linked, model_decision,
                     model_reasoning, rid=None, pid=None):
    payload = {"alert_text": alert_text, "policy_text": policy_text,
               "golden_says_linked": bool(golden_says_linked),
               "model_decision": model_decision, "model_reasoning": model_reasoning}
    messages = [{"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)}]
    for attempt in range(1, 3 + 1):
        try:
            parsed, _ = await chat_with_cost(
                "audit", messages, _audit_call_kwargs(),
                parse=lambda r: json.loads(r.choices[0].message.content),
                record_id=rid, unit=f"audit:{rid}|{pid}", attempt=attempt, n_items=1)
            return parsed
        except Exception as e:
            if "401" in str(e) or "unauth" in str(e).lower():
                try:
                    await tok.invalidate()
                except Exception:
                    pass
            await asyncio.sleep(1.5 ** attempt)
            last = e
    return {"audit_category": "AMBIGUOUS", "sub_reason": "INSUFFICIENT_INFORMATION",
            "explanation": f"audit failed: {last}"}


def select_audit_slice(perf_df, verdicts_df, which="false_negatives", sample=None, seed=42):
    """Rows to audit. 'false_negatives' = addressable golden pairs the model did
    not link -- the miss pile that decides the honest recall number."""
    if which == "false_negatives":
        sl = perf_df[perf_df.reachable & perf_df.gate.isin(
            ["JUDGED_NOT_LINKED", "JUDGED_INSUFFICIENT"])].copy()
        sl["golden_says_linked"] = True
    elif which == "false_positives":
        v = _best_verdicts(verdicts_df)
        gk = set(perf_df.REC + "|" + perf_df.POL)
        sl = v[(v.DEC == "LINKED") & ~(v.REC + "|" + v.POL).isin(gk)].copy()
        sl["gate"] = "DISCOVERY"; sl["golden_says_linked"] = False
    else:
        raise ValueError("which must be 'false_negatives' or 'false_positives'")
    if sample and len(sl) > sample:
        sl = sl.sample(sample, random_state=seed)
    print(f"[F] audit slice '{which}': {len(sl):,} pairs")
    return sl.reset_index(drop=True)


async def run_audit(slice_df, alert_text_by_id, policy_text_by_id, verdicts_df,
                    concurrency=20):
    v = _best_verdicts(verdicts_df).set_index(["REC", "POL"])
    sem = asyncio.Semaphore(concurrency)

    async def _one(row):
        async with sem:
            key = (row.REC, row.POL)
            reasoning = v.reasoning.get(key) if "reasoning" in v.columns else None
            dec       = v.DEC.get(key, "NOT_JUDGED")
            out = await _audit_one(alert_text_by_id.get(row.REC, ""),
                                   policy_text_by_id.get(row.POL, ""),
                                   row.golden_says_linked, dec, reasoning,
                                   rid=row.REC, pid=row.POL)
            return {"REC": row.REC, "POL": row.POL, "model_decision": dec, **out}

    tasks = [asyncio.create_task(_one(r)) for r in slice_df.itertuples()]
    out = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="audit"):
        out.append(await fut)
    write_team_cost_file()
    return pd.DataFrame(out)


def audit_report(audited_df):
    if not len(audited_df):
        print("[F] nothing audited."); return None
    bar = "=" * 70
    print(bar); print("[F] AUDIT OF THE MISS PILE"); print(bar)
    cat = audited_df.audit_category.value_counts()
    for k, c in cat.items():
        print(f"  {k:24s}: {c:,}  ({c/len(audited_df):.1%})")
    print("-" * 60)
    print(audited_df.sub_reason.value_counts().to_string())
    return cat


# ---- Execute (comment out while iterating on the judge prompt) ---------------
AUDIT_SLICE  = "false_negatives"
AUDIT_SAMPLE = None                 # an int for a fast estimate, or None for all

_alert_txt  = dict(zip(*(lambda d: (d.RECORD_ID, d.alert_text))(load_alerts(scope_records))))
_policy_txt = dict(zip(policies.POLICY_ID, policies.policy_text))
audit_slice = select_audit_slice(perf, verdicts, AUDIT_SLICE, AUDIT_SAMPLE)

audited = await run_audit(audit_slice, _alert_txt, _policy_txt, verdicts)
_ = audit_report(audited)


# %% ==========================================================================
# [F1] HONEST RECALL -- run AFTER [F], i.e. once `audited` exists.
# Strips the misses the audit says were never text-derivable, so the number the
# deck carries is the one the model can actually be held to.
# =============================================================================
def honest_recall(scorecard_counts, audited_df):
    k = scorecard_counts
    addr = max(k["addressable"], 1)
    n = len(audited_df)
    share = (audited_df.audit_category.value_counts() / max(n, 1)).to_dict() if n else {}

    miss = k["judged_not_linked"] + k["insufficient"]
    overmap = int(round(miss * share.get("STRUCTURAL_OVERMAPPING", 0)))
    correct = int(round(miss * share.get("CORRECT_REJECTION", 0)))
    sme     = int(round(miss * share.get("SME_JUDGEMENT_REQUIRED", 0)))

    text_derivable = k["validated"] / max(addr - overmap - correct, 1)
    model_addressable = k["validated"] / max(addr - overmap - correct - sme, 1)

    bar = "=" * 70
    print(bar); print("[F1] HONEST RECALL"); print(bar)
    print(f"  addressable golden pairs        : {addr:,}")
    print(f"  ... structural over-mapping     : {overmap:,}")
    print(f"  ... correct rejections          : {correct:,}")
    print(f"  ... SME judgement required      : {sme:,}")
    print(f"  NAIVE recall                    : {k['validated']/addr:.1%}")
    print(f"  TEXT-DERIVABLE recall           : {text_derivable:.1%}")
    print(f"  MODEL-ADDRESSABLE recall        : {model_addressable:.1%}")
    return {"text_derivable": text_derivable, "addressable": model_addressable,
            "overmapping": overmap, "correct_rejection": correct, "sme": sme}


honest = honest_recall(SCORECARD_COUNTS, audited)


# %% ==========================================================================
# [G] COST REPORT -> THE COST FILE
# Reads the ledger, RE-PRICES every row with the current PRICING table, writes:
#   cost/cost_log_<COST_RUN_ID>.parquet   team format (kind,in,out,ts,...)
#   cost_log_<MODEL>.csv                  one row per call attempt (detail)
#   cost_summary_<MODEL>.csv              run x step x kind x model roll-up
# Re-run this cell any time -- mid-run, after [F], after fixing a price.
# =============================================================================
def cost_report(save=True, verbose=True):
    rows = _read_jsonl(COST_LEDGER)
    if not rows:
        print("[G] cost ledger is empty."); return None, None
    df = pd.DataFrame(rows)
    for c in ["prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["cost_usd"] = pd.to_numeric(pd.Series(
        [_usd(m, p, c, o) for m, p, c, o in zip(df.model, df.prompt_tokens,
                                                df.cached_tokens, df.completion_tokens)],
        index=df.index, dtype="object"), errors="coerce")
    df["wasted"] = df.status.ne("ok")

    _sum_or_nan = lambda s: s.sum(min_count=1)
    summ = (df.groupby(["run_id", "step", "kind", "model"], dropna=False)
              .agg(calls=("status", "size"),
                   ok=("status", lambda s: int((s == "ok").sum())),
                   parse_errors=("status", lambda s: int((s == "parse_error").sum())),
                   call_errors=("status", lambda s: int((s == "call_error").sum())),
                   prompt_tokens=("prompt_tokens", "sum"),
                   cached_tokens=("cached_tokens", "sum"),
                   completion_tokens=("completion_tokens", "sum"),
                   reasoning_tokens=("reasoning_tokens", "sum"),
                   cost_usd=("cost_usd", _sum_or_nan),
                   estimated=("estimated", "max"))
              .reset_index())
    total = {"run_id": "TOTAL", "step": "", "kind": "", "model": "",
             **{c: summ[c].sum() for c in ["calls", "ok", "parse_errors", "call_errors",
                                           "prompt_tokens", "cached_tokens",
                                           "completion_tokens", "reasoning_tokens"]},
             "cost_usd": summ.cost_usd.sum(min_count=1), "estimated": summ.estimated.any()}
    summ = pd.concat([summ, pd.DataFrame([total])], ignore_index=True)

    if verbose:
        bar = "=" * 78
        print(bar); print("[G] COST REPORT"); print(bar)
        print(summ.to_string(index=False))
        cur = df[df.run_id == RUN_ID]
        j = cur[cur.step == "judge"]
        if len(j):
            per_rec = j.groupby("record_id").cost_usd.sum(min_count=1)
            print("-" * 78)
            if per_rec.notna().any():
                print(f"  [{RUN_ID}] judge cost per alert  mean ${per_rec.mean():.4f}   "
                      f"median ${per_rec.median():.4f}   max ${per_rec.max():.4f}")
            else:
                print(f"  [{RUN_ID}] judge cost per alert: unpriced (fill PRICING)")
            waste = cur.loc[cur.wasted, "cost_usd"].sum()
            print(f"  [{RUN_ID}] spend on failed / unparseable attempts: ${waste:.4f}")
            if "projection" in globals() and projection.get("usd_judge"):
                print(f"  [{RUN_ID}] judge actual vs projected: ${j.cost_usd.sum():.2f} vs "
                      f"${projection['usd_judge']:.2f}  (tune EST_OUT_TOKENS_PER_POLICY)")
        if df.estimated.any():
            print("  NOTE embedding rows are locally ESTIMATED upper bounds (cache hits free).")
        if df.cost_usd.isna().any():
            print(f"  NOTE {int(df.cost_usd.isna().sum()):,} rows unpriced -- fill PRICING "
                  f"and re-run this cell; tokens are already on disk.")

    if save:
        write_team_cost_file(verbose=verbose)
        df.to_csv(COST_FILE, index=False)
        summ.to_csv(COST_SUMMARY_FILE, index=False)
        if verbose:
            print(f"  saved -> {COST_FILE}\n           {COST_SUMMARY_FILE}")
    return df, summ


cost_log, cost_summary = cost_report()


# %% ==========================================================================
# [H] PERFORMANCE PACK -- confusion matrix + the run documentation workbook
#   TP  model LINKED and golden agrees
#   FN  model did not link but golden did        -> the miss pile ([F])
#   FP  model LINKED, pair absent from golden    -> DISCOVERY
#   TN  neither linked                           -> contaminated (non-exhaustive golden)
# =============================================================================
def label_cells():
    v = _best_verdicts(verdicts)[["REC", "POL", "DEC"]]
    v = v[v.REC.isin(scope_records)].copy()

    gp = golden[golden.RECORD_ID.isin(scope_records)][["RECORD_ID", "POLICY_ID"]].drop_duplicates()
    gkeys = set(gp.RECORD_ID + "|" + gp.POLICY_ID)

    v["in_golden"] = (v.REC + "|" + v.POL).isin(gkeys)
    v["linked"]    = v.DEC.eq("LINKED")
    v["cell"] = np.select(
        [v.in_golden & v.linked, v.in_golden & ~v.linked,
         ~v.in_golden & v.linked, ~v.in_golden & ~v.linked],
        ["TP", "FN", "FP", "TN"], default="?")

    judged = set(v.REC + "|" + v.POL)
    never  = gp[~(gp.RECORD_ID + "|" + gp.POLICY_ID).isin(judged)].copy()
    never["gap_type"] = np.select(
        [never.POLICY_ID.isin(reachable_policies),
         never.POLICY_ID.isin(policy_ids_any_status)],
        ["PUBLISHED_BUT_UNJUDGED", "NOT_PUBLISHED"], default="ABSENT")

    bar = "=" * 74
    print(bar); print("[H] CONFUSION MATRIX (judged pairs only)"); print(bar)
    cm = (v.pivot_table(index="linked", columns="in_golden", values="REC",
                        aggfunc="count", fill_value=0)
            .rename(index={True: "model LINKED", False: "model NOT LINKED"},
                    columns={True: "golden linked", False: "golden not linked"}))
    print(cm.to_string())
    c = v.cell.value_counts().to_dict()
    tp, fn = c.get("TP", 0), c.get("FN", 0)
    print("-" * 60)
    print(f"  TP {tp:,}   FN {fn:,}   FP {c.get('FP',0):,}   TN {c.get('TN',0):,}")
    print(f"  RECALL on judged pairs = {tp:,}/{tp+fn:,} = {tp/max(tp+fn,1):.1%}")
    print(f"  golden pairs NEVER judged: {len(never):,}  "
          f"{never.gap_type.value_counts().to_dict()}")
    return v, never


def build_funnel_table(cells):
    k = SCORECARD_COUNTS
    tp = int((cells.cell == "TP").sum()); fn = int((cells.cell == "FN").sum())
    addr = max(k["addressable"], 1)
    rows = [
        ("Total Linkages", "Golden Record -> Policy pairs for the records in scope "
         "(derived at regulation level).", k["golden_total"], 1.0),
        ("Linkages Missed (data gap)", "Golden links to a policy that is not published "
         "or absent from the extract -- cannot be judged.",
         k["data_gap"], k["data_gap"]/max(k["golden_total"], 1)),
        ("Addressable golden pairs", "Total less the data gap. Recall denominator.",
         k["addressable"], 1.0),
        ("Not Judged (never shown)", "Parent RSM not LINKED upstream, or no RSM->Policy row.",
         k["not_judged"], k["not_judged"]/addr),
        ("Judged Insufficient", "Text too sparse to evidence either verdict.",
         k["insufficient"], k["insufficient"]/addr),
        ("Judged Not Linked", "Model saw the pair and did not link it.",
         k["judged_not_linked"], k["judged_not_linked"]/addr),
        ("Caught (validated)", "Model linked and golden confirms.",
         k["validated"], k["validated"]/addr),
        ("Addressable Recall", "Caught / addressable golden links.",
         k["validated"], k["validated"]/addr),
        ("Recall on judged pairs", "Caught / golden links actually judged.",
         tp, tp/max(tp+fn, 1)),
        ("Upstream recall", "Golden policies reachable via the record's LINKED RSMs.",
         k["addressable"], RETRIEVAL_STATS.get("upstream_recall", float("nan"))),
    ]
    if "RANKING_METRICS" in globals():
        rm = RANKING_METRICS
        rows += [(f"Ranking {kk}", "Share of alerts whose golden policy ranks within k.",
                  rm["records"], vv) for kk, vv in rm.items() if kk.startswith("hit@")]
        rows += [("Ranking MRR", "Mean reciprocal rank of the best golden policy.",
                  rm["records"], rm["MRR"])]
    if "honest" in globals() and honest:
        rows += [("Text Derivable Recall", "Excludes over-mapping + correct rejections.",
                  k["validated"], honest["text_derivable"]),
                 ("Model-Addressable Recall", "Further excludes SME-only links.",
                  k["validated"], honest["addressable"])]
    return pd.DataFrame(rows, columns=["Category", "Explanation", "Counts", "Percentage"])


def build_by_policy(cells):
    """Where the misses concentrate -- the table that says what to fix next."""
    t = (cells.pivot_table(index="POL", columns="cell", values="REC",
                           aggfunc="count", fill_value=0).reset_index())
    for col in ["TP", "FN", "FP", "TN"]:
        if col not in t.columns:
            t[col] = 0
    t = t.rename(columns={"TP": "caught", "FN": "missed", "FP": "discoveries",
                          "TN": "judged_not_linked_nongolden"})
    t["golden_pairs"] = t.caught + t.missed
    t["recall"] = np.where(t.golden_pairs > 0,
                           t.caught / t.golden_pairs.replace(0, np.nan), np.nan)
    t = t.merge(policies[["POLICY_ID", "POL_TITLE", "COUNTRY", "POL_RISK_TYPE"]],
                left_on="POL", right_on="POLICY_ID", how="left").drop(columns="POLICY_ID")
    return (t.sort_values("missed", ascending=False)
             [["POL", "POL_TITLE", "COUNTRY", "POL_RISK_TYPE", "golden_pairs",
               "caught", "missed", "recall", "discoveries"]].reset_index(drop=True))


def performance_pack(path=None):
    """Write the documentation workbook for THIS run. Every number is derived."""
    cells, never = label_cells()
    c = cells.cell.value_counts().to_dict()
    tp, fn = c.get("TP", 0), c.get("FN", 0)
    k = SCORECARD_COUNTS
    led, summ = cost_report(save=True, verbose=False)
    cur = led[led.run_id == RUN_ID] if led is not None else pd.DataFrame()
    run_usd = cur.cost_usd.sum(min_count=1) if len(cur) else None

    meta = pd.DataFrame([
        ("Run ID", RUN_ID), ("Comment", RUN_COMMENT), ("Model", MODEL),
        ("Prompt file", POLICY_PROMPT_PATH.name), ("Prompt SHA (first 12)", PROMPT_SHA),
        ("Run at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Records in scope", len(scope_records)),
        ("Policies judged", int(cells.POL.nunique())),
        ("Alert-policy pairs judged", len(cells)),
        ("top_k", str(LANE["top_k"])), ("chunk", LANE["chunk"]),
        ("concurrency", LANE["concurrency"]),
        ("Policy body char budget", POLICY_TEXT_MAX_CHARS),
        ("Golden pairs (in scope)", k["golden_total"]),
        ("Addressable golden pairs", k["addressable"]),
        ("Data gap", k["data_gap"]),
        ("Golden pairs never judged", len(never)),
        ("Upstream recall", RETRIEVAL_STATS.get("upstream_recall")),
        ("VALIDATED (TP)", tp), ("MISSED (FN)", fn),
        ("DISCOVERIES (FP)", c.get("FP", 0)), ("TN", c.get("TN", 0)),
        ("Recall on judged pairs", tp/max(tp+fn, 1)),
        ("Recall on addressable golden", k["validated"]/max(k["addressable"], 1)),
        ("Run cost USD (all steps, this run)",
         "unpriced" if run_usd is None or pd.isna(run_usd) else round(float(run_usd), 4)),
        ("Cost file", str(COST_PARQUET)),
    ], columns=["Field", "Value"])

    funnel = build_funnel_table(cells)
    by_pol = build_by_policy(cells)
    cm = pd.DataFrame([["Model = LINKED", tp, c.get("FP", 0)],
                       ["Model = NOT LINKED", fn, c.get("TN", 0)]],
                      columns=["", "Golden = linked", "Golden = not linked"])

    path = Path(path or (OUTPUT_DIR / f"performance_pack_policy_{MODEL}.xlsx"))
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        meta.to_excel(xl, sheet_name="Run_summary", index=False)
        funnel.to_excel(xl, sheet_name="Performance_metrics", index=False)
        cm.to_excel(xl, sheet_name="Confusion_matrix", index=False)
        by_pol.to_excel(xl, sheet_name="By_policy", index=False)
        never.to_excel(xl, sheet_name="Never_judged", index=False)
        cells[cells.cell == "FP"].to_excel(xl, sheet_name="Discoveries", index=False)
        cells[cells.cell == "FN"].to_excel(xl, sheet_name="Missed_FN", index=False)
        if summ is not None:
            summ.to_excel(xl, sheet_name="Cost_summary", index=False)
            cur.to_excel(xl, sheet_name="Cost_ledger", index=False)
        if "audited" in globals() and audited is not None and len(audited):
            (audited.audit_category.value_counts().rename_axis("audit_category")
                .reset_index(name="count").to_excel(xl, sheet_name="Audit_drilldown", index=False))
            audited.to_excel(xl, sheet_name="Audit_rows", index=False)

        for ws in xl.book.worksheets:
            for col, width in zip("ABCDEFGHI", [40, 60, 14, 14, 14, 14, 14, 14, 14]):
                ws.column_dimensions[col].width = width
            for row in ws.iter_rows():
                for cellobj in row:
                    if isinstance(cellobj.value, float) and -1.01 <= cellobj.value <= 1.01:
                        cellobj.number_format = "0.0%"

    print(f"\n[H] performance pack -> {path}")
    return {"cells": cells, "never_judged": never, "funnel": funnel,
            "by_policy": by_pol, "path": path}


pack = performance_pack()


# %% ==========================================================================
# [P-after] SECOND PROVENANCE CHECK -- same function, run at the end.
# =============================================================================
run_provenance_check("AFTER [H] -- compare against the baseline printout")
