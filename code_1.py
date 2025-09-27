# Databricks / Fabric PySpark Notebook 1 — AOAI end-to-end
# ==========================================================
# Build base tables, apply cumulative mapping, compute similarities,
# route candidates, prepare AOAI LLM jobs & results (with optional stepped batching),
# and manage review queue (append-only log; no overwrites). No merges here (Notebook 2 does that).
#
# Key upgrades vs. prior draft:
# - Uses Azure OpenAI for BOTH embeddings and LLM judgments (no stubs).
# - Rerun scopes: ALL (default), FOCAL, or COMPONENT graph of focals connected via queued pairs.
# - Full prompt metadata tracked end-to-end: model/prompt_version/decided_at + context_used/context_hash + prompt_id + aoai_request_id + token usage.
# - Strict APPEND-ONLY writes for all tables; consumers can compute latest rows via windowing when needed.

import os, json, math, time, hashlib
from typing import Any, Dict, List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

# ===== Spark =====
spark = SparkSession.builder.getOrCreate()

# -----------------
# Config / params
# -----------------
T_AUTO = float(os.environ.get("T_AUTO", "0.95"))          # ≥95% auto-merge band
T_LLM_LOW = float(os.environ.get("T_LLM_LOW", "0.70"))    # LLM band lower bound
LLM_TOP_N = int(os.environ.get("LLM_TOP_N", "3"))         # top-N neighbors kept per focal
# If batch size < top N, split LLM requests into multiple "jobs" per focal
LLM_BATCH_SIZE = max(1, int(os.environ.get("LLM_BATCH_SIZE", str(LLM_TOP_N))))  # e.g., 1 → send 1 candidate per request

# Rerun scope: 'focal' | 'component' | 'all' (default all)
RERUN_SCOPE = os.environ.get("RERUN_SCOPE", "all").lower()
RERUN_FOCALS = [x.strip() for x in os.environ.get("RERUN_FOCALS", "").split(",") if x.strip()]  # optional focal seeds

RUN_ID = os.environ.get("RUN_ID", f"run_{int(time.time())}")
BASE_DIR = os.environ.get("BASE_DIR", "/lake/dedupe_demo")  # change for Fabric Lakehouse path

# Rerun controls (optional)
RERUN_ALL = os.environ.get("RERUN_ALL_LLM_QUEUED", "false").lower() == "true"
RERUN_PAIR_KEYS = [x.strip() for x in os.environ.get("RERUN_PAIR_KEYS", "").split(",") if x.strip()]
RERUN_NOTES_JSON = os.environ.get("RERUN_NOTES_JSON", "")  # {"pair_key":"note", ...}
FOCAL_NOTES_JSON = os.environ.get("FOCAL_NOTES_JSON", "")  # {"<focal_master_id>":"shared guidance", ...}
GLOBAL_NOTES = os.environ.get("GLOBAL_NOTES", "").strip()
AUTO_APPROVE_RERUN = os.environ.get("AUTO_APPROVE_RERUN_YES", "true").lower() == "true"
AUTO_APPROVE_RERUN_YES_CONF = float(os.environ.get("AUTO_APPROVE_RERUN_YES_CONF", "0.9"))

try:
    RERUN_NOTES = json.loads(RERUN_NOTES_JSON) if RERUN_NOTES_JSON else {}
except Exception:
    RERUN_NOTES = {}
try:
    FOCAL_NOTES = json.loads(FOCAL_NOTES_JSON) if FOCAL_NOTES_JSON else {}
except Exception:
    FOCAL_NOTES = {}

# AOAI config from environment (Managed Identity via DefaultAzureCredential)
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_CHAT_DEPLOYMENT = os.environ.get("AOAI_CHAT_DEPLOYMENT")
AOAI_EMBEDDING_DEPLOYMENT = os.environ.get("AOAI_EMBEDDING_DEPLOYMENT")
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION", "2024-08-01-preview")

# ---------- Paths / table helpers ----------

def path(table: str) -> str:
    return f"{BASE_DIR}/{table}"

# CDC table names (parquet append-only paths)
DECISIONS_TBL = "decisions_history.silver"                 # pair_key, source, decided_at
CUM_MAPPING_TBL = "clean_proposals_accumulated.silver"     # old_master_id, canonical_master_id, updated_at
REVIEW_QUEUE_TBL = "review_queue.silver"                   # append-only log of queue events/updates

# -----------------
# Utilities / UDFs
# -----------------

def _hash12(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]

def read_or_empty(tbl_name: str, schema: T.StructType | None = None):
    try:
        return spark.read.parquet(path(tbl_name))
    except Exception:
        return spark.createDataFrame([], schema) if schema else spark.createDataFrame([], T.StructType([]))

# Lightweight normalizer used upstream of embeddings
import re

def normalize_name_py(s: str) -> str:
    if s is None:
        return None
    s2 = s.lower().strip()
    s2 = re.sub(r"[^a-z0-9\s]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

normalize_name = F.udf(normalize_name_py, T.StringType())

# Cosine for local vectors

def cosine_py(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(n))) or 1.0
    nb = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(n))) or 1.0
    return float(dot / (na * nb))

cosine = F.udf(cosine_py, T.DoubleType())

# -----------------
# Azure OpenAI client (Managed Identity)
# -----------------
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class AOAI:
    def __init__(self):
        if not (AOAI_ENDPOINT and AOAI_CHAT_DEPLOYMENT and AOAI_EMBEDDING_DEPLOYMENT):
            raise RuntimeError("AOAI_* env vars missing. Set AOAI_ENDPOINT, AOAI_CHAT_DEPLOYMENT, AOAI_EMBEDDING_DEPLOYMENT.")
        scope = "https://cognitiveservices.azure.com/.default"
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), scope)
        self.client = AzureOpenAI(
            azure_endpoint=AOAI_ENDPOINT,
            api_version=AOAI_API_VERSION,
            azure_ad_token_provider=token_provider,
        )
        self.chat_deploy = AOAI_CHAT_DEPLOYMENT
        self.emb_deploy = AOAI_EMBEDDING_DEPLOYMENT

    # -------- Embeddings (batched on driver) --------
    def embed_all(self, texts: List[str], batch: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            resp = self.client.embeddings.create(model=self.emb_deploy, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return out

    # -------- LLM tool-call judging --------
    def judge_matches(self, focal: Dict[str, str], candidates: List[Dict[str, Any]], extra_context: str | None = None, prompt_version: str = "v1") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns (judgments_list, meta) where judgments_list has dicts with keys:
          candidate_master_id, score, llm_decision, llm_confidence, llm_reason, prompt_version
        and meta includes: model_name, decided_at, aoai_request_id, prompt_version, context_used, context_hash, token_usage
        """
        tools = [{
            "type": "function",
            "function": {
                "name": "submit_match_judgments",
                "description": "Return a decision for each candidate whether it is the same real-world account as the focal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "reason": {"type": "string"},
                                    "candidate_master_id": {"type": "string"},
                                    "score": {"type": "number"},
                                    "decision": {"type": "string", "enum": ["YES","NO","NEEDS_CONFIRMATION"]},
                                    "confidence": {"type": "number"}
                                },
                                "required": ["reason","candidate_master_id","decision","score","confidence"]
                            }
                        }
                    },
                    "required": ["decisions"]
                }
            }
        }]

        system_msg = {
            "role": "system",
            "content": (
                "You are an account deduper. Decide if each candidate is the SAME real-world company as the focal. "
                "Normalize abbreviations and legal suffixes (LLC/L.L.C., Ltd/Limited, Intl/International, Log/Logistics, etc.). "
                "Subsidiaries/holdcos are NOT the same as parents. "
                "Return ONLY the function call with a decision for EACH candidate: YES, NO, or NEEDS_CONFIRMATION. "
                "Provide a short reason and a confidence 0.0–1.0."
            )
        }
        ctx_text = f"\nReviewer notes: {extra_context.strip()}" if (extra_context and extra_context.strip()) else ""
        user_msg = {
            "role": "user",
            "content": (
                f"Focal: {json.dumps(focal, ensure_ascii=False)}\n"
                f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
                f"{ctx_text}"
            )
        }
        resp = self.client.chat.completions.create(
            model=self.chat_deploy,
            messages=[system_msg, user_msg],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_match_judgments"}},
            temperature=0.1,
            top_p=0.1,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        args = json.loads(tool_calls[0].function.arguments or "{}") if tool_calls else {"decisions": []}
        decisions = args.get("decisions", [])

        cand_map = {c["id"]: c for c in candidates}
        out_list = []
        for d in decisions:
            cid = d.get("candidate_master_id")
            if not cid:
                continue
            sc = float(d.get("score", cand_map.get(cid, {}).get("score", 0.0)))
            dec = d.get("decision", "NEEDS_CONFIRMATION")
            conf = float(d.get("confidence", 0.6))
            rea = d.get("reason", "model_decision")
            out_list.append({
                "candidate_master_id": cid,
                "score": sc,
                "llm_decision": dec,
                "llm_confidence": conf,
                "llm_reason": rea,
                "prompt_version": prompt_version,
            })

        meta = {
            "model_name": AOAI_CHAT_DEPLOYMENT,
            "decided_at": F.current_timestamp(),  # filled later in Spark
            "aoai_request_id": getattr(resp, "id", None),
            "prompt_version": prompt_version,
            "context_used": extra_context or None,
            "context_hash": _hash12(extra_context or ""),
            "token_usage": getattr(resp, "usage", None).__dict__ if getattr(resp, "usage", None) else None,
        }
        return out_list, meta

_aoai = AOAI()

# -----------------
# 0) Read ALL tables (if present)
# -----------------
inputs = {
    "accounts_raw": read_or_empty("accounts_raw", T.StructType([
        T.StructField("account_id", T.StringType()),
        T.StructField("account_name", T.StringType()),
        T.StructField("normalized_name", T.StringType()),
        T.StructField("run_id", T.StringType()),
    ])),
    "accounts_full_match.silver": read_or_empty("accounts_full_match.silver"),
    "accounts_full_match_filter.silver": read_or_empty("accounts_full_match_filter.silver"),
    "similarity_pairs.silver": read_or_empty("similarity_pairs.silver"),
    "similarity_candidates.silver": read_or_empty("similarity_candidates.silver"),
    "llm_jobs.silver": read_or_empty("llm_jobs.silver"),
    "llm_results.silver": read_or_empty("llm_results.silver"),
    "review_queue.silver": read_or_empty(REVIEW_QUEUE_TBL),
    "apply_proposals.silver": read_or_empty("apply_proposals.silver"),
    "clean_proposals.silver": read_or_empty("clean_proposals.silver"),
    "decisions_history.silver": read_or_empty(DECISIONS_TBL),
    "clean_proposals_accumulated.silver": read_or_empty(CUM_MAPPING_TBL),
    "accounts_full_match.silver.postapply": read_or_empty("accounts_full_match.silver.postapply"),
    "accounts_full_match_filter.gold": read_or_empty("accounts_full_match_filter.gold"),
    "entities_gold": read_or_empty("entities_gold"),
    "entity_members": read_or_empty("entity_members"),
}

# -----------------
# 1) Ingest accounts_raw (append-only)
# -----------------
accounts_raw = inputs["accounts_raw"]
if "normalized_name" not in accounts_raw.columns:
    accounts_raw = accounts_raw.withColumn("normalized_name", normalize_name(F.col("account_name")))
if "run_id" not in accounts_raw.columns:
    accounts_raw = accounts_raw.withColumn("run_id", F.lit(RUN_ID))

accounts_raw.write.mode("append").partitionBy("run_id").format("parquet").save(path("accounts_raw"))

# -----------------
# 2) Perfect match base (append-only)
# -----------------
masters = accounts_raw.groupBy("normalized_name").agg(F.min("account_id").alias("master_account_id"))

full = (accounts_raw.join(masters, "normalized_name")
    .select("account_id","account_name","master_account_id","normalized_name","run_id"))

grp = full.groupBy("master_account_id").agg(F.count("*").alias("group_size"))
full = (full.join(grp, "master_account_id")
    .withColumn("is_master", F.expr("account_id = master_account_id"))
    .withColumn("is_dupe", F.expr("group_size > 1"))
    .withColumn("stage_rule", F.when(F.col("is_dupe"), F.lit("PERFECT")).otherwise(F.lit(None))))

# -----------------
# 3) Apply cumulative mapping BEFORE computing pairs (CDC, append-only)
# -----------------
cum_map = inputs["clean_proposals_accumulated.silver"].select("old_master_id","canonical_master_id").dropDuplicates()

full_mapped = (full.alias("f").join(
        cum_map.alias("m"),
        F.col("f.master_account_id") == F.col("m.old_master_id"),
        "left"
    )
    .withColumn("master_account_id", F.coalesce(F.col("m.canonical_master_id"), F.col("f.master_account_id")))
    .select("account_id","account_name","master_account_id","normalized_name","run_id"))

grp2 = full_mapped.groupBy("master_account_id").agg(F.count("*").alias("group_size"))
full_mapped = (full_mapped.join(grp2, "master_account_id")
    .withColumn("is_master", F.expr("account_id = master_account_id"))
    .withColumn("is_dupe", F.expr("group_size > 1"))
    .withColumn("stage_rule", F.when(F.col("is_dupe"), F.lit("PERFECT")).otherwise(F.lit(None))))

(full_mapped.withColumn("run_id", F.lit(RUN_ID))
    .write.mode("append").partitionBy("run_id").format("parquet")
    .save(path("accounts_full_match.silver")))

# -----------------
# 4) Masters slice (append-only)
# -----------------
masters_only = (full_mapped.filter(F.col("account_id") == F.col("master_account_id"))
    .select("account_id","account_name","master_account_id","is_master","group_size")
    .withColumn("is_dupe", F.expr("group_size > 1"))
    .withColumn("run_id", F.lit(RUN_ID)))

masters_only.write.mode("append").partitionBy("run_id").format("parquet").save(path("accounts_full_match_filter.silver"))

# -----------------
# 5) Embeddings + pairs among masters (A<B); anti-join already decided pairs
# -----------------
# Compute normalized_name (if not present from earlier) and embed using AOAI embeddings on driver in batches.
masters_for_emb = (masters_only
    .withColumn("normalized_name", normalize_name(F.col("account_name")))
    .select("account_id","normalized_name").dropDuplicates())

pdf = masters_for_emb.toPandas()
text_list = pdf["normalized_name"].fillna("").tolist()
emb_list = _aoai.embed_all(text_list, batch=64)

# Join embeddings back
pdf_out = pdf.copy()
pdf_out["embedding"] = emb_list

emb_schema = T.StructType([
    T.StructField("account_id", T.StringType()),
    T.StructField("normalized_name", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
])
masters_emb = spark.createDataFrame(pdf_out, schema=emb_schema)

# Pairwise similarities (A<B)
a = masters_emb.select(
    F.col("account_id").alias("master_a_id"),
    F.col("normalized_name").alias("name_a"),
    F.col("embedding").alias("emb_a")
)
b = masters_emb.select(
    F.col("account_id").alias("master_b_id"),
    F.col("normalized_name").alias("name_b"),
    F.col("embedding").alias("emb_b")
)

pairs = (a.crossJoin(b)
    .filter(F.col("master_a_id") < F.col("master_b_id"))
    .withColumn("score", cosine(F.col("emb_a"), F.col("emb_b")))
    .withColumn("pair_key", F.concat_ws("|", F.col("master_a_id"), F.col("master_b_id")))
    .select("pair_key","master_a_id","master_b_id","score"))

# Remove already-decided pairs this cycle/history
if len(inputs["decisions_history.silver"].columns):
    decided_pairs = inputs["decisions_history.silver"].select("pair_key").dropDuplicates()
else:
    decided_pairs = spark.createDataFrame([], "pair_key string")

pairs = pairs.join(decided_pairs, "pair_key", "left_anti").withColumn("run_id", F.lit(RUN_ID))

pairs.write.mode("append").partitionBy("run_id").format("parquet").save(path("similarity_pairs.silver"))

# -----------------
# 6) Route pairs → similarity_candidates.silver (append-only)
# -----------------
candidates = (pairs
    .withColumn("route",
        F.when(F.col("score") >= F.lit(T_AUTO), F.lit("AUTO_YES_95"))
         .when(F.col("score") >= F.lit(T_LLM_LOW), F.lit("LLM"))
         .otherwise(F.lit("AUTO_NO"))
    )
    .withColumn("proposed_keep_master", F.when(F.col("score") >= F.lit(T_AUTO), F.col("master_a_id")))
    .withColumn("proposed_merge_master", F.when(F.col("score") >= F.lit(T_AUTO), F.col("master_b_id")))
    .withColumn("policy", F.lit("first_wins"))
    .withColumn("threshold_version", F.lit("v1"))
    .withColumn("tagged_at", F.current_timestamp())
    .withColumn("run_id", F.lit(RUN_ID))
)
candidates.write.mode("append").partitionBy("run_id").format("parquet").save(path("similarity_candidates.silver"))

# -----------------
# 7) LLM jobs (first pass) — append
#     Stepped batching via LLM_BATCH_SIZE
# -----------------
win = Window.partitionBy("master_a_id").orderBy(F.col("score").desc())
llm_band = (candidates.filter(F.col("route") == F.lit("LLM"))
    .withColumn("rank", F.row_number().over(win))
    .filter(F.col("rank") <= F.lit(LLM_TOP_N)))

# Compute batch_index per focal using integer division on rank-1
llm_band = llm_band.withColumn(
    "batch_index",
    F.floor((F.col("rank") - F.lit(1)) / F.lit(LLM_BATCH_SIZE)).cast("int")
)

# Join names for prompts
cand_with_names = (llm_band
    .join(masters_only.select(F.col("account_id").alias("master_a_id"), F.col("account_name").alias("focal_name")), "master_a_id")
    .join(masters_only.select(F.col("account_id").alias("master_b_id"), F.col("account_name").alias("candidate_name")), "master_b_id"))

llm_jobs = (cand_with_names.groupBy("master_a_id","focal_name","batch_index")
    .agg(F.collect_list(F.struct(
        F.col("master_b_id").alias("candidate_master_id"),
        F.col("candidate_name"),
        F.col("score"),
        F.col("rank")
    )).alias("candidates"))
    .withColumnRenamed("master_a_id","focal_master_id")
    .withColumn("model_name", F.lit(AOAI_CHAT_DEPLOYMENT))
    .withColumn("prompt_version", F.lit("v1"))
    .withColumn("created_at", F.current_timestamp())
    .withColumn("run_id", F.lit(RUN_ID))
)
llm_jobs.write.mode("append").partitionBy("run_id").format("parquet").save(path("llm_jobs.silver"))

# -----------------
# 7a) Execute AOAI for each job (driver), write results (append)
# -----------------
jobs_pdf = llm_jobs.select("focal_master_id","focal_name","batch_index","candidates","prompt_version","model_name").toPandas()

results_rows = []
for _, row in jobs_pdf.iterrows():
    focal_id = str(row["focal_master_id"]) ; focal_name = row["focal_name"]
    candidates_payload = [{"id": str(c["candidate_master_id"]), "name": c["candidate_name"], "score": float(c["score"])} for c in row["candidates"]]
    judgments, meta = _aoai.judge_matches(
        focal={"id": focal_id, "name": focal_name},
        candidates=candidates_payload,
        extra_context=None,
        prompt_version=row["prompt_version"]
    )
    results_rows.append({
        "focal_master_id": focal_id,
        "results": judgments,
        "prompt_version": meta["prompt_version"],
        "model_name": meta["model_name"],
        "aoai_request_id": meta.get("aoai_request_id"),
        "token_usage_json": json.dumps(meta.get("token_usage")) if meta.get("token_usage") else None,
        "context_used": None,
        "context_hash": _hash12("") ,
        "decided_at": None,  # set in Spark below
        "run_id": RUN_ID,
    })

res_schema = T.StructType([
    T.StructField("focal_master_id", T.StringType()),
    T.StructField("results", T.ArrayType(T.MapType(T.StringType(), T.StringType()))),  # store as array<map> for append simplicity
    T.StructField("prompt_version", T.StringType()),
    T.StructField("model_name", T.StringType()),
    T.StructField("aoai_request_id", T.StringType()),
    T.StructField("token_usage_json", T.StringType()),
    T.StructField("context_used", T.StringType()),
    T.StructField("context_hash", T.StringType()),
    T.StructField("decided_at", T.TimestampType()),
    T.StructField("run_id", T.StringType()),
])

llm_results_df = spark.createDataFrame(results_rows, schema=res_schema) \
    .withColumn("decided_at", F.current_timestamp())

# Convert array<map> back to array<struct> with expected fields to keep downstream stable
# (candidate_master_id, score, llm_decision, llm_confidence, llm_reason, prompt_version)

def _map_to_struct(m):
    return {
        "candidate_master_id": m.get("candidate_master_id"),
        "score": F.coalesce(F.col("m.score").cast("double"), F.lit(0.0)),
        "llm_decision": m.get("llm_decision"),
        "llm_confidence": F.col("m.llm_confidence").cast("double"),
        "llm_reason": m.get("llm_reason"),
        "prompt_version": m.get("prompt_version"),
    }

# Rebuild results array as array<struct>
res_exploded = (llm_results_df.select(
    "focal_master_id","prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at","run_id",
    F.explode_outer("results").alias("m"))
)
res_struct = res_exploded.select(
    "focal_master_id","prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at","run_id",
    F.col("m")["candidate_master_id"].alias("candidate_master_id"),
    F.col("m")["score"].cast("double").alias("score"),
    F.col("m")["llm_decision"].alias("llm_decision"),
    F.col("m")["llm_confidence"].cast("double").alias("llm_confidence"),
    F.col("m")["llm_reason"].alias("llm_reason"),
    F.col("m")["prompt_version"].alias("result_prompt_version")
)

# Group back to the original array<struct>
llm_results_final = (res_struct.groupBy("focal_master_id","prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at","run_id")
    .agg(F.collect_list(F.struct(
        F.col("candidate_master_id").alias("candidate_master_id"),
        F.col("score"),
        F.col("llm_decision"),
        F.col("llm_confidence"),
        F.col("llm_reason")
    )).alias("results"))
)

llm_results_final.write.mode("append").partitionBy("run_id").format("parquet").save(path("llm_results.silver"))

# -----------------
# 7b) OPTIONAL: Rerun LLM for queued pairs with reviewer notes
#       - Supports per-pair notes (RERUN_NOTES_JSON), per-focal notes (FOCAL_NOTES_JSON), and GLOBAL_NOTES
#       - Scope: ALL | FOCAL | COMPONENT graph
# -----------------
existing_q = inputs["review_queue.silver"]

if RERUN_ALL or RERUN_PAIR_KEYS or RERUN_FOCALS or RERUN_SCOPE in ("focal","component"):
    rq_sel = existing_q
    if len(rq_sel.columns):
        rq_sel = rq_sel.filter(F.col("status") == F.lit("QUEUED"))
        # Narrow by scope
        # Build seed node set from focals and/or pair_keys
        seed_nodes = set(RERUN_FOCALS)
        for pk in (RERUN_PAIR_KEYS or []):
            try:
                a, b = pk.split("|", 1)
                seed_nodes.add(a); seed_nodes.add(b)
            except ValueError:
                pass

        if RERUN_SCOPE == "focal" and seed_nodes:
            rq_sel = rq_sel.where(F.col("focal_master_id").isin(list(seed_nodes)))
        elif RERUN_SCOPE == "component" and seed_nodes:
            # Compute connected component on driver
            edges = rq_sel.select("focal_master_id","candidate_master_id").distinct().collect()
            adj = {}
            for r in edges:
                a, b = r["focal_master_id"], r["candidate_master_id"]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            seen = set([n for n in seed_nodes if (n in adj or n)])
            stack = list(seen)
            while stack:
                v = stack.pop()
                for nb in adj.get(v, ()):  # BFS
                    if nb not in seen:
                        seen.add(nb); stack.append(nb)
            rq_sel = rq_sel.where(F.col("focal_master_id").isin(list(seen)))
        # else: default ALL focals in queue
        if RERUN_PAIR_KEYS:
            rq_sel = rq_sel.where(F.col("pair_key").isin(RERUN_PAIR_KEYS))
    else:
        rq_sel = spark.createDataFrame([], "pair_key string, focal_master_id string, candidate_master_id string, score double, notes string, enqueued_at timestamp, updated_at timestamp, status string")

    # Join names for LLM prompt
    with_names = (rq_sel
        .join(masters_only.select(F.col("account_id").alias("focal_master_id"), F.col("account_name").alias("focal_name")), "focal_master_id")
        .join(masters_only.select(F.col("account_id").alias("candidate_master_id"), F.col("account_name").alias("candidate_name")), "candidate_master_id"))

    # Attach optional per-pair notes overrides
    if RERUN_NOTES:
        notes_items = [(k, v) for k, v in RERUN_NOTES.items()]
        notes_df = spark.createDataFrame(notes_items, schema="pair_key string, notes_override string")
        with_names = with_names.join(notes_df, "pair_key", "left")

    # Attach optional per-focal notes
    if FOCAL_NOTES:
        focal_items = [(k, v) for k, v in FOCAL_NOTES.items()]
        focal_df = spark.createDataFrame(focal_items, schema="focal_master_id string, focal_notes string")
        with_names = with_names.join(focal_df, "focal_master_id", "left")

    # Effective notes precedence: pair override > existing 'notes' column > focal-wide notes > GLOBAL_NOTES
    with_names = with_names.withColumn(
        "notes_eff",
        F.coalesce(F.col("notes_override"), F.col("notes"), F.col("focal_notes"), F.lit(GLOBAL_NOTES))
    )

    # Build focal → candidates map on driver
    rerun_pdf = (with_names
                 .select("focal_master_id","focal_name","candidate_master_id","candidate_name","score","notes_eff")
                 .toPandas())

    from collections import defaultdict
    grp_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"focal_name": None, "notes": None, "candidates": []})
    for _, r in rerun_pdf.iterrows():
        fid = str(r["focal_master_id"]) ; grp_map[fid]["focal_name"] = r["focal_name"]
        grp_map[fid]["notes"] = r["notes_eff"]
        grp_map[fid]["candidates"].append({
            "id": str(r["candidate_master_id"]),
            "name": r["candidate_name"],
            "score": float(r["score"]),
        })

    rr_rows = []
    for fid, payload in grp_map.items():
        judgments, meta = _aoai.judge_matches(
            focal={"id": fid, "name": payload["focal_name"]},
            candidates=payload["candidates"],
            extra_context=(payload["notes"] or ""),
            prompt_version="v2_with_context"
        )
        for j in judgments:
            rr_rows.append({
                "focal_master_id": fid,
                "candidate_master_id": j["candidate_master_id"],
                "score": j["score"],
                "llm_decision": j["llm_decision"],
                "llm_confidence": j["llm_confidence"],
                "llm_reason": j["llm_reason"],
                "prompt_version": meta["prompt_version"],
                "model_name": meta["model_name"],
                "aoai_request_id": meta.get("aoai_request_id"),
                "token_usage_json": json.dumps(meta.get("token_usage")) if meta.get("token_usage") else None,
                "context_used": meta.get("context_used"),
                "context_hash": meta.get("context_hash"),
                "decided_at": None,  # set below
                "run_id": RUN_ID,
            })

    rr_schema = T.StructType([
        T.StructField("focal_master_id", T.StringType()),
        T.StructField("candidate_master_id", T.StringType()),
        T.StructField("score", T.DoubleType()),
        T.StructField("llm_decision", T.StringType()),
        T.StructField("llm_confidence", T.DoubleType()),
        T.StructField("llm_reason", T.StringType()),
        T.StructField("prompt_version", T.StringType()),
        T.StructField("model_name", T.StringType()),
        T.StructField("aoai_request_id", T.StringType()),
        T.StructField("token_usage_json", T.StringType()),
        T.StructField("context_used", T.StringType()),
        T.StructField("context_hash", T.StringType()),
        T.StructField("decided_at", T.TimestampType()),
        T.StructField("run_id", T.StringType()),
    ])

    rr_df = spark.createDataFrame(rr_rows, schema=rr_schema).withColumn("decided_at", F.current_timestamp())

    # Append rerun results (as array-of-struct per focal) to llm_results.silver
    llm_results_rerun = (rr_df.groupBy("focal_master_id","prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at","run_id")
        .agg(F.collect_list(F.struct(
            F.col("candidate_master_id").alias("candidate_master_id"),
            F.col("score"),
            F.col("llm_decision"),
            F.col("llm_confidence"),
            F.col("llm_reason")
        )).alias("results")))
    llm_results_rerun.write.mode("append").partitionBy("run_id").format("parquet").save(path("llm_results.silver"))

# -----------------
# 8) Review queue APPEND-ONLY (carry forward; includes first pass + rerun updates)
# -----------------
# explode all results for *this RUN_ID* only (first pass + rerun)
all_llm_results = spark.read.parquet(path("llm_results.silver")).filter(F.col("run_id") == F.lit(RUN_ID))

results_exploded = (all_llm_results
    .select("focal_master_id","prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at","run_id", F.explode_outer("results").alias("r"))
    .select(
        "focal_master_id","prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at","run_id",
        F.col("r.candidate_master_id").alias("candidate_master_id"),
        F.col("r.score").alias("score"),
        F.col("r.llm_decision").alias("llm_decision"),
        F.col("r.llm_confidence").alias("llm_confidence"),
        F.col("r.llm_reason").alias("llm_reason"),
    )
    .withColumn("pair_key", F.concat_ws("|", F.least("focal_master_id","candidate_master_id"), F.greatest("focal_master_id","candidate_master_id")))
)

# Queue entries to append (NEEDS or borderline)
queue_increment = (results_exploded.filter(
        (F.col("llm_decision") == F.lit("NEEDS_CONFIRMATION")) |
        ((F.col("llm_decision") == F.lit("YES")) & (F.col("llm_confidence") < F.lit(0.70))) |
        ((F.col("llm_decision") == F.lit("NO"))  & (F.col("llm_confidence") < F.lit(0.60)) & (F.col("score") >= F.lit(T_LLM_LOW) - F.lit(0.05)))
    )
    .select("pair_key","focal_master_id","candidate_master_id","score","llm_confidence","llm_reason",
            "prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at")
    .withColumn("queue_reason", F.lit("NEEDS_CONFIRMATION"))
    .withColumn("status", F.lit("QUEUED"))
    .withColumn("reviewer", F.lit(None).cast("string"))
    .withColumn("notes", F.lit(None).cast("string"))
    .withColumn("route_at_enqueue", F.lit("LLM"))
    .withColumn("enqueued_at", F.current_timestamp())
    .withColumn("updated_at", F.lit(None).cast("timestamp"))
    .withColumn("run_id", F.lit(RUN_ID))
)

# Append new queue entries
if queue_increment.count() > 0:
    queue_increment.write.mode("append").partitionBy("run_id").format("parquet").save(path(REVIEW_QUEUE_TBL))

# If rerun happened this run, create queue update records (APPEND, not overwrite)
if RERUN_ALL or RERUN_PAIR_KEYS or RERUN_FOCALS or RERUN_SCOPE in ("focal","component"):
    # Derive updates from this run's results_exploded joined to existing queue (to carry reviewer/notes/enqueued_at)
    base_updates = (results_exploded
        .select("pair_key","focal_master_id","candidate_master_id","score","llm_confidence","llm_reason",
                "prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at")
        .join(existing_q.select("pair_key","reviewer","notes","enqueued_at"), "pair_key", "left"))

    queue_updates = (base_updates
        .withColumn("queue_reason", F.lit("NEEDS_CONFIRMATION"))
        .withColumn("status", F.when((F.lit(AUTO_APPROVE_RERUN)) & (F.col("llm_decision") == F.lit("YES")) & (F.col("llm_confidence") >= F.lit(AUTO_APPROVE_RERUN_YES_CONF)), F.lit("APPROVED")).otherwise(F.lit("QUEUED")))
        .withColumn("decision", F.when(F.col("status") == F.lit("APPROVED"), F.lit("HUMAN_APPROVED_AFTER_NOTES")).otherwise(F.lit(None).cast("string")))
        .withColumn("route_at_enqueue", F.lit("LLM"))
        .withColumn("updated_at", F.current_timestamp())
        .withColumn("run_id", F.lit(RUN_ID)))

    if queue_updates.count() > 0:
        queue_updates.write.mode("append").partitionBy("run_id").format("parquet").save(path(REVIEW_QUEUE_TBL))

print(f"[Notebook 1] Completed RUN_ID={RUN_ID}. Wrote/updated tables under {BASE_DIR}. LLM_TOP_N={LLM_TOP_N}, LLM_BATCH_SIZE={LLM_BATCH_SIZE}. RERUN_SCOPE={RERUN_SCOPE}, seeds_focals={RERUN_FOCALS}, selected_pair_keys={RERUN_PAIR_KEYS}")
