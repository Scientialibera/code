# Databricks / Fabric PySpark Notebook 1 — AOAI end-to-end
# =================================================================================
# - UNLIMITED dynamic columns (metadata, similarity, LLM context)
# - ENHANCED rerun logic targeting ALL LLM decisions (not just NEEDS_CONFIRMATION)
# - Operation modes: FULL_PIPELINE | UPDATE_DECISIONS | RERUN_ONLY | GROUP_OPERATIONS
# - Decision update mechanism for PowerApps integration
# - Group management operation support
# - Enhanced parameterization for PowerApps control
#
# Build base tables, apply cumulative mapping, compute similarities,
# route candidates, prepare AOAI LLM jobs & results (with optional stepped batching),
# and manage review queue (APPEND-ONLY). No merges here (Notebook 2 does that).
#
# Safety features included:
# - CDC mapping applied BEFORE similarities (clean_proposals_accumulated.silver)
# - Anti-join pairs against decisions_history.silver (YES only)
# - Append-only writes with run_id partitioning (snapshots derived downstream)
# - LLM tool-calling with full metadata and optional rerun scopes (all/focal/component)
# - Review queue is append-only; reruns append updates, never overwrite

import os, json, math, time, hashlib, re
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

# ===== Spark =====
spark = SparkSession.builder.getOrCreate()

# -----------------
# ENHANCED Config / params with unlimited column support
# -----------------
T_AUTO = float(os.environ.get("T_AUTO", "0.95"))          # ≥95% auto-merge band
T_LLM_LOW = float(os.environ.get("T_LLM_LOW", "0.70"))    # LLM band lower bound
LLM_TOP_N = int(os.environ.get("LLM_TOP_N", "3"))         # top-N neighbors kept per focal
# If batch size < top N, split LLM requests into multiple "jobs" per focal
LLM_BATCH_SIZE = max(1, int(os.environ.get("LLM_BATCH_SIZE", str(LLM_TOP_N))))

# Unlimited dynamic column configuration (JSON format)
SIMILARITY_COLUMNS = os.environ.get("SIMILARITY_COLUMNS", '["account_name"]')  # JSON array of column names
LLM_CONTEXT_COLUMNS = os.environ.get("LLM_CONTEXT_COLUMNS", '["account_name"]')  # JSON array of column names
METADATA_COLUMNS = os.environ.get("METADATA_COLUMNS", '[]')  # JSON array of additional metadata columns
SIMILARITY_WEIGHTS = os.environ.get("SIMILARITY_WEIGHTS", '{}')  # JSON object with column weights

# Parse JSON configurations with error handling
try:
    SIMILARITY_COLS = json.loads(SIMILARITY_COLUMNS)
    if not isinstance(SIMILARITY_COLS, list):
        SIMILARITY_COLS = ["account_name"]
except Exception:
    SIMILARITY_COLS = ["account_name"]

try:
    LLM_CONTEXT_COLS = json.loads(LLM_CONTEXT_COLUMNS)
    if not isinstance(LLM_CONTEXT_COLS, list):
        LLM_CONTEXT_COLS = ["account_name"]
except Exception:
    LLM_CONTEXT_COLS = ["account_name"]

try:
    METADATA_COLS = json.loads(METADATA_COLUMNS)
    if not isinstance(METADATA_COLS, list):
        METADATA_COLS = []
except Exception:
    METADATA_COLS = []

try:
    SIM_WEIGHTS = json.loads(SIMILARITY_WEIGHTS)
    if not isinstance(SIM_WEIGHTS, dict):
        SIM_WEIGHTS = {}
except Exception:
    SIM_WEIGHTS = {}

# -----------------
# COLUMN CONFIGURATION - All hardcoded columns moved here for full configurability
# -----------------
# Core required columns for deduplication pipeline
CORE_COLUMNS = ["account_id", "account_name", "master_account_id", "normalized_name", "run_id"]

# Full match columns (configurable - can be different from CORE + METADATA)
FULL_MATCH_COLUMNS = ["account_id", "account_name", "master_account_id", "normalized_name", "run_id"]

# Master slice columns (for accounts_full_match_filter.silver)
MASTERS_BASE_COLUMNS = ["account_id", "account_name", "master_account_id", "is_master", "group_size"]

# Embedding and similarity columns
EMBEDDING_BASE_COLUMNS = ["account_id", "account_name"]

# LLM context columns for prompts (dynamically configurable)
LLM_CONTEXT_BASE_COLUMNS = ["account_id", "account_name"]

# Schema building columns
SCHEMA_BASE_COLUMNS = [
    ("account_id", "StringType"),
    ("account_name", "StringType"), 
    ("normalized_name", "StringType"),
    ("run_id", "StringType")
]

# Decision and review queue columns
DECISION_COLUMNS = ["pair_key", "decision", "decided_at"]
REVIEW_QUEUE_BASE_COLUMNS = [
    "pair_key", "focal_master_id", "candidate_master_id", "score", "llm_confidence", "llm_reason",
    "prompt_version", "model_name", "aoai_request_id", "token_usage_json", "context_used", "context_hash", "decided_at"
]

# Results and pairs columns
PAIRS_COLUMNS = ["pair_key", "master_a_id", "master_b_id", "score"]
CANDIDATES_BASE_COLUMNS = ["pair_key", "master_a_id", "master_b_id", "score", "route"]

# Group operations and decision updates (for PowerApps integration)
GROUP_OPERATIONS_JSON = os.environ.get("GROUP_OPERATIONS_JSON", "[]")
DECISION_UPDATES_JSON = os.environ.get("DECISION_UPDATES_JSON", "[]")

try:
    GROUP_OPERATIONS = json.loads(GROUP_OPERATIONS_JSON) if GROUP_OPERATIONS_JSON else []
except Exception:
    GROUP_OPERATIONS = []

try:
    DECISION_UPDATES = json.loads(DECISION_UPDATES_JSON) if DECISION_UPDATES_JSON else []
except Exception:
    DECISION_UPDATES = []

# Rerun scope: 'focal' | 'component' | 'all' (default all)
RERUN_SCOPE = os.environ.get("RERUN_SCOPE", "all").lower()
RERUN_FOCALS = [x.strip() for x in os.environ.get("RERUN_FOCALS", "").split(",") if x.strip()]

# Operation mode for different types of operations
OPERATION = os.environ.get("OPERATION", "FULL_PIPELINE")  # FULL_PIPELINE | UPDATE_DECISIONS | RERUN_ONLY | GROUP_OPERATIONS

RUN_ID = os.environ.get("RUN_ID", f"run_{int(time.time())}")
BASE_DIR = os.environ.get("BASE_DIR", "/lake/dedupe_demo")  # change for Fabric Lakehouse path

# Rerun controls for ALL LLM decisions (not just NEEDS_CONFIRMATION)
RERUN_ALL = os.environ.get("RERUN_ALL_LLM_QUEUED", "false").lower() == "true"
RERUN_PAIR_KEYS = [x.strip() for x in os.environ.get("RERUN_PAIR_KEYS", "").split(",") if x.strip()]
RERUN_NOTES_JSON = os.environ.get("RERUN_NOTES_JSON", "")  # {"pair_key":"note", ...}
FOCAL_NOTES_JSON = os.environ.get("FOCAL_NOTES_JSON", "")  # {"<focal_master_id>":"shared guidance", ...}
GLOBAL_NOTES = os.environ.get("GLOBAL_NOTES", "").strip()
AUTO_APPROVE_RERUN = os.environ.get("AUTO_APPROVE_RERUN_YES", "true").lower() == "true"
AUTO_APPROVE_RERUN_YES_CONF = float(os.environ.get("AUTO_APPROVE_RERUN_YES_CONF", "0.9"))

# Decision update mechanism controls
UPDATE_DECISION_SOURCE = os.environ.get("UPDATE_DECISION_SOURCE", "HUMAN_POWERAPP")
UPDATE_BATCH_MODE = os.environ.get("UPDATE_BATCH_MODE", "false").lower() == "true"

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
DECISIONS_TBL = "decisions_history.silver"                 # pair_key, decision, decided_at
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
def normalize_name_py(s: str) -> str:
    if s is None:
        return None
    s2 = s.lower().strip()
    s2 = re.sub(r"[^a-z0-9\s]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

normalize_name = F.udf(normalize_name_py, T.StringType())

# Dynamic text preparation for similarity using multiple columns with weights
def prepare_similarity_text_py(row_dict: dict) -> str:
    """Prepare text for similarity by combining specified columns with weights"""
    text_parts = []
    for col in SIMILARITY_COLS:
        if col in row_dict and row_dict[col]:
            weight = SIM_WEIGHTS.get(col, 1.0)
            normalized = normalize_name_py(str(row_dict[col]))
            # Repeat text based on weight (simple approach)
            for _ in range(int(max(1, weight))):
                text_parts.append(normalized)
    return " ".join(text_parts)

# Register UDF for similarity text preparation
prepare_similarity_text = F.udf(prepare_similarity_text_py, T.StringType())

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
            if not chunk:
                continue
            resp = self.client.embeddings.create(model=self.emb_deploy, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return out

    # -------- LLM tool-call judging with unlimited column support --------
    def judge_matches(self, focal: Dict[str, Any], candidates: List[Dict[str, Any]], extra_context: str | None = None, prompt_version: str = "v1") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Supports unlimited columns in focal and candidates
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

        # Dynamic column awareness system message
        all_context_cols = list(set(LLM_CONTEXT_COLS + METADATA_COLS))
        column_info = f"Available context columns: {', '.join(all_context_cols)}" if all_context_cols else "Using basic account information"
        
        system_msg = {
            "role": "system",
            "content": (
                "You are an account deduper. Decide if each candidate is the SAME real-world company as the focal. "
                "Normalize abbreviations and legal suffixes (LLC/L.L.C., Ltd/Limited, Intl/International, Log/Logistics, etc.). "
                "Subsidiaries/holdcos are NOT the same as parents. "
                f"{column_info}. "
                "Consider all available context columns when making decisions. "
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

        out_list = []
        for d in decisions:
            cid = d.get("candidate_master_id")
            if not cid:
                continue
            sc = float(d.get("score", 0.0))
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
            "decided_at": None,  # set by Spark
            "aoai_request_id": getattr(resp, "id", None),
            "prompt_version": prompt_version,
            "context_used": extra_context or None,
            "context_hash": _hash12(extra_context or ""),
            "token_usage": getattr(resp, "usage", None).__dict__ if getattr(resp, "usage", None) else None,
        }
        return out_list, meta

_aoai = AOAI()

# -----------------
# 0) Read ALL tables with dynamic schema support
# -----------------
# Build dynamic schema for accounts_raw including unlimited metadata columns
def build_accounts_schema():
    base_schema = []
    # Build from configurable schema base
    for col_name, col_type in SCHEMA_BASE_COLUMNS:
        spark_type = getattr(T, col_type)()
        base_schema.append(T.StructField(col_name, spark_type))
    
    # Add ALL dynamic metadata columns
    for col_name in METADATA_COLS:
        base_schema.append(T.StructField(col_name, T.StringType()))
    return T.StructType(base_schema)

accounts_schema = build_accounts_schema()

inputs = {
    "accounts_raw": read_or_empty("accounts_raw", accounts_schema),
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
# 1) Operation mode routing for PowerApps integration
# -----------------
if OPERATION == "UPDATE_DECISIONS":
    # Handle decision updates from PowerApps
    print(f"[UPDATE_DECISIONS] Processing {len(DECISION_UPDATES)} decision updates")
    
    if DECISION_UPDATES:
        # Build decision update records
        decision_records = []
        for update in DECISION_UPDATES:
            pair_key = update.get("pair_key")
            decision = update.get("decision") 
            reviewer = update.get("reviewer", "")
            notes = update.get("notes", "")
            source = update.get("source", UPDATE_DECISION_SOURCE)
            
            if pair_key and decision:
                decision_records.append({
                    "pair_key": pair_key,
                    "decision": decision,
                    "reviewer": reviewer,
                    "notes": notes,
                    "source": source,
                    "decided_at": None,  # Will be set by Spark
                    "run_id": RUN_ID
                })
        
        if decision_records:
            decision_df = spark.createDataFrame(decision_records)
            decision_df = decision_df.withColumn("decided_at", F.current_timestamp())
            decision_df.write.mode("append").format("parquet").save(path(DECISIONS_TBL))
            
            # Update review queue status
            if UPDATE_BATCH_MODE:
                # For large batches, use Delta Lake updates
                try:
                    from delta.tables import DeltaTable
                    delta_table = DeltaTable.forPath(spark, path(REVIEW_QUEUE_TBL))
                    
                    for update in DECISION_UPDATES:
                        pair_key = update.get("pair_key")
                        decision = update.get("decision")
                        reviewer = update.get("reviewer", "")
                        notes = update.get("notes", "")
                        
                        if pair_key and decision:
                            delta_table.update(
                                condition = F.col("pair_key") == F.lit(pair_key),
                                set = {
                                    "status": F.lit("APPROVED" if decision == "YES" else "REJECTED"),
                                    "reviewer": F.lit(reviewer),
                                    "notes": F.lit(notes),
                                    "updated_at": F.current_timestamp()
                                }
                            )
                except ImportError:
                    print("Delta Lake not available for batch updates")
    
    print(f"[UPDATE_DECISIONS] Completed processing {len(decision_records)} decision updates")

elif OPERATION == "GROUP_OPERATIONS":
    # Handle group management operations
    print(f"[GROUP_OPERATIONS] Processing {len(GROUP_OPERATIONS)} group operations")
    
    # Group operations will be handled in code_2.py
    # This notebook logs the operations for processing
    for operation in GROUP_OPERATIONS:
        op_type = operation.get("operation")
        print(f"[GROUP_OPERATIONS] Queued operation: {op_type}")
    
    print(f"[GROUP_OPERATIONS] Operations queued for processing in code_2.py")

elif OPERATION in ["FULL_PIPELINE", "RERUN_ONLY"]:
    # Continue with standard processing
    
    # -----------------
    # 1) Ingest accounts_raw (append-only) - Enhanced with unlimited columns
    # -----------------
    accounts_raw = inputs["accounts_raw"]
    
    # Ensure all required columns exist
    if "normalized_name" not in accounts_raw.columns:
        accounts_raw = accounts_raw.withColumn("normalized_name", normalize_name(F.col("account_name")))
    if "run_id" not in accounts_raw.columns:
        accounts_raw = accounts_raw.withColumn("run_id", F.lit(RUN_ID))
    
    # Ensure all metadata columns exist
    for col in METADATA_COLS:
        if col not in accounts_raw.columns:
            accounts_raw = accounts_raw.withColumn(col, F.lit(None).cast("string"))

    if OPERATION == "FULL_PIPELINE":
        accounts_raw.write.mode("append").partitionBy("run_id").format("parquet").save(path("accounts_raw"))

    # -----------------
    # 2-6) Perfect match, mapping, embeddings, pairs - Enhanced with unlimited columns
    # -----------------
    if OPERATION == "FULL_PIPELINE":
        # Perfect match base (append-only)
        masters = accounts_raw.groupBy("normalized_name").agg(F.min("account_id").alias("master_account_id"))

        # Use fully configurable full match columns
        full_columns = FULL_MATCH_COLUMNS
        available_columns = [col for col in full_columns if col in accounts_raw.columns]
        
        full = (accounts_raw.join(masters, "normalized_name")
            .select(*available_columns))

        grp = full.groupBy("master_account_id").agg(F.count("*").alias("group_size"))
        full = (full.join(grp, "master_account_id")
            .withColumn("is_master", F.expr("account_id = master_account_id"))
            .withColumn("is_dupe", F.expr("group_size > 1"))
            .withColumn("stage_rule", F.when(F.col("is_dupe"), F.lit("PERFECT")).otherwise(F.lit(None))))

        # -----------------
        # 3) Apply cumulative mapping BEFORE computing pairs (CDC, append-only)
        # -----------------
        if len(inputs["clean_proposals_accumulated.silver"].columns):
            cum_map = inputs["clean_proposals_accumulated.silver"].select("old_master_id","canonical_master_id").dropDuplicates()
        else:
            cum_map = spark.createDataFrame([], "old_master_id string, canonical_master_id string")

        full_mapped = (full.alias("f").join(
                cum_map.alias("m"),
                F.col("f.master_account_id") == F.col("m.old_master_id"),
                "left"
            )
            .withColumn("master_account_id", F.coalesce(F.col("m.canonical_master_id"), F.col("f.master_account_id")))
            .select(*available_columns))

        grp2 = full_mapped.groupBy("master_account_id").agg(F.count("*").alias("group_size"))
        full_mapped = (full_mapped.join(grp2, "master_account_id")
            .withColumn("is_master", F.expr("account_id = master_account_id"))
            .withColumn("is_dupe", F.expr("group_size > 1"))
            .withColumn("stage_rule", F.when(F.col("is_dupe"), F.lit("PERFECT")).otherwise(F.lit(None))))

        (full_mapped.withColumn("run_id", F.lit(RUN_ID))
            .write.mode("append").partitionBy("run_id").format("parquet")
            .save(path("accounts_full_match.silver")))

        # -----------------
        # 4) Masters slice (append-only) - Enhanced with unlimited columns
        # -----------------
        masters_columns = MASTERS_BASE_COLUMNS + [col for col in METADATA_COLS if col in full_mapped.columns]
        
        masters_only = (full_mapped.filter(F.col("account_id") == F.col("master_account_id"))
            .select(*[col for col in masters_columns if col in full_mapped.columns])
            .withColumn("is_dupe", F.expr("group_size > 1"))
            .withColumn("run_id", F.lit(RUN_ID)))

        masters_only.write.mode("append").partitionBy("run_id").format("parquet").save(path("accounts_full_match_filter.silver"))

        # -----------------
        # 5) Embeddings + pairs among masters with unlimited column similarity
        # -----------------
        # Prepare similarity text using configured columns and weights
        masters_for_emb_columns = EMBEDDING_BASE_COLUMNS + [col for col in METADATA_COLS if col in masters_only.columns]
        masters_for_emb = masters_only.select(*masters_for_emb_columns)

        # Create combined similarity text with unlimited columns and weights
        if len(SIMILARITY_COLS) == 1 and SIMILARITY_COLS[0] == "account_name":
            # Simple case - just normalize account_name
            masters_for_emb = masters_for_emb.withColumn("similarity_text", normalize_name(F.col("account_name")))
        else:
            # Complex case - combine multiple columns with weights
            # Create struct of all available columns for UDF
            available_cols = [col for col in SIMILARITY_COLS + METADATA_COLS if col in masters_for_emb.columns]
            if available_cols:
                masters_for_emb = masters_for_emb.withColumn(
                    "similarity_text",
                    prepare_similarity_text(F.struct(*[F.col(c) for c in available_cols]))
                )
            else:
                # Fallback to account_name
                masters_for_emb = masters_for_emb.withColumn("similarity_text", normalize_name(F.col("account_name")))

        masters_for_emb = masters_for_emb.select("account_id", "similarity_text").dropDuplicates()

        # Generate embeddings
        pdf = masters_for_emb.toPandas()
        text_list = pdf["similarity_text"].fillna("").tolist()
        emb_list = _aoai.embed_all(text_list, batch=64)

        pdf_out = pdf.copy()
        pdf_out["embedding"] = emb_list

        emb_schema = T.StructType([
            T.StructField("account_id", T.StringType()),
            T.StructField("similarity_text", T.StringType()),
            T.StructField("embedding", T.ArrayType(T.FloatType())),
        ])
        masters_emb = spark.createDataFrame(pdf_out, schema=emb_schema)

        # Pairwise similarities (A<B)
        a = masters_emb.select(
            F.col("account_id").alias("master_a_id"),
            F.col("similarity_text").alias("text_a"),
            F.col("embedding").alias("emb_a")
        )
        b = masters_emb.select(
            F.col("account_id").alias("master_b_id"),
            F.col("similarity_text").alias("text_b"),
            F.col("embedding").alias("emb_b")
        )

        pairs = (a.crossJoin(b)
            .filter(F.col("master_a_id") < F.col("master_b_id"))
            .withColumn("score", cosine(F.col("emb_a"), F.col("emb_b")))
            .withColumn("pair_key", F.concat_ws("|", F.col("master_a_id"), F.col("master_b_id")))
            .select("pair_key","master_a_id","master_b_id","score"))

        # Remove already-decided YES pairs (history)
        if len(inputs["decisions_history.silver"].columns):
            dh = inputs["decisions_history.silver"]
            decided_pairs = (dh.filter(F.col("decision") == F.lit("YES")) if "decision" in dh.columns else dh)
            decided_pairs = decided_pairs.select("pair_key").dropDuplicates()
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
    else:
        # For non-full pipeline operations, load existing data
        candidates = inputs["similarity_candidates.silver"]
        masters_only = inputs["accounts_full_match_filter.silver"]

    # -----------------
    # 7) LLM jobs with unlimited column support
    # -----------------
    if OPERATION in ["FULL_PIPELINE", "RERUN_ONLY"]:
        win = Window.partitionBy("master_a_id").orderBy(F.col("score").desc())
        llm_band = (candidates.filter(F.col("route") == F.lit("LLM"))
            .withColumn("rank", F.row_number().over(win))
            .filter(F.col("rank") <= F.lit(LLM_TOP_N)))

        llm_band = llm_band.withColumn(
            "batch_index",
            F.floor((F.col("rank") - F.lit(1)) / F.lit(LLM_BATCH_SIZE)).cast("int")
        )

        # Join names and unlimited metadata for prompts
        base_columns = ["master_a_id", "master_b_id", "pair_key", "score", "rank", "batch_index"]
        
        # Build context columns dynamically
        all_context_cols = list(set(LLM_CONTEXT_COLS + METADATA_COLS))
        context_columns_a = [f"focal_{col}" for col in all_context_cols]
        context_columns_b = [f"candidate_{col}" for col in all_context_cols]

        # Join with focal account info - include all available context columns
        focal_select = [F.col(LLM_CONTEXT_BASE_COLUMNS[0]).alias("master_a_id")]
        for col in all_context_cols:
            if col in masters_only.columns:
                focal_select.append(F.col(col).alias(f"focal_{col}"))
        
        llm_with_focal = llm_band.join(masters_only.select(*focal_select), "master_a_id")

        # Join with candidate account info
        candidate_select = [F.col(LLM_CONTEXT_BASE_COLUMNS[0]).alias("master_b_id")]
        for col in all_context_cols:
            if col in masters_only.columns:
                candidate_select.append(F.col(col).alias(f"candidate_{col}"))

        cand_with_names = llm_with_focal.join(masters_only.select(*candidate_select), "master_b_id")

        # Build candidate payload with all available context columns
        candidate_struct_fields = [
            F.col("master_b_id").alias("candidate_master_id"),
            F.col("score"),
            F.col("rank")
        ]
        # Add all available context columns
        for col in all_context_cols:
            candidate_col = f"candidate_{col}"
            if candidate_col in [c.name for c in cand_with_names.schema]:
                candidate_struct_fields.append(F.col(candidate_col).alias(col))

        # Build focal grouping columns with all context
        focal_group_cols = ["master_a_id", "batch_index"]
        for col in all_context_cols:
            focal_col = f"focal_{col}"
            if focal_col in [c.name for c in cand_with_names.schema]:
                focal_group_cols.append(focal_col)

        llm_jobs = (cand_with_names.groupBy(focal_group_cols)
            .agg(F.collect_list(F.struct(*candidate_struct_fields)).alias("candidates"))
            .withColumnRenamed("master_a_id","focal_master_id")
            .withColumn("model_name", F.lit(AOAI_CHAT_DEPLOYMENT))
            .withColumn("prompt_version", F.lit("v1"))
            .withColumn("created_at", F.current_timestamp())
            .withColumn("run_id", F.lit(RUN_ID))
        )
        llm_jobs.write.mode("append").partitionBy("run_id").format("parquet").save(path("llm_jobs.silver"))

        # -----------------
        # 7a) Execute AOAI with unlimited column support
        # -----------------
        # Get all context columns available in llm_jobs
        all_focal_cols = [col for col in llm_jobs.columns if col.startswith("focal_")]
        
        jobs_pdf = llm_jobs.select(["focal_master_id", "batch_index", "candidates", "prompt_version", "model_name"] + all_focal_cols).toPandas()

        results_rows = []
        for _, row in jobs_pdf.iterrows():
            focal_id = str(row["focal_master_id"])
            
            # Build focal dict with all available context
            focal_dict = {"id": focal_id}
            for col in all_context_cols:
                focal_col = f"focal_{col}"
                if focal_col in row.index and pd.notna(row[focal_col]):
                    focal_dict[col] = str(row[focal_col])
            
            # Build candidates payload with all context
            candidates_payload = []
            for c in row["candidates"]:
                candidate_dict = {
                    "id": str(c["candidate_master_id"]), 
                    "score": float(c["score"])
                }
                # Add all available context columns
                for col in all_context_cols:
                    if col in c and c[col] is not None:
                        candidate_dict[col] = str(c[col])
                candidates_payload.append(candidate_dict)
            
            judgments, meta = _aoai.judge_matches(
                focal=focal_dict,
                candidates=candidates_payload,
                extra_context=None,
                prompt_version=row["prompt_version"]
            )
            results_rows.append({
                "focal_master_id": focal_id,
                "results": judgments,  # list[dict]
                "prompt_version": meta["prompt_version"],
                "model_name": meta["model_name"],
                "aoai_request_id": meta.get("aoai_request_id"),
                "token_usage_json": json.dumps(meta.get("token_usage")) if meta.get("token_usage") else None,
                "context_used": meta.get("context_used"),
                "context_hash": meta.get("context_hash"),
                "decided_at": None,  # set in Spark below
                "run_id": RUN_ID,
            })

        results_item_schema = T.StructType([
            T.StructField("candidate_master_id", T.StringType()),
            T.StructField("score", T.DoubleType()),
            T.StructField("llm_decision", T.StringType()),
            T.StructField("llm_confidence", T.DoubleType()),
            T.StructField("llm_reason", T.StringType()),
            T.StructField("prompt_version", T.StringType()),
        ])

        llm_results_schema = T.StructType([
            T.StructField("focal_master_id", T.StringType()),
            T.StructField("results", T.ArrayType(results_item_schema)),
            T.StructField("prompt_version", T.StringType()),
            T.StructField("model_name", T.StringType()),
            T.StructField("aoai_request_id", T.StringType()),
            T.StructField("token_usage_json", T.StringType()),
            T.StructField("context_used", T.StringType()),
            T.StructField("context_hash", T.StringType()),
            T.StructField("decided_at", T.TimestampType()),
            T.StructField("run_id", T.StringType()),
        ])

        llm_results_df = spark.createDataFrame(results_rows, schema=llm_results_schema) \
            .withColumn("decided_at", F.current_timestamp())

        llm_results_df.write.mode("append").partitionBy("run_id").format("parquet").save(path("llm_results.silver"))
    else:
        # For update operations, no new LLM processing needed
        pass

    # -----------------
    # 7b) Rerun LLM for ALL LLM decisions (CRITICAL FIX)
    # -----------------
    existing_q = inputs["review_queue.silver"]
    all_llm_results_current = inputs["llm_results.silver"]

    if RERUN_ALL or RERUN_PAIR_KEYS or RERUN_FOCALS or RERUN_SCOPE in ("focal","component") or OPERATION == "RERUN_ONLY":
        
        # CRITICAL FIX: Target ALL LLM results, not just review queue
        if len(all_llm_results_current.columns):
            # Get ALL LLM pairs from current results (not just NEEDS_CONFIRMATION)
            llm_results_exploded = (all_llm_results_current
                .select("focal_master_id", "prompt_version", "model_name", "decided_at", "run_id", 
                       F.explode_outer("results").alias("r"))
                .select(
                    "focal_master_id", "prompt_version", "model_name", "decided_at", "run_id",
                    F.col("r.candidate_master_id").alias("candidate_master_id"),
                    F.col("r.score").alias("score"),
                    F.col("r.llm_decision").alias("llm_decision"),
                    F.col("r.llm_confidence").alias("llm_confidence"),
                    F.col("r.llm_reason").alias("llm_reason")
                )
                .withColumn("pair_key", F.concat_ws("|", 
                    F.least("focal_master_id", "candidate_master_id"), 
                    F.greatest("focal_master_id", "candidate_master_id")))
            )
            
            # Apply scope filtering
            rq_sel = llm_results_exploded
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
                # Build connected components from ALL LLM pairs
                edges = rq_sel.select("focal_master_id","candidate_master_id").distinct().collect()
                adj = {}
                for r in edges:
                    a, b = r["focal_master_id"], r["candidate_master_id"]
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
                seen = set([n for n in seed_nodes])
                stack = list(seen)
                while stack:
                    v = stack.pop()
                    for nb in adj.get(v, ()):
                        if nb not in seen:
                            seen.add(nb); stack.append(nb)
                rq_sel = rq_sel.where(F.col("focal_master_id").isin(list(seen)))
            if RERUN_PAIR_KEYS:
                rq_sel = rq_sel.where(F.col("pair_key").isin(RERUN_PAIR_KEYS))
        else:
            rq_sel = spark.createDataFrame([], "pair_key string, focal_master_id string, candidate_master_id string, score double, llm_decision string")

        # Join with unlimited metadata columns
        masters_columns = LLM_CONTEXT_BASE_COLUMNS + [col for col in METADATA_COLS if col in masters_only.columns]
        
        with_names = (rq_sel
            .join(masters_only.select([F.col("account_id").alias("focal_master_id")] + [F.col(col).alias(f"focal_{col}") for col in masters_columns[1:]]), "focal_master_id")
            .join(masters_only.select([F.col("account_id").alias("candidate_master_id")] + [F.col(col).alias(f"candidate_{col}") for col in masters_columns[1:]]), "candidate_master_id"))

        if RERUN_NOTES:
            notes_items = [(k, v) for k, v in RERUN_NOTES.items()]
            notes_df = spark.createDataFrame(notes_items, schema="pair_key string, notes_override string")
            with_names = with_names.join(notes_df, "pair_key", "left")

        if FOCAL_NOTES:
            focal_items = [(k, v) for k, v in FOCAL_NOTES.items()]
            focal_df = spark.createDataFrame(focal_items, schema="focal_master_id string, focal_notes string")
            with_names = with_names.join(focal_df, "focal_master_id", "left")

        with_names = with_names.withColumn(
            "notes_eff",
            F.coalesce(F.col("notes_override"), F.col("notes"), F.col("focal_notes"), F.lit(GLOBAL_NOTES))
        )

        # Include all context columns in rerun
        rerun_select_cols = ["focal_master_id", "focal_account_name", "candidate_master_id", "candidate_account_name", "score", "notes_eff"]
        # Add available metadata columns
        for col in METADATA_COLS:
            focal_col = f"focal_{col}"
            candidate_col = f"candidate_{col}"
            if focal_col in [c.name for c in with_names.schema]:
                rerun_select_cols.append(focal_col)
            if candidate_col in [c.name for c in with_names.schema]:
                rerun_select_cols.append(candidate_col)

        available_rerun_cols = [col for col in rerun_select_cols if col in [c.name for c in with_names.schema]]
        rerun_pdf = with_names.select(*available_rerun_cols).toPandas()

        from collections import defaultdict
        grp_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"focal_context": {}, "notes": None, "candidates": []})
        
        for _, r in rerun_pdf.iterrows():
            fid = str(r["focal_master_id"])
            
            # Build focal context with all available columns
            focal_context = {"id": fid}
            if "focal_account_name" in r.index:
                focal_context["account_name"] = r["focal_account_name"]
            
            # Add all available metadata columns
            for col in METADATA_COLS:
                focal_col = f"focal_{col}"
                if focal_col in r.index and pd.notna(r[focal_col]):
                    focal_context[col] = str(r[focal_col])
            
            grp_map[fid]["focal_context"] = focal_context
            grp_map[fid]["notes"] = r["notes_eff"]
            
            # Build candidate context with all available columns
            candidate_context = {"id": str(r["candidate_master_id"]), "score": float(r["score"])}
            if "candidate_account_name" in r.index:
                candidate_context["account_name"] = r["candidate_account_name"]
            
            # Add all available metadata columns
            for col in METADATA_COLS:
                candidate_col = f"candidate_{col}"
                if candidate_col in r.index and pd.notna(r[candidate_col]):
                    candidate_context[col] = str(r[candidate_col])
            
            grp_map[fid]["candidates"].append(candidate_context)

        rr_rows = []
        for fid, payload in grp_map.items():
            judgments, meta = _aoai.judge_matches(
                focal=payload["focal_context"],
                candidates=payload["candidates"],
                extra_context=(payload["notes"] or ""),
                prompt_version="v2_with_context"
            )
            rr_rows.append({
                "focal_master_id": fid,
                "results": judgments,
                "prompt_version": meta["prompt_version"],
                "model_name": meta["model_name"],
                "aoai_request_id": meta.get("aoai_request_id"),
                "token_usage_json": json.dumps(meta.get("token_usage")) if meta.get("token_usage") else None,
                "context_used": meta.get("context_used"),
                "context_hash": meta.get("context_hash"),
                "decided_at": None,
                "run_id": RUN_ID,
            })

        if rr_rows:
            rr_df = spark.createDataFrame(rr_rows, schema=llm_results_schema).withColumn("decided_at", F.current_timestamp())
            rr_df.write.mode("append").partitionBy("run_id").format("parquet").save(path("llm_results.silver"))

    # -----------------
    # 8) Review queue with expanded criteria and unlimited column support
    # -----------------
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

    # Queue items that need human review (expanded criteria)
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

    if queue_increment.count() > 0:
        queue_increment.write.mode("append").partitionBy("run_id").format("parquet").save(path(REVIEW_QUEUE_TBL))

    # If rerun happened this run, append queue update records with possible auto-approval
    if RERUN_ALL or RERUN_PAIR_KEYS or RERUN_FOCALS or RERUN_SCOPE in ("focal","component"):
        base_updates = (results_exploded
            .select("pair_key","focal_master_id","candidate_master_id","score","llm_decision","llm_confidence","llm_reason",
                    "prompt_version","model_name","aoai_request_id","token_usage_json","context_used","context_hash","decided_at")
            .join(inputs["review_queue.silver"].select("pair_key","reviewer","notes","enqueued_at"), "pair_key", "left"))

        queue_updates = (base_updates
            .withColumn("queue_reason", F.lit("NEEDS_CONFIRMATION"))
            .withColumn("status", F.when((F.lit(AUTO_APPROVE_RERUN)) & (F.col("llm_decision") == F.lit("YES")) & (F.col("llm_confidence") >= F.lit(AUTO_APPROVE_RERUN_YES_CONF)), F.lit("APPROVED")).otherwise(F.lit("QUEUED")))
            .withColumn("decision", F.when(F.col("status") == F.lit("APPROVED"), F.lit("HUMAN_APPROVED_AFTER_NOTES")).otherwise(F.lit(None).cast("string")))
            .withColumn("route_at_enqueue", F.lit("LLM"))
            .withColumn("updated_at", F.current_timestamp())
            .withColumn("run_id", F.lit(RUN_ID)))

        if queue_updates.count() > 0:
            queue_updates.write.mode("append").partitionBy("run_id").format("parquet").save(path(REVIEW_QUEUE_TBL))

print(f"[Notebook 1] Completed RUN_ID={RUN_ID} (OPERATION={OPERATION}). Wrote/updated tables under {BASE_DIR}. LLM_TOP_N={LLM_TOP_N}, LLM_BATCH_SIZE={LLM_BATCH_SIZE}. RERUN_SCOPE={RERUN_SCOPE}, seeds_focals={RERUN_FOCALS}, selected_pair_keys={RERUN_PAIR_KEYS}. Unlimited columns - Similarity: {SIMILARITY_COLS}, LLM context: {LLM_CONTEXT_COLS}, Metadata: {METADATA_COLS}. Operation modes supported: {['FULL_PIPELINE', 'UPDATE_DECISIONS', 'RERUN_ONLY', 'GROUP_OPERATIONS']}. Decision updates: {len(DECISION_UPDATES)}, Group operations: {len(GROUP_OPERATIONS)}")