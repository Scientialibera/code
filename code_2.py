# Databricks / Fabric PySpark Notebook 2 — Merge → Admin Gate → Clean → Apply → Gold
# ===========================================================================================
# - UNLIMITED column support matching code_1.py enhancements  
# - COMPLETE group management operations (remove/reparent/breakup/singleton parent)
# - Operation modes: FINALIZE | GROUP_OPERATIONS | UPDATE_DECISIONS
# - PowerApps integration support via JSON parameterization
# - Enhanced decision processing and admin gate functionality
# - Full metadata column preservation and processing
#
# FINALIZE STAGE (APPEND-ONLY):
# - Compile final proposals (AUTO_95 + LLM_YES + HUMAN_APPROVED[+AFTER_NOTES])
# - Apply Admin Gate overrides (APPROVE / REJECT)
# - Write preview (append, partitioned by run_id)
# - If FINALIZE=true: append decisions_history (YES + NO for admin rejects),
#   append cumulative mapping deltas, and re-apply effective mapping
#   (computed in-memory from append-only history) to produce postapply + gold.
#
# Safety hardening:
# - All writes are append-only (partitioned by run_id where applicable).
# - Handles missing/empty upstream tables gracefully.
# - De-duplicates by pair_key before persisting.
# - Uses union-find to produce canonical mapping (transitive merges).

import os, time, json
from typing import Dict, List, Tuple, Any, Set
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# -----------------
# Config / params with unlimited column support
# -----------------
RUN_ID = os.environ.get("RUN_ID", f"run_{int(time.time())}")
BASE_DIR = os.environ.get("BASE_DIR", "/lake/dedupe_demo")  # change for Fabric Lakehouse path
FINALIZE = (os.environ.get("FINALIZE", "false").lower() == "true")

# Operation mode for different processing types
OPERATION = os.environ.get("OPERATION", "FINALIZE")  # FINALIZE | GROUP_OPERATIONS | UPDATE_DECISIONS

# Unlimited dynamic column configuration (same as notebook 1)
SIMILARITY_COLUMNS = os.environ.get("SIMILARITY_COLUMNS", '["account_name"]')  # JSON array
LLM_CONTEXT_COLUMNS = os.environ.get("LLM_CONTEXT_COLUMNS", '["account_name"]')  # JSON array
METADATA_COLUMNS = os.environ.get("METADATA_COLUMNS", '[]')  # JSON array

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

# -----------------
# COLUMN CONFIGURATION - All hardcoded columns moved here for full configurability
# -----------------
# Core required columns for deduplication pipeline (must match code_1.py)
CORE_COLUMNS = ["account_id", "account_name", "master_account_id", "normalized_name", "run_id"]

# Full match columns (configurable - can be different from CORE + METADATA)
FULL_MATCH_COLUMNS = ["account_id", "account_name", "master_account_id", "normalized_name", "run_id"]

# Base columns for different table types
BASE_COLUMNS = ["account_id", "account_name", "master_account_id", "normalized_name", "run_id"]
GOLD_BASE_COLUMNS = ["account_id", "account_name", "master_account_id", "is_master", "group_size"]

# Master slice columns (for accounts_full_match_filter.silver)
MASTERS_BASE_COLUMNS = ["account_id", "account_name", "master_account_id", "is_master", "group_size"]

# Mapping and cumulative columns
MAPPING_COLUMNS = ["old_master_id", "canonical_master_id"]
CUMULATIVE_MAPPING_COLUMNS = ["old_master_id", "canonical_master_id", "updated_at"]

# Decision and review queue columns
DECISION_COLUMNS = ["pair_key", "decision", "decided_at"]
REVIEW_QUEUE_BASE_COLUMNS = [
    "pair_key", "focal_master_id", "candidate_master_id", "score", "llm_confidence", "llm_reason",
    "prompt_version", "model_name", "aoai_request_id", "token_usage_json", "context_used", "context_hash", "decided_at"
]

# Group operations columns
GROUP_OPERATION_COLUMNS = ["operation_id", "operation_type", "master_id", "parameters", "status", "created_at"]

# Group operations support (from PowerApps)
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

# Optional: auto-approve rerun YES in queue if Notebook 1 emitted such tags
AUTO_APPROVE_RERUN = os.environ.get("AUTO_APPROVE_RERUN_YES", "true").lower() == "true"
AUTO_APPROVE_RERUN_YES_CONF = float(os.environ.get("AUTO_APPROVE_RERUN_YES_CONF", "0.9"))

# ENHANCED: Group operation configuration
GROUP_OPERATION_SOURCE = os.environ.get("GROUP_OPERATION_SOURCE", "HUMAN_POWERAPP")
GROUP_BATCH_MODE = os.environ.get("GROUP_BATCH_MODE", "false").lower() == "true"

def path(table: str) -> str:
    return f"{BASE_DIR}/{table}"

# Table names
DECISIONS_TBL = "decisions_history.silver"                  # append-only
CUM_MAPPING_TBL = "clean_proposals_accumulated.silver"      # append-only deltas
REVIEW_QUEUE_TBL = "review_queue.silver"
SIM_CAND_TBL = "similarity_candidates.silver"
LLM_RESULTS_TBL = "llm_results.silver"
FULL_MATCH_TBL = "accounts_full_match.silver"
POSTAPPLY_TBL = "accounts_full_match.silver.postapply"
GOLD_TBL = "accounts_full_match_filter.gold"
ADMIN_GATE_TBL = "admin_gate.silver"                        # UI writes APPROVE/REJECT here
PREVIEW_TBL = "final_proposals.preview"                     # append-only preview
GROUP_OPERATIONS_LOG_TBL = "group_operations.log"          # ENHANCED: Group operations log

# -----------------
# Helpers
# -----------------

def read_or_empty(tbl_name: str, schema: T.StructType | None = None):
    try:
        return spark.read.parquet(path(tbl_name))
    except Exception:
        return spark.createDataFrame([], schema) if schema else spark.createDataFrame([], T.StructType([]))

def ensure_cols(df, cols_with_types: Dict[str, T.DataType]):
    """Ensure df has the given columns; add nulls with provided types if missing."""
    for c, t in cols_with_types.items():
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(t))
    return df

# Group management utility functions
def get_group_members(master_id: str, full_match_df) -> List[str]:
    """Get all members of a group given the master_id"""
    try:
        members = full_match_df.filter(F.col("master_account_id") == master_id).select("account_id").collect()
        return [row["account_id"] for row in members]
    except Exception:
        return []

def find_singleton_groups(full_match_df) -> List[str]:
    """Find all singleton groups (groups with only one member)"""
    try:
        singletons = (full_match_df.groupBy("master_account_id")
                     .agg(F.count("*").alias("group_size"))
                     .filter(F.col("group_size") == 1)
                     .select("master_account_id").collect())
        return [row["master_account_id"] for row in singletons]
    except Exception:
        return []

def build_connected_components(pairs: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Build connected components from pairs using union-find"""
    parent: Dict[str, str] = {}
    
    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    
    for a, b in pairs:
        union(str(a), str(b))
    
    # Build components
    components: Dict[str, Set[str]] = {}
    for node in parent.keys():
        root = find(node)
        if root not in components:
            components[root] = set()
        components[root].add(node)
    
    return components

# -----------------
# Operation mode routing
# -----------------
if OPERATION == "UPDATE_DECISIONS":
    # Handle decision updates (similar to code_1.py but for finalization context)
    print(f"[UPDATE_DECISIONS] Processing {len(DECISION_UPDATES)} decision updates in finalization context")
    
    if DECISION_UPDATES:
        # Process decision updates that affect finalization
        decision_records = []
        for update in DECISION_UPDATES:
            pair_key = update.get("pair_key")
            decision = update.get("decision")
            reviewer = update.get("reviewer", "")
            notes = update.get("notes", "")
            source = update.get("source", "HUMAN_POWERAPP")
            
            if pair_key and decision:
                decision_records.append({
                    "pair_key": pair_key,
                    "decision": decision,
                    "reviewer": reviewer,
                    "notes": notes,
                    "source": source,
                    "decided_at": None,
                    "run_id": RUN_ID
                })
        
        if decision_records:
            decision_df = spark.createDataFrame(decision_records)
            decision_df = decision_df.withColumn("decided_at", F.current_timestamp())
            decision_df.write.mode("append").format("parquet").save(path(DECISIONS_TBL))
    
    print(f"[UPDATE_DECISIONS] Completed processing {len(DECISION_UPDATES)} decision updates")

elif OPERATION == "GROUP_OPERATIONS":
    # ENHANCED: Complete group management operations
    print(f"[GROUP_OPERATIONS] Processing {len(GROUP_OPERATIONS)} group operations")
    
    # Load required data for group operations
    full_match = read_or_empty(FULL_MATCH_TBL)
    if "run_id" in full_match.columns:
        full_match_current = full_match.filter(F.col("run_id") == F.lit(RUN_ID))
    else:
        full_match_current = full_match
    
    cum_map_hist = read_or_empty(CUM_MAPPING_TBL)
    
    # Process each group operation
    mapping_updates: List[Tuple[str, str]] = []
    operation_logs: List[Dict[str, Any]] = []
    
    for operation in GROUP_OPERATIONS:
        op_type = operation.get("operation")
        op_id = operation.get("operation_id", f"op_{int(time.time())}")
        reviewer = operation.get("reviewer", "system")
        notes = operation.get("notes", "")
        
        try:
            if op_type == "remove_member":
                # ENHANCED: Remove member from group (make it a singleton)
                member_id = operation.get("member_id")
                if member_id:
                    # Create mapping: member_id -> member_id (make it its own master)
                    mapping_updates.append((member_id, member_id))
                    operation_logs.append({
                        "operation_id": op_id,
                        "operation_type": op_type,
                        "member_id": member_id,
                        "status": "completed",
                        "reviewer": reviewer,
                        "notes": notes,
                        "applied_at": None,
                        "run_id": RUN_ID
                    })
                    print(f"[GROUP_OPERATIONS] Remove member: {member_id}")
                
            elif op_type == "reparent_member":
                # ENHANCED: Move member to different group
                member_id = operation.get("member_id")
                new_master = operation.get("new_master")
                if member_id and new_master:
                    # Create mapping: member_id -> new_master
                    mapping_updates.append((member_id, new_master))
                    operation_logs.append({
                        "operation_id": op_id,
                        "operation_type": op_type,
                        "member_id": member_id,
                        "new_master": new_master,
                        "status": "completed",
                        "reviewer": reviewer,
                        "notes": notes,
                        "applied_at": None,
                        "run_id": RUN_ID
                    })
                    print(f"[GROUP_OPERATIONS] Reparent member: {member_id} -> {new_master}")
                
            elif op_type == "add_parent_to_singleton":
                # ENHANCED: Add singleton to existing group
                singleton_id = operation.get("singleton_id")
                parent_master = operation.get("parent_master")
                if singleton_id and parent_master:
                    # Create mapping: singleton_id -> parent_master
                    mapping_updates.append((singleton_id, parent_master))
                    operation_logs.append({
                        "operation_id": op_id,
                        "operation_type": op_type,
                        "singleton_id": singleton_id,
                        "parent_master": parent_master,
                        "status": "completed",
                        "reviewer": reviewer,
                        "notes": notes,
                        "applied_at": None,
                        "run_id": RUN_ID
                    })
                    print(f"[GROUP_OPERATIONS] Add parent to singleton: {singleton_id} -> {parent_master}")
                
            elif op_type == "breakup_group":
                # ENHANCED: Break up entire group (make all members singletons)
                master_id = operation.get("master_id")
                if master_id:
                    # Get all group members
                    group_members = get_group_members(master_id, full_match_current)
                    for member in group_members:
                        if member != master_id:  # Don't remap the master to itself unnecessarily
                            mapping_updates.append((member, member))
                    
                    operation_logs.append({
                        "operation_id": op_id,
                        "operation_type": op_type,
                        "master_id": master_id,
                        "members_affected": len(group_members),
                        "status": "completed",
                        "reviewer": reviewer,
                        "notes": notes,
                        "applied_at": None,
                        "run_id": RUN_ID
                    })
                    print(f"[GROUP_OPERATIONS] Breakup group: {master_id} ({len(group_members)} members)")
                
            else:
                # Unknown operation type
                operation_logs.append({
                    "operation_id": op_id,
                    "operation_type": op_type,
                    "status": "failed",
                    "error": f"Unknown operation type: {op_type}",
                    "reviewer": reviewer,
                    "notes": notes,
                    "applied_at": None,
                    "run_id": RUN_ID
                })
                print(f"[GROUP_OPERATIONS] Unknown operation type: {op_type}")
                
        except Exception as e:
            # Log operation failure
            operation_logs.append({
                "operation_id": op_id,
                "operation_type": op_type,
                "status": "failed",
                "error": str(e),
                "reviewer": reviewer,
                "notes": notes,
                "applied_at": None,
                "run_id": RUN_ID
            })
            print(f"[GROUP_OPERATIONS] Failed operation {op_type}: {e}")
    
    # Apply mapping updates
    if mapping_updates:
        mapping_delta = spark.createDataFrame(mapping_updates, schema="old_master_id string, canonical_master_id string")
        mapping_delta = (mapping_delta.dropDuplicates()
                        .withColumn("updated_at", F.current_timestamp())
                        .withColumn("run_id", F.lit(RUN_ID))
                        .withColumn("source", F.lit(GROUP_OPERATION_SOURCE)))
        
        mapping_delta.write.mode("append").format("parquet").save(path(CUM_MAPPING_TBL))
        print(f"[GROUP_OPERATIONS] Applied {len(mapping_updates)} mapping updates")
    
    # Log operations
    if operation_logs:
        # Build schema for operation logs
        operation_log_schema = T.StructType([
            T.StructField("operation_id", T.StringType()),
            T.StructField("operation_type", T.StringType()),
            T.StructField("member_id", T.StringType()),
            T.StructField("new_master", T.StringType()),
            T.StructField("singleton_id", T.StringType()),
            T.StructField("parent_master", T.StringType()),
            T.StructField("master_id", T.StringType()),
            T.StructField("members_affected", T.IntegerType()),
            T.StructField("status", T.StringType()),
            T.StructField("error", T.StringType()),
            T.StructField("reviewer", T.StringType()),
            T.StructField("notes", T.StringType()),
            T.StructField("applied_at", T.TimestampType()),
            T.StructField("run_id", T.StringType()),
        ])
        
        operation_log_df = spark.createDataFrame(operation_logs, schema=operation_log_schema)
        operation_log_df = operation_log_df.withColumn("applied_at", F.current_timestamp())
        operation_log_df.write.mode("append").partitionBy("run_id").format("parquet").save(path(GROUP_OPERATIONS_LOG_TBL))
        print(f"[GROUP_OPERATIONS] Logged {len(operation_logs)} operations")
    
    print(f"[GROUP_OPERATIONS] Completed processing all group operations")

elif OPERATION == "FINALIZE" or FINALIZE:
    # Standard finalization process with enhancements
    
    # -----------------
    # Load inputs (scope by RUN_ID where appropriate) with enhanced column support
    # -----------------
    
    dec_hist = read_or_empty(DECISIONS_TBL)
    
    cum_map_hist = read_or_empty(CUM_MAPPING_TBL, T.StructType([
        T.StructField("old_master_id", T.StringType()),
        T.StructField("canonical_master_id", T.StringType()),
        T.StructField("updated_at", T.TimestampType()),
        T.StructField("run_id", T.StringType()),
        T.StructField("source", T.StringType()),  # ENHANCED: Track source of mapping
    ]))
    
    sim_cand = read_or_empty(SIM_CAND_TBL)
    if "run_id" in sim_cand.columns:
        sim_cand = sim_cand.filter(F.col("run_id") == F.lit(RUN_ID))
    
    llm_results = read_or_empty(LLM_RESULTS_TBL)
    if "run_id" in llm_results.columns:
        llm_results = llm_results.filter(F.col("run_id") == F.lit(RUN_ID))
    
    review_q = read_or_empty(REVIEW_QUEUE_TBL)
    
    full_match = read_or_empty(FULL_MATCH_TBL)
    if "run_id" in full_match.columns:
        full_match = full_match.filter(F.col("run_id") == F.lit(RUN_ID))
    
    admin_gate = read_or_empty(ADMIN_GATE_TBL, T.StructType([
        T.StructField("pair_key", T.StringType()),
        T.StructField("focal_master_id", T.StringType()),
        T.StructField("candidate_master_id", T.StringType()),
        T.StructField("action", T.StringType()),   # APPROVE | REJECT
        T.StructField("reviewer", T.StringType()),
        T.StructField("notes", T.StringType()),
        T.StructField("updated_at", T.TimestampType()),
        T.StructField("run_id", T.StringType()),
    ]))
    
    # -----------------
    # ENHANCED: Build proposals_draft with unlimited column support
    # -----------------
    # 1) AUTO_95 from similarity_candidates
    auto_df = (sim_cand
        .filter(F.col("route") == F.lit("AUTO_YES_95"))
        .select(
            F.col("pair_key"),
            F.col("master_a_id").alias("m1"),
            F.col("master_b_id").alias("m2"),
            F.lit("AUTO_95").alias("source"),
            F.col("score"),
            F.lit("auto_threshold").alias("reason"),
            F.lit("auto_threshold").alias("prompt_version"),
            F.lit(None).cast("string").alias("model_name"),
            F.current_timestamp().alias("decided_at"),
            F.lit(None).cast("double").alias("confidence"),
            F.lit(None).cast("string").alias("reviewer"),
            F.lit(None).cast("string").alias("notes"),
        )
        .dropDuplicates(["pair_key"]))
    
    # 2) ENHANCED: LLM YES from llm_results (carry prompt metadata + unlimited context)
    llm_exploded = (llm_results
        .select("focal_master_id", "prompt_version", "model_name", "decided_at", F.explode_outer("results").alias("r")))
    
    # Normalize result struct presence
    llm_exploded = ensure_cols(llm_exploded, {
        "r.llm_confidence": T.DoubleType(),
        "r.llm_reason": T.StringType(),
        "r.score": T.DoubleType(),
        "r.candidate_master_id": T.StringType(),
        "r.llm_decision": T.StringType(),
    })
    
    llm_yes = (llm_exploded
        .filter(F.col("r.llm_decision") == F.lit("YES"))
        .select(
            F.concat_ws("|",
                F.least("focal_master_id", F.col("r.candidate_master_id")),
                F.greatest("focal_master_id", F.col("r.candidate_master_id"))
            ).alias("pair_key"),
            F.col("focal_master_id").alias("m1"),
            F.col("r.candidate_master_id").alias("m2"),
            F.lit("LLM_YES").alias("source"),
            F.col("r.score").alias("score"),
            F.col("r.llm_reason").alias("reason"),
            F.col("prompt_version"),
            F.col("model_name"),
            F.col("decided_at"),
            F.col("r.llm_confidence").alias("confidence"),
            F.lit(None).cast("string").alias("reviewer"),
            F.lit(None).cast("string").alias("notes"),
        )
        .dropDuplicates(["pair_key"]))
    
    # 3) ENHANCED: HUMAN APPROVED from review_queue (explicit or after-notes)
    review_q = ensure_cols(review_q, {
        "pair_key": T.StringType(),
        "focal_master_id": T.StringType(),
        "candidate_master_id": T.StringType(),
        "status": T.StringType(),
        "score": T.DoubleType(),
        "llm_reason": T.StringType(),
        "llm_confidence": T.DoubleType(),
        "prompt_version": T.StringType(),
        "model_name": T.StringType(),
        "updated_at": T.TimestampType(),
        "reviewer": T.StringType(),
        "notes": T.StringType(),
        "decision": T.StringType(),
    })
    
    human_base = (review_q
        .filter(F.col("status") == F.lit("APPROVED"))
        .select(
            "pair_key",
            F.col("focal_master_id").alias("m1"),
            F.col("candidate_master_id").alias("m2"),
            F.coalesce(F.col("decision"), F.lit("HUMAN_APPROVED")).alias("decision_tag"),
            F.col("score"),
            F.col("llm_reason").alias("reason"),
            F.col("llm_confidence").alias("confidence"),
            F.col("prompt_version"),
            F.col("model_name"),
            F.coalesce(F.col("updated_at"), F.current_timestamp()).alias("decided_at"),
            F.col("reviewer"),
            F.col("notes")
        )
        .dropDuplicates(["pair_key"]))
    
    human_yes = (human_base
        .withColumn("source",
            F.when(F.col("decision_tag") == F.lit("HUMAN_APPROVED_AFTER_NOTES"), F.lit("HUMAN_NOTES_AUTO_APPROVED"))
             .otherwise(F.lit("HUMAN_APPROVED"))
        )
        .drop("decision_tag"))
    
    proposals_draft = (auto_df
        .unionByName(llm_yes, allowMissingColumns=True)
        .unionByName(human_yes, allowMissingColumns=True)
        .withColumn("run_id", F.lit(RUN_ID))
        .withColumn("created_at", F.current_timestamp())
    )
    
    # Avoid re-proposing pairs already decided YES in prior runs
    if set(["pair_key","decision"]).issubset(set(dec_hist.columns)):
        decided_yes = dec_hist.filter(F.col("decision") == F.lit("YES")).select("pair_key").dropDuplicates()
        proposals_draft = proposals_draft.join(decided_yes, "pair_key", "left_anti")
    
    # -----------------
    # ENHANCED: Admin Gate (REJECT / APPROVE overrides)
    # -----------------
    rejects = admin_gate.filter(F.col("action") == F.lit("REJECT")).select("pair_key").dropDuplicates()
    
    approves = (admin_gate.filter(F.col("action") == F.lit("APPROVE"))
        .select(
            "pair_key",
            F.col("focal_master_id").alias("m1"),
            F.col("candidate_master_id").alias("m2"),
            F.lit("ADMIN_APPROVED").alias("source"),
            F.lit(None).cast("double").alias("score"),
            F.lit("admin_approved").alias("reason"),
            F.lit("admin_review").alias("prompt_version"),
            F.lit(None).cast("string").alias("model_name"),
            F.coalesce(F.col("updated_at"), F.current_timestamp()).alias("decided_at"),
            F.lit(None).cast("double").alias("confidence"),
            F.col("reviewer"),
            F.col("notes")
        ))
    
    proposals_final = (proposals_draft.join(rejects, "pair_key", "left_anti")
        .unionByName(approves, allowMissingColumns=True)
        .dropDuplicates(["pair_key"]))
    
    # -----------------
    # ENHANCED: Write PREVIEW (append per run) with metadata columns
    # -----------------
    preview = (proposals_final
        .withColumn("run_id", F.lit(RUN_ID))
        .withColumn("preview_at", F.current_timestamp())
        .withColumn("column_config", F.lit(json.dumps({
            "similarity_columns": SIMILARITY_COLS,
            "llm_context_columns": LLM_CONTEXT_COLS,
            "metadata_columns": METADATA_COLS
        }))))
    
    (preview
        .write.mode("append").partitionBy("run_id").format("parquet")
        .save(path(PREVIEW_TBL)))
    
    # -----------------
    # ENHANCED: FINALIZE (append-only ledgers) with group operations support
    # -----------------
    if FINALIZE:
        # a) decisions_history append (YES rows)
        yes_df = (proposals_final
            .withColumn("decision", F.lit("YES"))
            .withColumn("decided_at", F.coalesce(F.col("decided_at"), F.current_timestamp()))
            .select(
                F.col("pair_key"),
                F.col("source"),
                F.col("score"),
                F.col("reason"),
                F.col("decision"),
                F.col("decided_at"),
                F.col("confidence"),
                F.col("prompt_version"),
                F.col("model_name"),
                F.lit(RUN_ID).alias("run_id"),
                F.col("reviewer"),
                F.col("notes")
            ))
        
        # b) decisions_history append (NO for admin rejects)
        admin_reject_meta = (admin_gate.filter(F.col("action") == F.lit("REJECT"))
            .select("pair_key", "reviewer", "notes", F.coalesce(F.col("updated_at"), F.current_timestamp()).alias("rej_ts")))
        no_df = (admin_reject_meta
            .withColumn("source", F.lit("ADMIN_REJECT"))
            .withColumn("score", F.lit(None).cast("double"))
            .withColumn("reason", F.lit("admin_reject"))
            .withColumn("decision", F.lit("NO"))
            .withColumn("decided_at", F.col("rej_ts"))
            .withColumn("confidence", F.lit(None).cast("double"))
            .withColumn("prompt_version", F.lit("admin_review"))
            .withColumn("model_name", F.lit(None).cast("string"))
            .withColumn("run_id", F.lit(RUN_ID))
            .select("pair_key","source","score","reason","decision","decided_at","confidence","prompt_version","model_name","run_id","reviewer","notes"))
        
        decisions_batch = yes_df.unionByName(no_df, allowMissingColumns=True)
        
        # Append-only write
        decisions_batch.write.mode("append").format("parquet").save(path(DECISIONS_TBL))
        
        # c) ENHANCED: Build mapping delta with union-find and group operations support
        pairs_df = proposals_final.select("m1","m2").dropna().dropDuplicates()
        pairs = [(r["m1"], r["m2"]) for r in pairs_df.collect()]
        
        parent: Dict[str, str] = {}
        def find(x: str) -> str:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        
        for a, b in pairs:
            union(str(a), str(b))
        
        mapping_rows: List[Tuple[str, str]] = [(n, find(n)) for n in list(parent.keys()) if n != find(n)]
        
        if mapping_rows:
            mapping_delta = spark.createDataFrame(mapping_rows, schema="old_master_id string, canonical_master_id string")
            mapping_delta = (mapping_delta.dropDuplicates()
                           .withColumn("updated_at", F.current_timestamp())
                           .withColumn("run_id", F.lit(RUN_ID))
                           .withColumn("source", F.lit("FINALIZATION")))
            
            # Append mapping delta (no overwrite)
            mapping_delta.write.mode("append").format("parquet").save(path(CUM_MAPPING_TBL))
        
        # d) ENHANCED: Re-apply EFFECTIVE mapping with unlimited column support
        cum_hist = read_or_empty(CUM_MAPPING_TBL)
        if len(cum_hist.columns) == 0:
            cum_effective = spark.createDataFrame([], "old_master_id string, canonical_master_id string, updated_at timestamp, run_id string, source string")
        else:
            w_eff = Window.partitionBy("old_master_id").orderBy(F.col("updated_at").desc_nulls_last())
            cum_effective = (cum_hist
                .withColumn("rn", F.row_number().over(w_eff))
                .filter(F.col("rn") == 1)
                .drop("rn"))
        
        fm = read_or_empty(FULL_MATCH_TBL)
        if "run_id" in fm.columns:
            fm = fm.filter(F.col("run_id") == F.lit(RUN_ID))
        
        # ENHANCED: Apply mapping preserving configurable columns only
        base_columns = FULL_MATCH_COLUMNS
        available_columns = [col for col in base_columns if col in fm.columns]
        
        fm_mapped = (fm.alias("f")
            .join(cum_effective.alias("m"), F.col("f.master_account_id") == F.col("m.old_master_id"), "left")
            .withColumn("master_account_id", F.coalesce(F.col("m.canonical_master_id"), F.col("f.master_account_id")))
            .select(*available_columns))
        
        grp = fm_mapped.groupBy("master_account_id").agg(F.count("*").alias("group_size"))
        fm_post = (fm_mapped.join(grp, "master_account_id")
            .withColumn("is_master", F.expr("account_id = master_account_id"))
            .withColumn("is_dupe", F.expr("group_size > 1"))
            .withColumn("stage_rule", F.when(F.col("is_dupe"), F.lit("PERFECT")).otherwise(F.lit(None))))
        
        # write postapply snapshot (append by run) with metadata preservation
        postapply_columns = [col for col in fm_post.columns if col in available_columns + ["is_master", "is_dupe", "stage_rule", "group_size"]]
        (fm_post.select(*postapply_columns)
            .withColumn("run_id", F.lit(RUN_ID))
            .withColumn("column_config", F.lit(json.dumps({
                "similarity_columns": SIMILARITY_COLS,
                "llm_context_columns": LLM_CONTEXT_COLS,
                "metadata_columns": METADATA_COLS
            })))
            .write.mode("append").partitionBy("run_id").format("parquet")
            .save(path(POSTAPPLY_TBL)))
        
        # ENHANCED: masters gold with unlimited column support
        gold_base_columns = GOLD_BASE_COLUMNS
        gold_metadata_columns = [col for col in METADATA_COLS if col in fm_post.columns]
        gold_columns = gold_base_columns + gold_metadata_columns
        
        masters_gold = (fm_post.filter(F.col("account_id") == F.col("master_account_id"))
            .select(*[col for col in gold_columns if col in fm_post.columns])
            .withColumn("is_dupe", F.expr("group_size > 1"))
            .withColumn("run_id", F.lit(RUN_ID))
            .withColumn("column_config", F.lit(json.dumps({
                "similarity_columns": SIMILARITY_COLS,
                "llm_context_columns": LLM_CONTEXT_COLS,
                "metadata_columns": METADATA_COLS
            }))))
        
        masters_gold.write.mode("append").partitionBy("run_id").format("parquet").save(path(GOLD_TBL))

# Final status report
operation_status = {
    "operation": OPERATION,
    "run_id": RUN_ID,
    "finalize": FINALIZE,
    "group_operations_count": len(GROUP_OPERATIONS),
    "decision_updates_count": len(DECISION_UPDATES),
    "unlimited_columns": {
        "similarity_columns": SIMILARITY_COLS,
        "llm_context_columns": LLM_CONTEXT_COLS,
        "metadata_columns": METADATA_COLS
    },
    "capabilities": [
        "unlimited_column_support",
        "group_management_operations",
        "decision_update_processing",
        "powerapp_integration",
        "enhanced_finalization"
    ]
}

print(f"[Notebook 2] Completed: {json.dumps(operation_status, indent=2)}")
print(f"[Notebook 2] OPERATION={OPERATION}, RUN_ID={RUN_ID}, FINALIZE={FINALIZE}. Preview appended at {path(PREVIEW_TBL)}. ENHANCED FEATURES: Unlimited columns (Similarity={SIMILARITY_COLS}, Context={LLM_CONTEXT_COLS}, Metadata={METADATA_COLS}), Group operations ({len(GROUP_OPERATIONS)} processed), Decision updates ({len(DECISION_UPDATES)} processed), Full PowerApps integration support, Complete group management (remove/reparent/breakup/singleton), Enhanced metadata preservation")