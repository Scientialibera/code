# Databricks / Fabric PySpark Notebook 2 — Merge → Admin Gate → Clean → Apply → Gold
# ==============================================================================
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

import os, time
from typing import Dict, List, Tuple
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# -----------------
# Config / params
# -----------------
RUN_ID = os.environ.get("RUN_ID", f"run_{int(time.time())}")
BASE_DIR = os.environ.get("BASE_DIR", "/lake/dedupe_demo")  # change for Fabric Lakehouse path
FINALIZE = (os.environ.get("FINALIZE", "false").lower() == "true")

# Optional: auto-approve rerun YES in queue if Notebook 1 emitted such tags
AUTO_APPROVE_RERUN = os.environ.get("AUTO_APPROVE_RERUN_YES", "true").lower() == "true"
AUTO_APPROVE_RERUN_YES_CONF = float(os.environ.get("AUTO_APPROVE_RERUN_YES_CONF", "0.9"))


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

# -----------------
# Load inputs (scope by RUN_ID where appropriate)
# -----------------

dec_hist = read_or_empty(DECISIONS_TBL)

cum_map_hist = read_or_empty(CUM_MAPPING_TBL, T.StructType([
    T.StructField("old_master_id", T.StringType()),
    T.StructField("canonical_master_id", T.StringType()),
    T.StructField("updated_at", T.TimestampType()),
    T.StructField("run_id", T.StringType()),
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
# Build proposals_draft
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

# 2) LLM YES from llm_results (carry prompt metadata)
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

# 3) HUMAN APPROVED from review_queue (explicit or after-notes)
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
# Admin Gate (REJECT / APPROVE overrides)
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
# Write PREVIEW (append per run)
# -----------------
preview = (proposals_final
    .withColumn("run_id", F.lit(RUN_ID))
    .withColumn("preview_at", F.current_timestamp()))

(preview
    .write.mode("append").partitionBy("run_id").format("parquet")
    .save(path(PREVIEW_TBL)))

# -----------------
# FINALIZE (append-only ledgers)
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

    # c) Build mapping delta with union-find on (m1,m2) and append to cumulative mapping
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
    mapping_delta = spark.createDataFrame(mapping_rows, schema="old_master_id string, canonical_master_id string")\
                         .dropDuplicates()\
                         .withColumn("updated_at", F.current_timestamp())\
                         .withColumn("run_id", F.lit(RUN_ID))

    # Append mapping delta (no overwrite)
    if mapping_delta.count() > 0:
        mapping_delta.write.mode("append").format("parquet").save(path(CUM_MAPPING_TBL))

    # d) Re-apply EFFECTIVE mapping (latest per old_master_id) to this run's full_match → POSTAPPLY + GOLD
    cum_hist = read_or_empty(CUM_MAPPING_TBL)
    if len(cum_hist.columns) == 0:
        cum_effective = spark.createDataFrame([], "old_master_id string, canonical_master_id string, updated_at timestamp, run_id string")
    else:
        w_eff = Window.partitionBy("old_master_id").orderBy(F.col("updated_at").desc_nulls_last())
        cum_effective = (cum_hist
            .withColumn("rn", F.row_number().over(w_eff))
            .filter(F.col("rn") == 1)
            .drop("rn"))

    fm = read_or_empty(FULL_MATCH_TBL)
    if "run_id" in fm.columns:
        fm = fm.filter(F.col("run_id") == F.lit(RUN_ID))

    fm_mapped = (fm.alias("f")
        .join(cum_effective.alias("m"), F.col("f.master_account_id") == F.col("m.old_master_id"), "left")
        .withColumn("master_account_id", F.coalesce(F.col("m.canonical_master_id"), F.col("f.master_account_id")))
        .select("account_id","account_name","master_account_id","normalized_name","run_id"))

    grp = fm_mapped.groupBy("master_account_id").agg(F.count("*").alias("group_size"))
    fm_post = (fm_mapped.join(grp, "master_account_id")
        .withColumn("is_master", F.expr("account_id = master_account_id"))
        .withColumn("is_dupe", F.expr("group_size > 1"))
        .withColumn("stage_rule", F.when(F.col("is_dupe"), F.lit("PERFECT")).otherwise(F.lit(None))))

    # write postapply snapshot (append by run)
    (fm_post.withColumn("run_id", F.lit(RUN_ID))
        .write.mode("append").partitionBy("run_id").format("parquet")
        .save(path(POSTAPPLY_TBL)))

    # masters gold
    masters_gold = (fm_post.filter(F.col("account_id") == F.col("master_account_id"))
        .select("account_id","account_name","master_account_id","is_master","group_size")
        .withColumn("is_dupe", F.expr("group_size > 1"))
        .withColumn("run_id", F.lit(RUN_ID)))

    masters_gold.write.mode("append").partitionBy("run_id").format("parquet").save(path(GOLD_TBL))

print(f"[Notebook 2] Completed RUN_ID={RUN_ID} (FINALIZE={FINALIZE}). Preview appended at {path(PREVIEW_TBL)}")
