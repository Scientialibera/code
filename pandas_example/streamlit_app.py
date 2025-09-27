#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit App for Account Deduplication Pipeline
Interactive UI for the Azure OpenAI-based account deduper

Usage:
  streamlit run app.py

Assumptions:
- Azure OpenAI environment variables are configured (AOAI_ENDPOINT, AOAI_CHAT_DEPLOYMENT, AOAI_EMBEDDING_DEPLOYMENT, etc.)
- The underlying pipeline code is available as `example_run.py` in the same working directory.
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Tuple

import pandas as pd
import streamlit as st

# Ensure .env is loaded so AOAI_* env vars are available to this process (Streamlit spawns workers)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Import the pipeline/library with AOAI + dedupe logic
import example_run as er


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Account Deduplication Pipeline",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Helpers
# =========================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_ids(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_seed_pairs(s: str) -> List[str]:
    if not s:
        return []
    parts = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        # allow either "a|b" or "a | b"
        if "|" not in p:
            # ignore malformed piece silently
            continue
        a, b = p.split("|", 1)
        a, b = a.strip(), b.strip()
        if a and b:
            parts.append(f"{a}|{b}")
    return parts


def init_session_state():
    """Initialize all session state variables."""
    if "accounts_df" not in st.session_state:
        st.session_state.accounts_df = pd.DataFrame(
            [
                ("a1", "ACME Corp"),
                ("a2", "ACME Corp"),
                ("a3", "Acme Corporation"),
                ("a4", "ACME Corp International"),
                ("a5", "ACME Corp Int"),
                ("g1", "ACME Global Holdings Public Ltd Company"),
                ("g2", "ACME Global Holdings Public Limited Company"),
                ("f1", "Globex I Logistics"),
                ("f2", "Globex International Logistics"),
                ("f3", "Globex Int Log"),
                ("b1", "Globex LLC"),
                ("b2", "Globex, L.L.C."),
                ("b3", "Globex I"),
                ("x1", "Unrelated Co"),
                ("x2", "Loblaws Ltd"),
                ("x3", "Loblows"),
            ],
            columns=["account_id", "account_name"],
        )

    if "cycle_count" not in st.session_state:
        st.session_state.cycle_count = 0

    if "pipeline_stage" not in st.session_state:
        st.session_state.pipeline_stage = "input"  # input → processing → review → final_review → complete

    # Intermediate artifacts
    for key, default in [
        ("full_base", None),
        ("masters", None),
        ("candidates", None),
        ("llm_results", None),
        ("review_queue", None),
        ("final_proposals", None),
        ("final_results", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if "id2name" not in st.session_state:
        st.session_state.id2name = {}

    if "human_decisions" not in st.session_state:
        # map pair_key -> dict
        st.session_state.human_decisions = {}

    if "context_book" not in st.session_state:
        st.session_state.context_book = er.ContextBook()

    # Config / controls
    if "scope" not in st.session_state:
        st.session_state.scope = "all"
    if "rerun_scope" not in st.session_state:
        st.session_state.rerun_scope = "all"
    if "admin_review" not in st.session_state:
        st.session_state.admin_review = False

    if "seed_focals" not in st.session_state:
        st.session_state.seed_focals = set()
    if "seed_pairs" not in st.session_state:
        st.session_state.seed_pairs = set()

    # Temp inputs for seeds
    if "seed_focals_input" not in st.session_state:
        st.session_state.seed_focals_input = ""
    if "seed_pairs_input" not in st.session_state:
        st.session_state.seed_pairs_input = ""


# =========================
# Sidebar
# =========================
def sidebar_info():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown(
            f"""
**Thresholds**
- Auto-merge ≥ **T_AUTO**: `{er.T_AUTO:.2f}`
- LLM band ≥ **T_LLM_LOW**: `{er.T_LLM_LOW:.2f}`
- Embedding weight: `{er.EMB_WEIGHT:.2f}`, Fuzzy weight: `{er.FUZZ_WEIGHT:.2f}`
- LLM Top-N per focal: `{er.LLM_TOP_N}`

**Rerun Auto-approve**
- Enabled: `{er.AUTO_APPROVE_RERUN}`
- YES Confidence ≥ `{er.AUTO_APPROVE_RERUN_YES_CONF:.2f}`
"""
        )
        st.markdown("---")
        st.markdown("### AOAI Deployments")
        st.caption(
            f"Endpoint: `{os.getenv('AOAI_ENDPOINT', 'not-set')}`\n\n"
            f"Chat: `{os.getenv('AOAI_CHAT_DEPLOYMENT', 'not-set')}`\n\n"
            f"Embedding: `{os.getenv('AOAI_EMBEDDING_DEPLOYMENT', 'not-set')}`"
        )
        st.markdown("---")
        st.markdown("### Ledgers & Artifacts")
        st.caption(
            f"- Decisions: `{er.DECISIONS_PATH}`\n"
            f"- Cumulative Mapping: `{er.CUM_MAPPING_PATH}`\n"
            f"- Notes: `{er.NOTES_PATH}`\n"
            f"- Demo Input (stateful): `{er.DEMO_ACCOUNTS_PATH}`"
        )


# =========================
# Stage: Input
# =========================
def display_accounts_input():
    st.header("📊 Input Accounts")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Edit Accounts")
        edited_df = st.data_editor(
            st.session_state.accounts_df,
            num_rows="dynamic",
            use_container_width=True,
            key="accounts_editor",
            column_config={
                "account_id": st.column_config.TextColumn(
                    "account_id", required=True
                ),
                "account_name": st.column_config.TextColumn(
                    "account_name", required=True
                ),
            },
        )
        if not edited_df.equals(st.session_state.accounts_df):
            st.session_state.accounts_df = edited_df

    with col2:
        st.subheader("Quick Actions")

        up = st.file_uploader("Upload CSV (account_id, account_name)", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up, dtype=str)
                if not {"account_id", "account_name"}.issubset(df.columns):
                    st.error("CSV must include columns: account_id, account_name")
                else:
                    st.session_state.accounts_df = df[["account_id", "account_name"]].astype(str)
                    st.success("Loaded CSV ✔")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")

        if st.button("Load Extended Demo", use_container_width=True):
            extended_demo = pd.DataFrame(
                [
                    ("a1", "ACME Corp"),
                    ("a2", "ACME Corp"),
                    ("a3", "Acme Corporation"),
                    ("a4", "ACME Corp International"),
                    ("a5", "ACME Corp Int"),
                    ("g1", "ACME Global Holdings Public Ltd Company"),
                    ("g2", "ACME Global Holdings Public Limited Company"),
                    ("f1", "Globex I Logistics"),
                    ("f2", "Globex International Logistics"),
                    ("f3", "Globex Int Log"),
                    ("b1", "Globex LLC"),
                    ("b2", "Globex, L.L.C."),
                    ("m1", "Microsoft Corporation"),
                    ("m2", "Microsoft Corp"),
                    ("m3", "Microsoft Inc"),
                    ("ap1", "Apple Inc"),
                    ("ap2", "Apple Incorporated"),
                    ("ap3", "Apple Computer Inc"),
                ],
                columns=["account_id", "account_name"],
            )
            st.session_state.accounts_df = extended_demo
            st.rerun()

        if st.button("Clear All", use_container_width=True):
            st.session_state.accounts_df = pd.DataFrame(
                columns=["account_id", "account_name"]
            )
            st.rerun()

    st.markdown("---")
    st.subheader("🔧 LLM Judging & Rerun Controls")

    c1, c2, c3 = st.columns(3)
    with c1:
        scope = st.selectbox(
            "Initial LLM Judging Scope",
            options=["all", "foci", "foci_related"],
            index=["all", "foci", "foci_related"].index(st.session_state.scope),
            help="Restrict which focals are judged initially by the LLM.",
        )
    with c2:
        rerun_scope = st.selectbox(
            "Rerun Scope (when you rerun with notes)",
            options=["all", "foci", "foci_related"],
            index=["all", "foci", "foci_related"].index(st.session_state.rerun_scope),
            help="Subset of queued pairs to rejudge after adding notes.",
        )
    with c3:
        admin_review = st.checkbox(
            "Require Admin Review",
            value=st.session_state.admin_review,
            help="Treat the final proposals table as the admin review gate.",
        )

    c4, c5 = st.columns(2)
    with c4:
        seed_focals_input = st.text_input(
            "Seed Focals (IDs, comma-separated)",
            value=st.session_state.seed_focals_input,
            placeholder="e.g., a1,a3",
            help="Used when scope is 'foci' or 'foci_related'.",
        )
    with c5:
        seed_pairs_input = st.text_input(
            "Seed Pairs (pair_keys a|b, comma-separated)",
            value=st.session_state.seed_pairs_input,
            placeholder="e.g., a1|a2,a3|a6",
            help="Used for graph expansion when scope is 'foci_related'.",
        )

    st.markdown("---")
    start_disabled = len(st.session_state.accounts_df) < 2
    if start_disabled:
        st.warning("Please include at least 2 accounts to start.")
    if st.button("🚀 Start Deduplication Process", type="primary", use_container_width=True, disabled=start_disabled):
        st.session_state.scope = scope
        st.session_state.rerun_scope = rerun_scope
        st.session_state.admin_review = admin_review
        st.session_state.seed_focals_input = seed_focals_input
        st.session_state.seed_pairs_input = seed_pairs_input
        st.session_state.seed_focals = set(parse_csv_ids(seed_focals_input))
        st.session_state.seed_pairs = set(parse_seed_pairs(seed_pairs_input))
        st.session_state.pipeline_stage = "processing"
        st.rerun()


# =========================
# Stage: Processing
# =========================
def run_initial_pipeline():
    try:
        with st.spinner("Running deduplication pipeline..."):
            # 1) Perfect match
            st.session_state.full_base = er.perfect_match_base(st.session_state.accounts_df)

            # 2) Apply historical mapping (cumulative)
            cum_map_prev = er.read_csv(er.CUM_MAPPING_PATH, cols=["old_master_id", "canonical_master_id"])
            full_after_hist = er.apply_mapping(st.session_state.full_base, cum_map_prev)

            # 3) Masters
            st.session_state.masters = er.masters_slice(full_after_hist)
            st.session_state.id2name = dict(
                zip(st.session_state.masters["account_id"].astype(str), st.session_state.masters["account_name"])
            )

            # 4) Similarity pairs
            pairs = er.similarity_pairs(st.session_state.masters)

            # 5) Route candidates
            st.session_state.candidates = er.route_candidates(pairs)

            # 6) LLM results (with persisted notes/context)
            st.session_state.context_book = er.load_context_from_notes()
            st.session_state.llm_results = er.llm_results_df(
                st.session_state.candidates,
                st.session_state.masters,
                array_batch_size=er.LLM_ARRAY_SIZE,
                context_book=st.session_state.context_book,
                scope=st.session_state.scope,
                seed_focals=st.session_state.seed_focals,
                seed_pairs=st.session_state.seed_pairs,
                pairs_for_graph=pairs,
            )

            # 7) Build review queue (NEEDS_CONFIRMATION)
            st.session_state.review_queue = er.build_review_queue(st.session_state.llm_results)

            # Next stage
            if st.session_state.review_queue is not None and not st.session_state.review_queue.empty:
                st.session_state.pipeline_stage = "review"
            else:
                st.session_state.pipeline_stage = "final_review"

            st.success("Initial pipeline completed ✔")
            st.rerun()

    except Exception as e:
        st.error(f"Error during pipeline: {e}")
        st.session_state.pipeline_stage = "input"


# =========================
# Stage: Review (LLM NEEDS_CONFIRMATION)
# =========================
def display_llm_review():
    st.header("🤖 LLM Review — Human Confirmation")

    if st.session_state.review_queue.empty:
        st.info("No pairs require human confirmation. Proceeding to Final Review.")
        st.session_state.pipeline_stage = "final_review"
        st.rerun()
        return

    st.write(f"**{len(st.session_state.review_queue)}** pairs require review.")

    st.markdown("#### Rerun Options")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.session_state.rerun_scope = st.radio(
            "Rerun Scope (applies when you click 'Rerun LLM with Notes')",
            options=["all", "foci", "foci_related"],
            index=["all", "foci", "foci_related"].index(st.session_state.rerun_scope),
            horizontal=True,
        )
    with c2:
        st.caption("Seed focals")
        st.code(", ".join(sorted(st.session_state.seed_focals)) or "—")
    with c3:
        st.caption("Seed pairs")
        st.code(", ".join(sorted(st.session_state.seed_pairs)) or "—")

    st.markdown("---")
    queue_df = st.session_state.review_queue.copy()

    for idx, row in queue_df.iterrows():
        if str(row.get("status", "QUEUED")) != "QUEUED":
            continue

        pair_key = str(row["pair_key"])
        focal_id = str(row["focal_master_id"])
        candidate_id = str(row["candidate_master_id"])
        score = float(row["score"]) if row.get("score") is not None else 0.0
        llm_confidence = row.get("llm_confidence", None)
        llm_reason = row.get("llm_reason", "")

        focal_name = st.session_state.id2name.get(focal_id, "")
        candidate_name = st.session_state.id2name.get(candidate_id, "")

        with st.expander(f"**{focal_name}** ↔ **{candidate_name}**  ·  similarity={score:.3f}", expanded=True):
            left, right = st.columns([4, 2])
            with left:
                st.write(f"**Focal:** [{focal_id}] {focal_name}")
                st.write(f"**Candidate:** [{candidate_id}] {candidate_name}")
                st.write(f"**Similarity:** {score:.3f}")
                if llm_confidence is not None:
                    st.write(f"**LLM Confidence:** {llm_confidence:.3f}")
                if llm_reason:
                    st.write(f"**LLM Reason:** {llm_reason}")

            with right:
                approve, reject, skip = st.columns(3)
                if approve.button("✅ Approve", use_container_width=True, key=f"approve_{pair_key}"):
                    st.session_state.human_decisions[pair_key] = {
                        "decision": "APPROVED",
                        "pair_key": pair_key,
                        "focal_master_id": focal_id,
                        "candidate_master_id": candidate_id,
                        "score": score,
                        "status": "APPROVED",
                        "reviewer": "streamlit_user",
                        "updated_at": utc_now(),
                        "llm_confidence": llm_confidence,
                        "llm_reason": llm_reason,
                    }
                    st.rerun()
                if reject.button("❌ Reject", use_container_width=True, key=f"reject_{pair_key}"):
                    st.session_state.human_decisions[pair_key] = {
                        "decision": "REJECTED",
                        "status": "REJECTED",
                        "updated_at": utc_now(),
                    }
                    st.rerun()
                if skip.button("⏭️ Skip", use_container_width=True, key=f"skip_{pair_key}"):
                    st.session_state.human_decisions[pair_key] = {
                        "decision": "SKIPPED",
                        "status": "SKIPPED",
                        "updated_at": utc_now(),
                    }
                    st.rerun()

            notes = st.text_area(
                "Add notes for the LLM/context (optional, persisted & used in reruns)",
                key=f"notes_{pair_key}",
                height=70,
                placeholder="e.g., Treat 'Intl' as 'International'; ACME Global Holdings is a holding company, do not merge with operational subsidiaries.",
            )
            cols = st.columns([1, 2, 2, 2])
            with cols[0]:
                if st.button("💾 Save Notes", key=f"save_notes_{pair_key}", use_container_width=True):
                    if notes.strip():
                        st.session_state.context_book.add_global(notes.strip())
                        st.session_state.context_book.add_focal(focal_id, notes.strip())
                        er.persist_note("GLOBAL", notes.strip())
                        er.persist_note("FOCAL", notes.strip(), focal_id=focal_id)
                        st.success("Notes saved")
                    else:
                        st.info("No notes to save.")
            with cols[1]:
                if st.button("🔄 Rerun LLM (full)", key=f"rerun_all_{pair_key}", use_container_width=True):
                    st.session_state.rerun_scope = "all"
                    rerun_llm_with_context()
            with cols[2]:
                if st.button("🔁 Rerun LLM (foci only)", key=f"rerun_foci_{pair_key}", use_container_width=True):
                    st.session_state.rerun_scope = "foci"
                    # ensure current focal is included in seed focals
                    st.session_state.seed_focals = set(st.session_state.seed_focals) | {focal_id}
                    rerun_llm_with_context()
            with cols[3]:
                st.caption("No rerun → just proceed to final table when ready.")

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔄 Rerun LLM with All Current Notes", use_container_width=True):
            rerun_llm_with_context()
    with b2:
        if st.button("✅ Approve All Remaining", use_container_width=True):
            approve_all_remaining()
    with b3:
        if st.button("➡️ Proceed to Final Review", use_container_width=True):
            st.session_state.pipeline_stage = "final_review"
            st.rerun()


def rerun_llm_with_context():
    try:
        with st.spinner("Rejudging queued pairs using accumulated notes/context..."):
            queued = st.session_state.review_queue[
                st.session_state.review_queue["status"] == "QUEUED"
            ].copy()
            if queued.empty:
                st.info("No queued pairs to rerun.")
                return

            judgments_by_pk = er._rerun_llm_for_scope(
                queued,
                st.session_state.id2name,
                st.session_state.context_book,
                st.session_state.rerun_scope,
                set(st.session_state.seed_focals),
            )

            auto_approved = 0
            for pk, j in judgments_by_pk.items():
                mask = st.session_state.review_queue["pair_key"] == pk
                if not mask.any():
                    continue
                idx = st.session_state.review_queue[mask].index[0]

                st.session_state.review_queue.at[idx, "llm_decision"] = j.get("llm_decision")
                st.session_state.review_queue.at[idx, "llm_confidence"] = j.get("llm_confidence")
                st.session_state.review_queue.at[idx, "llm_reason"] = j.get("llm_reason")
                st.session_state.review_queue.at[idx, "context_used"] = j.get("context_used")
                st.session_state.review_queue.at[idx, "prompt_version"] = j.get("prompt_version")
                st.session_state.review_queue.at[idx, "model_name"] = j.get("model_name")
                st.session_state.review_queue.at[idx, "decided_at"] = j.get("decided_at")

                if (
                    j.get("llm_decision") == "YES"
                    and j.get("llm_confidence") is not None
                    and float(j["llm_confidence"]) >= er.AUTO_APPROVE_RERUN_YES_CONF
                ):
                    row = st.session_state.review_queue.loc[idx]
                    st.session_state.human_decisions[pk] = {
                        "decision": "APPROVED",
                        "pair_key": pk,
                        "focal_master_id": str(row["focal_master_id"]),
                        "candidate_master_id": str(row["candidate_master_id"]),
                        "score": float(row.get("score", 0.0)),
                        "status": "APPROVED",
                        "reviewer": "auto_approved_after_context",
                        "updated_at": utc_now(),
                        "llm_confidence": j.get("llm_confidence"),
                        "llm_reason": j.get("llm_reason"),
                    }
                    auto_approved += 1

            st.success(f"LLM rerun complete. Auto-approved {auto_approved} strong YES decisions.")
            st.rerun()
    except Exception as e:
        st.error(f"Error during LLM rerun: {e}")


def approve_all_remaining():
    queued = st.session_state.review_queue[
        st.session_state.review_queue["status"] == "QUEUED"
    ].copy()
    count = 0
    for _, row in queued.iterrows():
        pk = str(row["pair_key"])
        if pk in st.session_state.human_decisions:
            continue
        st.session_state.human_decisions[pk] = {
            "decision": "APPROVED",
            "pair_key": pk,
            "focal_master_id": str(row["focal_master_id"]),
            "candidate_master_id": str(row["candidate_master_id"]),
            "score": float(row["score"]),
            "status": "APPROVED",
            "reviewer": "bulk_approved",
            "updated_at": utc_now(),
        }
        count += 1
    st.success(f"Approved {count} remaining pairs.")
    st.rerun()


# =========================
# Stage: Final Review (ALL pairs)
# =========================
def display_final_review():
    st.header("📋 Final Table — Review ALL Pairs Before Merging")

    # Build a lookup from LLM results (pair_key → (decision, confidence, score))
    llm_map: Dict[str, Dict[str, Any]] = {}
    if st.session_state.llm_results is not None and not st.session_state.llm_results.empty:
        for _, rr in st.session_state.llm_results.iterrows():
            focal = str(rr["focal_master_id"])
            for it in rr["results"]:
                cand = str(it["candidate_master_id"])
                pk = er.pair_key(focal, cand)
                llm_map[pk] = {
                    "llm_decision": it.get("llm_decision"),
                    "llm_confidence": it.get("llm_confidence"),
                    "llm_score": float(it.get("score", 0.0)),
                    "llm_reason": it.get("llm_reason"),
                }

    proposals: List[Dict[str, Any]] = []

    # Start from ALL candidate pairs (every pairwise combination)
    for _, r in st.session_state.candidates.iterrows():
        pk = str(r["pair_key"])
        m1 = str(r["master_a_id"])
        m2 = str(r["master_b_id"])
        score = float(r["score"])
        route = r["route"]
        focal_name = st.session_state.id2name.get(m1, "")
        cand_name = st.session_state.id2name.get(m2, "")

        # Annotate from LLM if available
        l = llm_map.get(pk, {})
        llm_decision = l.get("llm_decision")
        llm_conf = l.get("llm_confidence")
        llm_reason = l.get("llm_reason")

        # Default source/status based on route/LLM
        source = route
        if route == "AUTO_YES_95":
            status = "PROPOSED"
            source = "AUTO_95"
        elif route == "AUTO_NO":
            status = "REJECTED"
            source = "AUTO_NO"
        else:
            # LLM band
            if llm_decision == "YES":
                status = "PROPOSED"
                source = "LLM_YES"
            elif llm_decision == "NO":
                status = "REJECTED"
                source = "LLM_NO"
            elif llm_decision == "NEEDS_CONFIRMATION":
                status = "PROPOSED"
                source = "LLM_NEEDS"
            else:
                # not judged (e.g., beyond top-N) → let user decide
                status = "PROPOSED"
                source = "LLM_UNJUDGED"

        proposals.append(
            {
                "pair_key": pk,
                "focal_id": m1,
                "candidate_id": m2,
                "focal_name": focal_name,
                "candidate_name": cand_name,
                "score": score,
                "source": source,
                "status": status,
                "confidence": float(llm_conf) if llm_conf is not None else (1.0 if source == "AUTO_95" else 0.6),
                "reason": llm_reason or ("auto_threshold" if source == "AUTO_95" else ""),
            }
        )

    # Overlay any explicit HUMAN approvals gathered during Review step (still show as PROPOSED by default)
    for pk, d in st.session_state.human_decisions.items():
        if d.get("decision") == "APPROVED":
            # If exists, update; otherwise append
            found_idx = next((i for i, x in enumerate(proposals) if x["pair_key"] == pk), None)
            row_update = {
                "pair_key": pk,
                "focal_id": d["focal_master_id"],
                "candidate_id": d["candidate_master_id"],
                "focal_name": st.session_state.id2name.get(d["focal_master_id"], ""),
                "candidate_name": st.session_state.id2name.get(d["candidate_master_id"], ""),
                "score": float(d.get("score", 0.0)),
                "source": "HUMAN_APPROVED",
                "status": "PROPOSED",
                "confidence": float(d.get("llm_confidence", 1.0)),
                "reason": d.get("llm_reason", "human_approved"),
            }
            if found_idx is not None:
                proposals[found_idx].update(row_update)
            else:
                proposals.append(row_update)

    if not proposals:
        st.info("No pairs to review.")
        if st.button("✅ Complete Cycle", use_container_width=True):
            st.session_state.pipeline_stage = "complete"
            st.rerun()
        return

    proposals_df = pd.DataFrame(proposals)

    st.subheader(f"📊 Review {len(proposals_df)} Pairs")
    edited = st.data_editor(
        proposals_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status", options=["PROPOSED", "APPROVED", "REJECTED"], required=True
            ),
            "score": st.column_config.NumberColumn("Similarity", format="%.3f", min_value=0.0, max_value=1.0),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f", min_value=0.0, max_value=1.0),
            "source": st.column_config.TextColumn("Source"),
            "reason": st.column_config.TextColumn("Reason"),
            "focal_name": st.column_config.TextColumn("Focal"),
            "candidate_name": st.column_config.TextColumn("Candidate"),
        },
        key="final_proposals_editor",
    )
    st.session_state.final_proposals = edited

    approved_count = len(edited[edited["status"] == "APPROVED"])
    rejected_count = len(edited[edited["status"] == "REJECTED"])
    proposed_count = len(edited[edited["status"] == "PROPOSED"])

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Approved", approved_count)
    c2.metric("❌ Rejected", rejected_count)
    c3.metric("🔄 Proposed", proposed_count)

    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("✅ Approve All Proposed", use_container_width=True):
            st.session_state.final_proposals.loc[
                st.session_state.final_proposals["status"] == "PROPOSED", "status"
            ] = "APPROVED"
            st.rerun()
    with a2:
        if st.button("❌ Reject All Proposed", use_container_width=True):
            st.session_state.final_proposals.loc[
                st.session_state.final_proposals["status"] == "PROPOSED", "status"
            ] = "REJECTED"
            st.rerun()
    with a3:
        if st.button("🚀 Execute Merges", type="primary", use_container_width=True):
            execute_merges()


def execute_merges():
    try:
        with st.spinner("Executing merges and updating mapping..."):
            if st.session_state.final_proposals is None or st.session_state.final_proposals.empty:
                st.warning("No proposals to execute.")
                st.session_state.pipeline_stage = "complete"
                st.rerun()
                return

            approved = st.session_state.final_proposals[
                st.session_state.final_proposals["status"] == "APPROVED"
            ].copy()

            if approved.empty:
                st.info("No approved merges. Completing cycle.")
                st.session_state.final_results = {
                    "full_post": st.session_state.full_base.copy(),
                    "masters_final": er.masters_slice(st.session_state.full_base),
                    "merges_executed": 0,
                }
                st.session_state.pipeline_stage = "complete"
                st.rerun()
                return

            # Convert approved rows into apply_df format expected by cleaner
            apply_rows = []
            for _, row in approved.iterrows():
                apply_rows.append(
                    {
                        "pair_key": row["pair_key"],
                        "m1": row["focal_id"],
                        "m2": row["candidate_id"],
                        "source": row["source"],
                        "score": float(row["score"]),
                        "confidence": float(row["confidence"]),
                        "reason": row.get("reason", ""),
                        "prompt_version": "streamlit_app",
                        "model_name": "streamlit_user",
                        "decided_at": utc_now(),
                    }
                )
            apply_df = pd.DataFrame(apply_rows)

            # Clean → delta mapping
            clean_delta = er.clean_proposals(apply_df)

            # Append decisions history as YES
            decisions_hist = er.read_csv(
                er.DECISIONS_PATH, cols=["pair_key", "decision", "source", "decided_at", "score", "reason"]
            )
            if not apply_df.empty:
                dec = apply_df[
                    ["pair_key", "source", "score", "confidence", "reason", "prompt_version", "model_name", "decided_at"]
                ].copy()
                dec["decision"] = "YES"
                decisions_hist2 = pd.concat([decisions_hist, dec], ignore_index=True, sort=False).drop_duplicates(
                    subset=["pair_key", "decision"], keep="last"
                )
                er.write_csv(decisions_hist2, er.DECISIONS_PATH)

            # Update cumulative mapping
            cum_map_prev = er.read_csv(er.CUM_MAPPING_PATH, cols=["old_master_id", "canonical_master_id"])
            if not clean_delta.empty:
                cum_concat = pd.concat([cum_map_prev, clean_delta], ignore_index=True).drop_duplicates()
                cum_map_new = er.compress_mapping_df(cum_concat)
                er.write_csv(cum_map_new, er.CUM_MAPPING_PATH)
            else:
                cum_map_new = er.compress_mapping_df(cum_map_prev) if not cum_map_prev.empty else cum_map_prev

            # Apply mapping to full_base
            full_post = er.apply_mapping(st.session_state.full_base, cum_map_new)
            masters_final = er.masters_slice(full_post)

            st.session_state.final_results = {
                "full_post": full_post,
                "masters_final": masters_final,
                "merges_executed": len(approved),
            }

            st.session_state.cycle_count += 1
            st.session_state.pipeline_stage = "complete"
            st.success(f"Executed {len(approved)} merges ✔")
            st.rerun()
    except Exception as e:
        st.error(f"Error executing merges: {e}")


# =========================
# Stage: Complete
# =========================
def display_results():
    st.header("🎉 Cycle Complete")

    if "final_results" in st.session_state and st.session_state.final_results is not None:
        res = st.session_state.final_results

        st.subheader("📈 Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Cycle Number", st.session_state.cycle_count)
        with c2:
            st.metric("Merges Executed", res.get("merges_executed", 0))
        with c3:
            original_masters = len(st.session_state.masters) if st.session_state.masters is not None else 0
            final_masters = len(res["masters_final"]) if res.get("masters_final") is not None else 0
            st.metric("Masters Reduced", f"{max(original_masters - final_masters, 0)} ({original_masters} → {final_masters})")

        st.subheader("📊 Final Master Accounts")
        st.dataframe(res["masters_final"], use_container_width=True)

        st.subheader("📋 All Accounts (Post-Merge)")
        display_df = res["full_post"][["account_id", "account_name", "master_account_id", "is_master", "group_size"]].copy()
        display_df = display_df.sort_values(["master_account_id", "account_id"])
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "is_master": st.column_config.CheckboxColumn("Is Master"),
                "group_size": st.column_config.NumberColumn("Group Size", format="%d"),
            },
        )
    else:
        st.info("No results to display.")

    st.markdown("---")
    st.subheader("🔄 Next Steps")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Add New Accounts & Start Next Cycle", type="primary", use_container_width=True):
            # Keep existing mapping/ledgers on disk; reset app state for new cycle
            st.session_state.pipeline_stage = "input"
            # Keep current accounts so the user can append/modify
            st.toast("Ready for next cycle. Add new accounts and run again.")
            st.rerun()
    with c2:
        if st.button("🔁 Rerun on Same Accounts", use_container_width=True):
            # Reset intermediates only
            for key in ["full_base", "masters", "candidates", "llm_results", "review_queue", "final_proposals", "final_results"]:
                st.session_state[key] = None
            st.session_state.human_decisions = {}
            st.session_state.context_book = er.load_context_from_notes()
            st.session_state.pipeline_stage = "processing"
            st.rerun()


# =========================
# Main
# =========================
def main():
    init_session_state()
    sidebar_info()

    st.title("🔄 Account Deduplication Pipeline (Azure OpenAI + Human-in-the-Loop)")
    st.caption(
        "Upload or edit accounts → run dedupe → review LLM uncertainties → adjust final table of **ALL pairs** → execute merges → start next cycle."
    )

    steps = ["input", "processing", "review", "final_review", "complete"]
    current_step = steps.index(st.session_state.pipeline_stage)
    st.progress((current_step + 1) / len(steps), text=f"Stage: {st.session_state.pipeline_stage}")

    if st.session_state.pipeline_stage == "input":
        display_accounts_input()
    elif st.session_state.pipeline_stage == "processing":
        run_initial_pipeline()
    elif st.session_state.pipeline_stage == "review":
        display_llm_review()
    elif st.session_state.pipeline_stage == "final_review":
        display_final_review()
    else:
        display_results()


if __name__ == "__main__":
    main()
