#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Set, List, Any, Tuple
import os

# Import functions from the main deduplication script
from example_run import (
    perfect_match_base, apply_mapping, masters_slice, similarity_pairs,
    route_candidates, llm_results_df, build_review_queue, build_apply,
    clean_proposals, compress_mapping_df, load_context_from_notes,
    persist_note, ContextBook, get_aoai_client, pair_key, utc_now,
    read_csv, write_csv, show, ROOT, DECISIONS_PATH, CUM_MAPPING_PATH,
    NOTES_PATH, _parse_seed_focals, filter_llm_band_by_scope,
    _rerun_llm_for_scope, LLM_ARRAY_SIZE, normalize_name, EMB_WEIGHT, FUZZ_WEIGHT,
    LLM_TOP_N, cosine
)
from rapidfuzz import fuzz

st.set_page_config(
    page_title="Account Deduplication Assistant",
    page_icon="🔍",
    layout="wide"
)

def incremental_recalculate_enhanced(affected_accounts, original_masters, original_pairs, similarity_cols):
    """Enhanced recalculate only the similarity pairs involving affected accounts"""
    if not affected_accounts:
        return original_pairs

    # Keep unaffected pairs
    unaffected_pairs = original_pairs[
        ~(original_pairs['master_a_id'].astype(str).isin(affected_accounts) |
          original_pairs['master_b_id'].astype(str).isin(affected_accounts))
    ].copy()

    # Recalculate affected pairs
    affected_masters = original_masters[
        original_masters['account_id'].astype(str).isin(affected_accounts)
    ].copy()

    if not affected_masters.empty:
        # Recalculate similarities for affected accounts using enhanced function
        new_affected_pairs = similarity_pairs_enhanced(affected_masters, similarity_cols)

        # Also recalculate cross-pairs (affected with unaffected)
        unaffected_masters = original_masters[
            ~original_masters['account_id'].astype(str).isin(affected_accounts)
        ].copy()

        if not unaffected_masters.empty:
            # Calculate affected × unaffected pairs
            cross_pairs = []
            for _, affected_row in affected_masters.iterrows():
                for _, unaffected_row in unaffected_masters.iterrows():
                    a_id = affected_row['account_id']
                    u_id = unaffected_row['account_id']

                    # Build combined text for similarity (same as enhanced function)
                    a_parts = [affected_row['account_name']]
                    u_parts = [unaffected_row['account_name']]
                    
                    if similarity_cols:
                        for col in similarity_cols:
                            if col in affected_row:
                                a_parts.append(str(affected_row[col]) if pd.notna(affected_row[col]) else "")
                            if col in unaffected_row:
                                u_parts.append(str(unaffected_row[col]) if pd.notna(unaffected_row[col]) else "")
                    
                    a_combined = normalize_name(" | ".join(a_parts))
                    u_combined = normalize_name(" | ".join(u_parts))

                    # Simple calculation (would need full embedding in real implementation)
                    fuzz_score = float(fuzz.token_sort_ratio(a_combined, u_combined) / 100.0)
                    # Use fake embedding for now - in real implementation, would recalculate
                    emb_score = fuzz_score * 0.9  # Approximate
                    combined = EMB_WEIGHT * emb_score + FUZZ_WEIGHT * fuzz_score

                    cross_pairs.append((
                        pair_key(a_id, u_id), a_id, u_id,
                        emb_score, fuzz_score, combined
                    ))

            if cross_pairs:
                cross_pairs_df = pd.DataFrame(cross_pairs,
                    columns=["pair_key", "master_a_id", "master_b_id", "emb_score", "fuzz_score", "score"])
                new_affected_pairs = pd.concat([new_affected_pairs, cross_pairs_df], ignore_index=True)

        # Combine all pairs
        updated_pairs = pd.concat([unaffected_pairs, new_affected_pairs], ignore_index=True)
        return updated_pairs.drop_duplicates(subset=['pair_key'])

    return original_pairs


def incremental_recalculate(affected_accounts, original_masters, original_pairs):
    """Original incremental recalculate function - delegates to enhanced version"""
    return incremental_recalculate_enhanced(affected_accounts, original_masters, original_pairs, [])


# Enhanced functions for flexible metadata handling
def similarity_pairs_enhanced(masters_only: pd.DataFrame, similarity_cols: List[str] = None) -> pd.DataFrame:
    """Enhanced similarity calculation that includes metadata columns"""
    mo = masters_only.copy()
    
    # Build combined text for embeddings (name + selected metadata)
    combined_parts = mo["account_name"].fillna("").astype(str)
    
    if similarity_cols:
        for col in similarity_cols:
            if col in mo.columns:
                # Clean and add metadata to the combined string
                combined_parts = combined_parts + " | " + mo[col].fillna("").astype(str)
    
    # Set the combined text
    mo["combined_text"] = combined_parts.str.strip()
    mo["normalized_combined"] = mo["combined_text"].map(normalize_name)
    
    # Use the original similarity_pairs function but with combined text
    try:
        client = get_aoai_client()
        vectors = client.embed_batch(mo["normalized_combined"].tolist())
    except Exception:
        vectors = []
        for n in mo["normalized_combined"].tolist():
            h = abs(hash(n))
            rng = np.random.RandomState(h % (2**32))
            vectors.append(rng.normal(size=1536).tolist())
    mo["embedding"] = vectors

    ids = mo["account_id"].tolist()
    rows = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            va = mo.loc[mo["account_id"]==a, "embedding"].iloc[0]
            vb = mo.loc[mo["account_id"]==b, "embedding"].iloc[0]
            emb_score = float(cosine(va, vb))

            text_a = mo.loc[mo["account_id"]==a, "normalized_combined"].iloc[0]
            text_b = mo.loc[mo["account_id"]==b, "normalized_combined"].iloc[0]
            fuzz_score = float(fuzz.token_sort_ratio(text_a, text_b) / 100.0)

            combined = EMB_WEIGHT * emb_score + FUZZ_WEIGHT * fuzz_score
            rows.append((pair_key(a,b), a, b, emb_score, fuzz_score, combined))
    return pd.DataFrame(rows, columns=["pair_key","master_a_id","master_b_id","emb_score","fuzz_score","score"])


def llm_results_df_enhanced(
    candidates: pd.DataFrame,
    masters_only: pd.DataFrame,
    array_batch_size: int | None = None,
    context_book: ContextBook | None = None,
    scope: str = "all",
    seed_focals: Set[str] | None = None,
    seed_pairs: Set[str] | None = None,
    pairs_for_graph: pd.DataFrame | None = None,
    llm_context_cols: List[str] = None,
) -> pd.DataFrame:
    """Enhanced LLM results that include metadata in context"""
    
    # Use the original function but enhance the payload with metadata
    llm_band = candidates[candidates["route"]=="LLM"].copy()

    # scope restriction (for initial judging)
    llm_band = filter_llm_band_by_scope(
        llm_band,
        scope=scope,
        seed_focals=_parse_seed_focals(seed_focals),
        seed_pairs=_parse_seed_focals(seed_pairs),
        pairs_df=pairs_for_graph if pairs_for_graph is not None else candidates.rename(columns={"master_a_id":"master_a_id","master_b_id":"master_b_id"})
    )

    if llm_band.empty:
        return pd.DataFrame(columns=["focal_master_id","results","model_name","prompt_version","decided_at"])

    array_batch_size = array_batch_size or LLM_ARRAY_SIZE
    # rank by score within focal
    llm_band["rank"] = llm_band.groupby("master_a_id")["score"].rank(ascending=False, method="first")
    llm_band = llm_band[llm_band["rank"] <= LLM_TOP_N]

    # Build enhanced id2data mapping with metadata
    id2data = {}
    for _, row in masters_only.iterrows():
        account_id = row["account_id"]
        data = {"id": str(account_id), "name": row["account_name"]}
        
        # Add selected metadata for LLM context
        if llm_context_cols:
            metadata = {}
            for col in llm_context_cols:
                if col in row and pd.notna(row[col]):
                    metadata[col] = str(row[col])
            if metadata:
                data["metadata"] = metadata
        
        id2data[account_id] = data

    results = []
    client = get_aoai_client()

    for focal, grp in llm_band.groupby("master_a_id"):
        focal_payload = id2data.get(focal, {"id": str(focal), "name": ""})
        grp_sorted = grp.sort_values("score", ascending=False)
        # chunk into arrays of size array_batch_size
        chunked = [grp_sorted.iloc[i:i+array_batch_size] for i in range(0, len(grp_sorted), array_batch_size)]
        judgments_all: List[Dict[str, Any]] = []
        extra_ctx = context_book.render_context(str(focal)) if context_book else None
        for ch in chunked:
            candidates_payload = [
                id2data.get(r["master_b_id"], {"id": str(r["master_b_id"]), "name": "", "score": float(r["score"])})
                for _, r in ch.iterrows()
            ]
            # Ensure score is set for each candidate
            for i, (_, r) in enumerate(ch.iterrows()):
                candidates_payload[i]["score"] = float(r["score"])
                
            judgments = client.judge_matches(focal_payload, candidates_payload, extra_context=extra_ctx)
            judgments_all.extend(judgments)

        results.append({
            "focal_master_id": str(focal),
            "results": judgments_all,
            "model_name": os.getenv("AOAI_CHAT_DEPLOYMENT","gpt-4.1"),
            "prompt_version": f"v1_batch{array_batch_size}_with_metadata",
            "decided_at": utc_now()
        })
    return pd.DataFrame(results)

def init_session_state():
    """Initialize session state variables"""
    if 'accounts_df' not in st.session_state:
        # Start with demo data including metadata columns
        st.session_state.accounts_df = pd.DataFrame([
            ("a1", "ACME Corp", "Manufacturing", "New York", "Large", "B2B"),
            ("a2", "ACME Corporation", "Manufacturing", "NY", "Enterprise", "B2B"),
            ("a3", "Acme C", "Manufacturing", "New York City", "Large", "Business"),
            ("a4", "A.C.M.E. Corp.", "Industrial", "NYC", "Big", "B2B"),

            ("b1", "Globex, LLC", "Logistics", "Texas", "Medium", "B2B"),
            ("b2", "Globex, L.L.C.", "Shipping", "TX", "Mid-size", "B2B"),
            ("b3", "Globex Limited Liability Company", "Logistics", "Texas", "Medium", "B2B"),

            ("c1", "Loblaws Ltd", "Retail", "Canada", "Large", "B2C"),
            ("c2", "Loblaws Limited", "Grocery", "Toronto", "Large", "Consumer"),

            ("d1", "Microsoft", "Technology", "Washington", "Enterprise", "B2B"),
            ("d2", "Microsoft Corporation", "Software", "WA", "Large", "B2B"),

            ("k3", "Microsoft International LLC", "Consulting", "NYC", "Enterprise", "B2B"),

            ("e1", "Unrelated Co", "Services", "California", "Small", "B2B"),
            ("e2", "Unrelated Company", "Consulting", "CA", "Small", "Business"),

            ("f1", "Zeta Holdings", "Finance", "Delaware", "Large", "B2B"),
            ("f2", "Zta Holding Company", "Investment", "DE", "Large", "Finance"),
            ("f3", "Z Holding Company", "Holdings", "Delaware", "Large", "Investment"),

            ("g1", "Initech", "Technology", "California", "Medium", "B2B"),
            ("g2", "Initech LLC", "Software", "CA", "Medium", "B2B"),

            ("h1", "Wayne Enterprises", "Conglomerate", "Gotham", "Enterprise", "Mixed"),
        ], columns=["account_id", "account_name", "industry", "location", "company_size", "business_type"])

    if 'dedup_results' not in st.session_state:
        st.session_state.dedup_results = {}

    if 'context_book' not in st.session_state:
        st.session_state.context_book = load_context_from_notes()

    if 'review_comments' not in st.session_state:
        st.session_state.review_comments = {}

    # New: Configuration for metadata column usage
    if 'similarity_columns' not in st.session_state:
        st.session_state.similarity_columns = []  # Up to 2 columns for embeddings

    if 'llm_context_columns' not in st.session_state:
        st.session_state.llm_context_columns = []  # Up to 4 columns for LLM context

    if 'llm_decision_comments' not in st.session_state:
        st.session_state.llm_decision_comments = {}  # Comments for LLM decisions rerun

def add_account_row():
    """Add a new empty row to accounts"""
    # Get current column structure
    columns = st.session_state.accounts_df.columns.tolist()
    # Create empty row with same structure
    empty_values = [""] * len(columns)
    new_row = pd.DataFrame([empty_values], columns=columns)
    st.session_state.accounts_df = pd.concat([st.session_state.accounts_df, new_row], ignore_index=True)

def remove_account_row(index):
    """Remove a specific row from accounts"""
    st.session_state.accounts_df = st.session_state.accounts_df.drop(index).reset_index(drop=True)

def detect_decision_changes():
    """Detect what LLM decisions have been modified from original"""
    if 'dedup_results' not in st.session_state:
        return set(), {}

    # Get original and current decisions
    original_llm = st.session_state.dedup_results.get('llm_results', pd.DataFrame())
    current_decisions = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())

    if original_llm.empty or current_decisions.empty:
        return set(), {}

    # Build original decisions lookup
    original_decisions = {}
    for _, r in original_llm.iterrows():
        focal = r["focal_master_id"]
        for item in r["results"]:
            pk = pair_key(focal, item["candidate_master_id"])
            original_decisions[pk] = item.get('llm_decision', 'UNKNOWN')

    # Find changes
    changed_pairs = set()
    changes_detail = {}

    for _, row in current_decisions.iterrows():
        pk = row['pair_key']
        current_decision = row['llm_decision']
        original_decision = original_decisions.get(pk)

        if original_decision and current_decision != original_decision:
            changed_pairs.add(pk)
            changes_detail[pk] = {
                'original': original_decision,
                'current': current_decision,
                'focal_id': row['focal_id'],
                'candidate_id': row['candidate_id']
            }

    return changed_pairs, changes_detail

def find_affected_accounts(changed_pairs, changes_detail, current_decisions_df):
    """Find all accounts that could be affected by the changed decisions"""
    affected_accounts = set()

    # Direct accounts from changed pairs
    for pk in changed_pairs:
        if pk in changes_detail:
            affected_accounts.add(changes_detail[pk]['focal_id'])
            affected_accounts.add(changes_detail[pk]['candidate_id'])

    # Find potential cascade effects
    # Look for any YES decisions involving the affected accounts
    for _, row in current_decisions_df.iterrows():
        if row['llm_decision'] == 'YES':
            focal_id = str(row['focal_id'])
            candidate_id = str(row['candidate_id'])

            # If either account is already affected, add the other
            if focal_id in affected_accounts:
                affected_accounts.add(candidate_id)
            elif candidate_id in affected_accounts:
                affected_accounts.add(focal_id)

    return affected_accounts

def incremental_recalculate(affected_accounts, original_masters, original_pairs):
    """Recalculate only the similarity pairs involving affected accounts"""
    if not affected_accounts:
        return original_pairs

    # Keep unaffected pairs
    unaffected_pairs = original_pairs[
        ~(original_pairs['master_a_id'].astype(str).isin(affected_accounts) |
          original_pairs['master_b_id'].astype(str).isin(affected_accounts))
    ].copy()

    # Recalculate affected pairs
    affected_masters = original_masters[
        original_masters['account_id'].astype(str).isin(affected_accounts)
    ].copy()

    if not affected_masters.empty:
        # Recalculate similarities for affected accounts
        new_affected_pairs = similarity_pairs(affected_masters)

        # Also recalculate cross-pairs (affected with unaffected)
        unaffected_masters = original_masters[
            ~original_masters['account_id'].astype(str).isin(affected_accounts)
        ].copy()

        if not unaffected_masters.empty:
            # Calculate affected × unaffected pairs
            cross_pairs = []
            for _, affected_row in affected_masters.iterrows():
                for _, unaffected_row in unaffected_masters.iterrows():
                    a_id = affected_row['account_id']
                    u_id = unaffected_row['account_id']

                    # Calculate similarity between these two
                    a_name = normalize_name(affected_row['account_name'])
                    u_name = normalize_name(unaffected_row['account_name'])

                    # Simple calculation (would need full embedding in real implementation)
                    fuzz_score = float(fuzz.token_sort_ratio(a_name, u_name) / 100.0)
                    # Use fake embedding for now - in real implementation, would recalculate
                    emb_score = fuzz_score * 0.9  # Approximate
                    combined = EMB_WEIGHT * emb_score + FUZZ_WEIGHT * fuzz_score

                    cross_pairs.append((
                        pair_key(a_id, u_id), a_id, u_id,
                        emb_score, fuzz_score, combined
                    ))

            if cross_pairs:
                cross_pairs_df = pd.DataFrame(cross_pairs,
                    columns=["pair_key", "master_a_id", "master_b_id", "emb_score", "fuzz_score", "score"])
                new_affected_pairs = pd.concat([new_affected_pairs, cross_pairs_df], ignore_index=True)

        # Combine all pairs
        updated_pairs = pd.concat([unaffected_pairs, new_affected_pairs], ignore_index=True)
        return updated_pairs.drop_duplicates(subset=['pair_key'])

    return original_pairs

def apply_decisions_and_finalize():
    """Apply approved decisions with smart incremental updates"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to finalize")
        return

    try:
        # Detect what changed
        changed_pairs, changes_detail = detect_decision_changes()

        with st.spinner("Analyzing changes and computing updates..."):
            if changed_pairs:
                st.info(f"🔍 Detected {len(changed_pairs)} changed decisions - using incremental update")

                # Find affected accounts
                current_decisions = st.session_state.dedup_results['llm_decisions']
                affected_accounts = find_affected_accounts(changed_pairs, changes_detail, current_decisions)
                st.info(f"📊 {len(affected_accounts)} accounts affected by changes")

                # Incremental recalculation
                original_masters = st.session_state.dedup_results['masters']
                original_pairs = st.session_state.dedup_results['pairs']

                progress_bar = st.progress(0.3)
                updated_pairs = incremental_recalculate_enhanced(affected_accounts, original_masters, original_pairs, st.session_state.similarity_columns)
                progress_bar.progress(0.6)

                # Update candidates with new routing
                updated_candidates = route_candidates(updated_pairs)
                progress_bar.progress(0.8)

                # Store updated data
                st.session_state.dedup_results['pairs'] = updated_pairs
                st.session_state.dedup_results['candidates'] = updated_candidates

                progress_bar.progress(1.0)
                st.success(f"✅ Incremental update completed! Only recalculated {len(affected_accounts)} accounts instead of all {len(original_masters)}")
            else:
                st.info("✅ No changes detected - using existing calculations")

        # Get current data
        candidates = st.session_state.dedup_results['candidates']
        llm_results = st.session_state.dedup_results['llm_results']
        llm_decisions_df = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())
        accounts_df = st.session_state.dedup_results['accounts_input']

        # Create human approved dataframe from modified LLM decisions
        human_approved_rows = []
        if not llm_decisions_df.empty:
            yes_decisions = llm_decisions_df[llm_decisions_df['llm_decision'] == 'YES']
            for _, row in yes_decisions.iterrows():
                human_approved_rows.append({
                    'pair_key': row['pair_key'],
                    'focal_master_id': row['focal_id'],
                    'candidate_master_id': row['candidate_id'],
                    'score': row['score'],
                    'llm_confidence': row['llm_confidence'],
                    'llm_reason': row['llm_reason'],
                    'status': 'APPROVED',
                    'decision': 'HUMAN_APPROVED',
                    'reviewer': 'streamlit_user',
                    'notes': 'Modified in UI',
                    'updated_at': utc_now()
                })

        human_approved = pd.DataFrame(human_approved_rows)

        # Build apply proposals
        decisions_hist = read_csv(DECISIONS_PATH, cols=["pair_key", "decision", "source", "decided_at", "score", "reason"])
        apply_df = build_apply(candidates, llm_results, human_approved, decisions_hist)

        # Clean proposals and create mapping
        clean_delta = clean_proposals(apply_df)

        # Apply mapping to get final results
        full_base = st.session_state.dedup_results['full_base']
        cum_map_prev = read_csv(CUM_MAPPING_PATH, cols=["old_master_id", "canonical_master_id"])

        if not clean_delta.empty:
            cum_concat = pd.concat([cum_map_prev, clean_delta], ignore_index=True).drop_duplicates()
            cum_map_now = compress_mapping_df(cum_concat)
        else:
            cum_map_now = cum_map_prev

        # Apply final mapping
        full_final = apply_mapping(full_base, cum_map_now)
        masters_final = masters_slice(full_final)

        # Update session state with final results
        st.session_state.dedup_results.update({
            'apply_proposals': apply_df,
            'clean_mapping': clean_delta,
            'cumulative_mapping': cum_map_now,
            'full_final': full_final,
            'masters_final': masters_final,
            'human_approved': human_approved,
            'changes_applied': changes_detail
        })

        if changed_pairs:
            st.success(f"✅ Smart incremental update completed! Changed {len(changed_pairs)} decisions affecting {len(find_affected_accounts(changed_pairs, changes_detail, llm_decisions_df))} accounts.")
        else:
            st.success("✅ Decisions applied and final results generated!")

    except Exception as e:
        st.error(f"Error applying decisions: {str(e)}")

def run_deduplication_process():
    """Execute the full deduplication pipeline"""
    try:
        accounts_df = st.session_state.accounts_df.copy()

        # Validate required columns exist
        if 'account_id' not in accounts_df.columns or 'account_name' not in accounts_df.columns:
            st.error("account_id and account_name columns are required")
            return

        # Filter out empty rows (check required columns only)
        accounts_df = accounts_df[
            (accounts_df['account_id'].str.strip() != '') &
            (accounts_df['account_name'].str.strip() != '')
        ]

        if accounts_df.empty:
            st.error("No valid accounts to process")
            return

        with st.spinner("Running deduplication process..."):
            # Get metadata configuration
            similarity_cols = st.session_state.similarity_columns
            llm_context_cols = st.session_state.llm_context_columns

            # Step 1: Perfect match base
            full_base = perfect_match_base(accounts_df)

            # Step 2: Apply cumulative mapping
            cum_map_prev = read_csv(CUM_MAPPING_PATH, cols=["old_master_id", "canonical_master_id"])
            full_after_hist = apply_mapping(full_base, cum_map_prev)

            # Step 3: Masters & pairs
            masters = masters_slice(full_after_hist)
            
            # Enhanced similarity calculation with metadata
            pairs = similarity_pairs_enhanced(masters, similarity_cols)

            # Step 4: Route candidates
            candidates = route_candidates(pairs)

            # Step 5: LLM judgments with enhanced context
            context_book = st.session_state.context_book
            llm_res = llm_results_df_enhanced(
                candidates, masters,
                array_batch_size=LLM_ARRAY_SIZE,
                context_book=context_book,
                scope="all",
                seed_focals=set(),
                seed_pairs=set(),
                pairs_for_graph=pairs,
                llm_context_cols=llm_context_cols
            )

            # Step 6: Build review queue
            review_queue = build_review_queue(llm_res)

            # Create comprehensive LLM decisions table
            llm_decisions = []
            if not llm_res.empty:
                for _, r in llm_res.iterrows():
                    focal = r["focal_master_id"]
                    focal_name = dict(zip(masters["account_id"].astype(str), masters["account_name"])).get(focal, "")
                    for item in r["results"]:
                        candidate_id = item["candidate_master_id"]
                        candidate_name = dict(zip(masters["account_id"].astype(str), masters["account_name"])).get(candidate_id, "")
                        llm_decisions.append({
                            'pair_key': pair_key(focal, candidate_id),
                            'focal_id': focal,
                            'focal_name': focal_name,
                            'candidate_id': candidate_id,
                            'candidate_name': candidate_name,
                            'llm_decision': item.get('llm_decision', 'UNKNOWN'),
                            'llm_confidence': item.get('llm_confidence', None),
                            'llm_reason': item.get('llm_reason', ''),
                            'score': item.get('score', None),
                            'context_used': item.get('context_used', ''),
                            'prompt_version': item.get('prompt_version', ''),
                            'rerun_from': None,  # No previous decision for initial run
                            'rerun_at': None,    # No rerun for initial run
                        })

            llm_decisions_df = pd.DataFrame(llm_decisions)

            # Store results in session state
            st.session_state.dedup_results = {
                'accounts_input': accounts_df,
                'full_base': full_base,
                'full_after_hist': full_after_hist,
                'masters': masters,
                'pairs': pairs,
                'candidates': candidates,
                'llm_results': llm_res,
                'llm_decisions': llm_decisions_df,
                'review_queue': review_queue,
                'id2name': dict(zip(masters["account_id"].astype(str), masters["account_name"]))
            }

        st.success("Deduplication process completed!")

    except Exception as e:
        st.error(f"Error during deduplication: {str(e)}")

def handle_human_decision(pair_key, focal_id, candidate_id, decision, notes=""):
    """Handle human accept/deny decisions with lineage tracking"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        return

    review_queue = st.session_state.dedup_results.get('review_queue', pd.DataFrame())

    if review_queue.empty:
        return

    # Find the pair in review queue
    mask = review_queue['pair_key'] == pair_key
    if not mask.any():
        return

    idx = review_queue[mask].index[0]

    # Store original decision for lineage
    original_decision = review_queue.loc[idx, 'llm_decision'] if 'llm_decision' in review_queue.columns else 'NEEDS_CONFIRMATION'

    # Update the review queue with human decision
    review_queue.loc[idx, 'llm_decision'] = decision
    review_queue.loc[idx, 'human_decision'] = decision
    review_queue.loc[idx, 'reviewer'] = os.getenv("USERNAME") or os.getenv("USER") or "streamlit_user"
    review_queue.loc[idx, 'notes'] = notes
    review_queue.loc[idx, 'status'] = 'APPROVED' if decision == 'YES' else 'REJECTED'
    review_queue.loc[idx, 'decision'] = 'HUMAN_APPROVED' if decision == 'YES' else 'HUMAN_REJECTED'
    review_queue.loc[idx, 'updated_at'] = utc_now()

    # Add lineage tracking
    if 'rerun_from' not in review_queue.columns:
        review_queue['rerun_from'] = None
    if 'human_decision_from' not in review_queue.columns:
        review_queue['human_decision_from'] = None

    review_queue.loc[idx, 'human_decision_from'] = original_decision
    review_queue.loc[idx, 'human_decision_at'] = utc_now()

    # Update session state
    st.session_state.dedup_results['review_queue'] = review_queue

    # Also update LLM decisions dataframe
    llm_decisions_df = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())
    if not llm_decisions_df.empty:
        llm_mask = llm_decisions_df['pair_key'] == pair_key
        if llm_mask.any():
            llm_idx = llm_decisions_df[llm_mask].index[0]
            llm_decisions_df.loc[llm_idx, 'llm_decision'] = decision
            llm_decisions_df.loc[llm_idx, 'human_override'] = True
            llm_decisions_df.loc[llm_idx, 'human_decision_from'] = original_decision
            st.session_state.dedup_results['llm_decisions'] = llm_decisions_df

def rerun_llm_with_decision_comments(scope="all", focal_ids=None):
    """Rerun LLM with accumulated decision comments and changes"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to rerun")
        return

    try:
        # Add decision comments to context book
        for comment_id, comment_text in st.session_state.llm_decision_comments.items():
            if comment_text.strip():
                if scope == "all":
                    st.session_state.context_book.add_global(comment_text)
                    persist_note("GLOBAL", comment_text)
                else:
                    # For focal-specific runs, add to specific focal
                    if focal_ids:
                        for focal_id in focal_ids:
                            st.session_state.context_book.add_focal(focal_id, comment_text)
                            persist_note("FOCAL", comment_text, focal_id)

        # Get decision changes info for context
        changed_pairs, changes_detail = detect_decision_changes()
        if changed_pairs:
            change_summary = f"User made {len(changed_pairs)} decision changes: " + ", ".join([
                f"{pk}({detail['original']}→{detail['current']})" 
                for pk, detail in list(changes_detail.items())[:3]  # Show first 3
            ])
            if len(changes_detail) > 3:
                change_summary += f" and {len(changes_detail)-3} more"
            
            # Add change summary as global context
            st.session_state.context_book.add_global(change_summary)
            persist_note("GLOBAL", change_summary)

        # Rerun ALL LLM pairs (not just review queue) with enhanced context
        with st.spinner(f"Rerunning LLM with decision context for scope: {scope}..."):
            candidates = st.session_state.dedup_results['candidates']
            masters = st.session_state.dedup_results['masters']
            
            # Count what we're about to rerun for user feedback
            llm_band = candidates[candidates["route"] == "LLM"]
            if scope == "all":
                pairs_to_rerun = len(llm_band)
                scope_description = "ALL LLM pairs"
            elif scope == "foci" and focal_ids:
                pairs_to_rerun = len(llm_band[llm_band["master_a_id"].astype(str).isin(focal_ids)])
                scope_description = f"pairs involving focals: {', '.join(focal_ids)}"
            elif scope == "foci_related" and focal_ids:
                # Get graph-connected focals
                from example_run import _build_adj_from_pairs, _connected_component_nodes
                pairs_df = st.session_state.dedup_results['pairs']
                adj = _build_adj_from_pairs(pairs_df)
                related_focals = _connected_component_nodes(adj, focal_ids)
                pairs_to_rerun = len(llm_band[llm_band["master_a_id"].astype(str).isin(related_focals)])
                scope_description = f"pairs involving {len(related_focals)} related focals (from seeds: {', '.join(focal_ids)})"
            else:
                pairs_to_rerun = len(llm_band)
                scope_description = "ALL LLM pairs (fallback)"

            st.info(f"🔄 **About to rerun {pairs_to_rerun} LLM decisions** for {scope_description}")
            
            # Store original decisions for comparison
            original_llm_decisions = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())
            
            # Re-run LLM for all LLM candidates with new context
            llm_res = llm_results_df_enhanced(
                candidates, masters,
                array_batch_size=LLM_ARRAY_SIZE,
                context_book=st.session_state.context_book,
                scope=scope,
                seed_focals=set(focal_ids) if focal_ids else set(),
                seed_pairs=set(),
                pairs_for_graph=st.session_state.dedup_results['pairs'],
                llm_context_cols=st.session_state.llm_context_columns
            )

            # Update results
            st.session_state.dedup_results['llm_results'] = llm_res

            # Update existing LLM decisions dataframe with rerun results (don't replace entirely)
            original_llm_decisions = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())
            decision_changes = {}
            
            if not llm_res.empty:
                id2name = dict(zip(masters["account_id"].astype(str), masters["account_name"]))
                
                # Start with the original llm_decisions and update with rerun results
                updated_llm_decisions = original_llm_decisions.copy() if not original_llm_decisions.empty else pd.DataFrame()
                
                # Ensure rerun tracking columns exist
                if not updated_llm_decisions.empty:
                    if 'rerun_from' not in updated_llm_decisions.columns:
                        updated_llm_decisions['rerun_from'] = None
                    if 'rerun_at' not in updated_llm_decisions.columns:
                        updated_llm_decisions['rerun_at'] = None
                
                # Process rerun results
                rerun_entries = []
                for _, r in llm_res.iterrows():
                    focal = r["focal_master_id"]
                    focal_name = id2name.get(focal, "")
                    for item in r["results"]:
                        candidate_id = item["candidate_master_id"]
                        candidate_name = id2name.get(candidate_id, "")
                        pk = pair_key(focal, candidate_id)
                        new_decision = item.get('llm_decision', 'UNKNOWN')
                        
                        # Track what changed
                        rerun_from = None
                        if not original_llm_decisions.empty:
                            original_row = original_llm_decisions[original_llm_decisions['pair_key'] == pk]
                            if not original_row.empty:
                                old_decision = original_row.iloc[0]['llm_decision']
                                rerun_from = old_decision  # Track what it was before rerun
                                if old_decision != new_decision:
                                    decision_changes[pk] = {
                                        'old': old_decision,
                                        'new': new_decision,
                                        'focal_id': focal,
                                        'candidate_id': candidate_id
                                    }
                        
                        rerun_entry = {
                            'pair_key': pk,
                            'focal_id': focal,
                            'focal_name': focal_name,
                            'candidate_id': candidate_id,
                            'candidate_name': candidate_name,
                            'llm_decision': new_decision,
                            'llm_confidence': item.get('llm_confidence', None),
                            'llm_reason': item.get('llm_reason', ''),
                            'score': item.get('score', None),
                            'context_used': item.get('context_used', ''),
                            'prompt_version': item.get('prompt_version', ''),
                            'rerun_from': rerun_from,  # Track original decision for lineage
                            'rerun_at': utc_now() if rerun_from else None,  # Track when rerun happened
                        }
                        rerun_entries.append(rerun_entry)
                
                # Update or add rerun entries to the existing dataframe
                if updated_llm_decisions.empty:
                    # If no existing decisions, create new dataframe
                    updated_llm_decisions = pd.DataFrame(rerun_entries)
                else:
                    # Update existing entries or add new ones
                    for entry in rerun_entries:
                        pk = entry['pair_key']
                        mask = updated_llm_decisions['pair_key'] == pk
                        if mask.any():
                            # Update existing entry
                            idx = updated_llm_decisions[mask].index[0]
                            for key, value in entry.items():
                                updated_llm_decisions.loc[idx, key] = value
                        else:
                            # Add new entry
                            updated_llm_decisions = pd.concat([updated_llm_decisions, pd.DataFrame([entry])], ignore_index=True)

            st.session_state.dedup_results['llm_decisions'] = updated_llm_decisions

            # Rebuild review queue
            review_queue = build_review_queue(llm_res)
            st.session_state.dedup_results['review_queue'] = review_queue

            # Show what changed with detailed feedback
            if decision_changes:
                st.success(f"✅ **Rerun completed!** {len(decision_changes)} decisions changed out of {pairs_to_rerun} total:")
                
                change_summary = []
                for pk, change in list(decision_changes.items())[:10]:  # Show first 10
                    change_summary.append(f"• **{pk}**: {change['old']} → **{change['new']}**")
                
                if len(decision_changes) > 10:
                    change_summary.append(f"• ... and {len(decision_changes)-10} more changes")
                
                st.markdown("\n".join(change_summary))
                
                # Store changes for later reference
                st.session_state.dedup_results['last_rerun_changes'] = decision_changes
                
            else:
                st.info(f"✅ **Rerun completed!** No decisions changed out of {pairs_to_rerun} total - your context was already well incorporated.")
            
            # Always show total rerun info
            total_rerun_pairs = len(rerun_entries) if 'rerun_entries' in locals() else 0
            st.info(f"📊 **Rerun Summary**: {total_rerun_pairs} total pairs processed, {len(decision_changes)} decisions changed. All rerun pairs now visible in 'Recently Resolved Pairs' section.")

        st.success(f"LLM rerun completed for scope: {scope} with decision changes context!")

    except Exception as e:
        st.error(f"Error during LLM rerun: {str(e)}")


def rerun_llm_with_comments(scope="all", focal_ids=None):
    """Rerun LLM with accumulated comments"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to rerun")
        return

    try:
        # Add comments to context book
        for comment_id, comment_text in st.session_state.review_comments.items():
            if comment_text.strip():
                if scope == "all":
                    st.session_state.context_book.add_global(comment_text)
                    persist_note("GLOBAL", comment_text)
                else:
                    # For focal-specific runs, add to specific focal
                    if focal_ids:
                        for focal_id in focal_ids:
                            st.session_state.context_book.add_focal(focal_id, comment_text)
                            persist_note("FOCAL", comment_text, focal_id)

        # Rerun LLM for the specified scope - but only for pairs that still need confirmation
        review_queue = st.session_state.dedup_results['review_queue']
        id2name = st.session_state.dedup_results['id2name']

        # Filter to only pairs that still need confirmation (not human-decided)
        if 'llm_decision' in review_queue.columns:
            pending_queue = review_queue[
                (review_queue['llm_decision'] == 'NEEDS_CONFIRMATION') &
                (~review_queue.get('human_decision', pd.Series([None] * len(review_queue))).notna())
            ].copy()
        else:
            pending_queue = review_queue.copy()

        if pending_queue.empty:
            st.info("No pairs remaining that need LLM rerun - all have been decided by human or previous runs.")
            return

        seed_focals = set(focal_ids) if focal_ids else set()

        with st.spinner(f"Rerunning LLM with scope: {scope} ({len(pending_queue)} pairs remaining)..."):
            judgments_by_pk = _rerun_llm_for_scope(
                pending_queue, id2name, st.session_state.context_book, scope, seed_focals
            )

            # Update review queue with new judgments AND create decision history
            for pk, judgment in judgments_by_pk.items():
                mask = review_queue['pair_key'] == pk
                if mask.any():
                    idx = review_queue[mask].index[0]
                    # Store original decision for lineage - ALWAYS track this
                    original_decision = review_queue.loc[idx, 'llm_decision'] if 'llm_decision' in review_queue.columns else 'NEEDS_CONFIRMATION'

                    # Ensure lineage columns exist
                    if 'rerun_from' not in review_queue.columns:
                        review_queue['rerun_from'] = None
                    if 'rerun_at' not in review_queue.columns:
                        review_queue['rerun_at'] = None

                    # Update with new judgment
                    new_decision = judgment.get('llm_decision')
                    review_queue.loc[idx, 'llm_decision'] = new_decision
                    review_queue.loc[idx, 'llm_confidence'] = judgment.get('llm_confidence')
                    review_queue.loc[idx, 'llm_reason'] = judgment.get('llm_reason')
                    review_queue.loc[idx, 'context_used'] = judgment.get('context_used')

                    # ALWAYS set lineage - even if decision didn't change
                    review_queue.loc[idx, 'rerun_from'] = original_decision
                    review_queue.loc[idx, 'rerun_at'] = utc_now()

                    # Log the rerun even if decision stayed the same
                    print(f"DEBUG: Rerun {pk}: {original_decision} → {new_decision} (context: {judgment.get('context_used', 'None')})")

            st.session_state.dedup_results['review_queue'] = review_queue

            # Update LLM decisions dataframe with ALL rerun judgments (not just resolved ones)
            llm_decisions_df = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())
            if not llm_decisions_df.empty and judgments_by_pk:
                # FIRST: Capture original state before any modifications
                original_decisions = {}
                for idx, row in llm_decisions_df.iterrows():
                    original_decisions[row['pair_key']] = row['llm_decision']
                
                # Ensure rerun tracking columns exist
                if 'rerun_from' not in llm_decisions_df.columns:
                    llm_decisions_df['rerun_from'] = None
                if 'rerun_at' not in llm_decisions_df.columns:
                    llm_decisions_df['rerun_at'] = None
                
                current_time = utc_now()
                
                # Update pairs that were directly rerun
                for pk, judgment in judgments_by_pk.items():
                    mask = llm_decisions_df['pair_key'] == pk
                    if mask.any():
                        idx = llm_decisions_df[mask].index[0]
                        
                        # Track original decision from BEFORE rerun
                        original_decision = original_decisions.get(pk, 'UNKNOWN')
                        
                        # Update with new judgment
                        llm_decisions_df.loc[idx, 'llm_decision'] = judgment.get('llm_decision')
                        llm_decisions_df.loc[idx, 'llm_confidence'] = judgment.get('llm_confidence')
                        llm_decisions_df.loc[idx, 'llm_reason'] = judgment.get('llm_reason')
                        llm_decisions_df.loc[idx, 'context_used'] = judgment.get('context_used')
                        
                        # Set rerun tracking with ORIGINAL decision
                        llm_decisions_df.loc[idx, 'rerun_from'] = original_decision
                        llm_decisions_df.loc[idx, 'rerun_at'] = current_time

                # For focal-specific reruns, also mark ALL pairs involving the focal accounts
                if scope == "foci" and seed_focals:
                    # Get all accounts that are connected to the seed focals through the rerun
                    all_affected_accounts = set(seed_focals)
                    
                    # Add accounts that were involved in any of the direct rerun pairs
                    for pk in judgments_by_pk.keys():
                        parts = pk.split('|')
                        if len(parts) == 2:
                            all_affected_accounts.add(parts[0])
                            all_affected_accounts.add(parts[1])
                    
                    # Mark ALL pairs between any of these affected accounts as rerun-affected
                    for idx, row in llm_decisions_df.iterrows():
                        pk = row['pair_key']
                        parts = pk.split('|')
                        if len(parts) == 2:
                            focal_id, candidate_id = parts[0], parts[1]
                            
                            # If this pair involves any of the affected accounts and doesn't already have rerun tracking
                            if (focal_id in all_affected_accounts or candidate_id in all_affected_accounts):
                                if pd.isna(row.get('rerun_at')) or row.get('rerun_at') != current_time:
                                    # Mark as affected by the focal rerun, using ORIGINAL decision
                                    original_decision = original_decisions.get(pk, 'UNKNOWN')
                                    llm_decisions_df.loc[idx, 'rerun_from'] = original_decision
                                    llm_decisions_df.loc[idx, 'rerun_at'] = current_time

            # After processing the rerun, check if any connected pairs are now effectively merged
            # and update their decisions accordingly
            if scope == "foci" and seed_focals and judgments_by_pk:
                # Apply the new decisions and see what groups are formed
                temp_decisions = llm_decisions_df.copy()
                
                # Create a mapping of current decisions
                decision_map = {}
                for _, row in temp_decisions.iterrows():
                    decision_map[row['pair_key']] = row['llm_decision']
                
                # Build groups based on current decisions
                from collections import defaultdict
                groups = defaultdict(set)
                
                # Add all YES decisions to groups
                for pk, decision in decision_map.items():
                    if decision == 'YES':
                        parts = pk.split('|')
                        if len(parts) == 2:
                            focal_id, candidate_id = parts[0], parts[1]
                            groups[focal_id].add(candidate_id)
                            groups[candidate_id].add(focal_id)
                
                # Find connected components (transitive closure)
                def find_all_connected(account_id, visited=None):
                    if visited is None:
                        visited = set()
                    if account_id in visited:
                        return visited
                    visited.add(account_id)
                    for connected in groups.get(account_id, set()):
                        find_all_connected(connected, visited)
                    return visited
                
                # Update decisions for pairs that are now transitively connected
                all_accounts = set()
                for pk in decision_map.keys():
                    parts = pk.split('|')
                    if len(parts) == 2:
                        all_accounts.add(parts[0])
                        all_accounts.add(parts[1])
                
                for account in all_accounts:
                    if account in groups:  # This account has direct connections
                        connected_group = find_all_connected(account)
                        
                        # For all pairs within this connected group, they should be YES
                        for acc1 in connected_group:
                            for acc2 in connected_group:
                                if acc1 < acc2:  # Avoid duplicates and self-pairs
                                    pk = f"{acc1}|{acc2}"
                                    if pk in decision_map and decision_map[pk] != 'YES':
                                        # This pair should now be YES due to transitivity
                                        mask = llm_decisions_df['pair_key'] == pk
                                        if mask.any():
                                            idx = llm_decisions_df[mask].index[0]
                                            # Only update if this was marked as rerun-affected
                                            if not pd.isna(llm_decisions_df.loc[idx, 'rerun_at']) and llm_decisions_df.loc[idx, 'rerun_at'] == current_time:
                                                llm_decisions_df.loc[idx, 'llm_decision'] = 'YES'
                                                llm_decisions_df.loc[idx, 'llm_reason'] = f"Transitively connected through focal rerun (via {', '.join(sorted(connected_group - {acc1, acc2}))})"

                st.session_state.dedup_results['llm_decisions'] = llm_decisions_df

            # Show info about resolved pairs for auto-apply
            changed_to_decided = [
                pk for pk, judgment in judgments_by_pk.items()
                if judgment.get('llm_decision') in ['YES', 'NO']
            ]

            if changed_to_decided:
                st.info(f"🔄 {len(changed_to_decided)} pairs resolved - auto-applying decisions...")

                # Auto-apply the decisions
                apply_decisions_and_finalize()

        # Show comprehensive summary
        total_rerun = len(judgments_by_pk)
        total_resolved = len(changed_to_decided)
        total_still_pending = total_rerun - total_resolved
        
        # Count total pairs with rerun tracking 
        final_llm_decisions = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())
        if not final_llm_decisions.empty and 'rerun_at' in final_llm_decisions.columns:
            total_with_rerun_tracking = len(final_llm_decisions[final_llm_decisions['rerun_at'].notna()])
        else:
            total_with_rerun_tracking = total_rerun
        
        st.success(f"LLM rerun completed for scope: {scope}")
        st.info(f"📊 **Rerun Summary**: {total_rerun} pairs directly processed, {total_resolved} resolved to definitive decisions, {total_still_pending} still need review.")
        
        if scope == "foci" and total_with_rerun_tracking > total_rerun:
            st.info(f"🔗 **Connected Impact**: {total_with_rerun_tracking} total pairs marked as affected (including connected pairs). All {total_with_rerun_tracking} pairs now visible in 'Recently Resolved Pairs' section with rerun indicators.")
        else:
            st.info(f"📋 All {total_with_rerun_tracking} rerun pairs now visible in 'Recently Resolved Pairs' section.")

    except Exception as e:
        st.error(f"Error during LLM rerun: {str(e)}")

def add_parent_to_singleton(singleton_id: str, parent_master: str):
    """Add a singleton account to an existing master group"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to modify")
        return
    
    # Update the final results if they exist
    if 'full_final' in st.session_state.dedup_results:
        full_df = st.session_state.dedup_results['full_final']
        
        # Move the singleton under the parent master
        mask = full_df['account_id'] == singleton_id
        if mask.any():
            full_df.loc[mask, 'master_account_id'] = parent_master
            full_df.loc[mask, 'is_master'] = False  # It's no longer a master
            
            # Recalculate group sizes
            full_df['group_size'] = full_df.groupby('master_account_id')['account_id'].transform('count')
            full_df['is_dupe'] = full_df['group_size'] > 1
            
            # Update session state
            st.session_state.dedup_results['full_final'] = full_df
            
            # Update masters list (singleton is no longer a master)
            masters_final = masters_slice(full_df)
            st.session_state.dedup_results['masters_final'] = masters_final
            
            # Track this change
            if 'manual_adjustments' not in st.session_state.dedup_results:
                st.session_state.dedup_results['manual_adjustments'] = []
            
            st.session_state.dedup_results['manual_adjustments'].append({
                'action': 'add_parent_to_singleton',
                'singleton_id': singleton_id,
                'new_parent': parent_master,
                'timestamp': utc_now()
            })


def remove_member_from_group(member_id: str, current_master: str):
    """Remove a member from its current group and make it standalone"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to modify")
        return False
    
    # Update the final results if they exist
    if 'full_final' in st.session_state.dedup_results:
        full_df = st.session_state.dedup_results['full_final']
        
        # Make the member its own master
        mask = full_df['account_id'] == member_id
        if mask.any():
            full_df.loc[mask, 'master_account_id'] = member_id
            full_df.loc[mask, 'is_master'] = True
            
            # Recalculate group sizes
            full_df['group_size'] = full_df.groupby('master_account_id')['account_id'].transform('count')
            full_df['is_dupe'] = full_df['group_size'] > 1
            
            # Update session state
            st.session_state.dedup_results['full_final'] = full_df
            
            # Update masters list
            masters_final = masters_slice(full_df)
            st.session_state.dedup_results['masters_final'] = masters_final
            
            # Add a mapping entry to track this change
            if 'manual_adjustments' not in st.session_state.dedup_results:
                st.session_state.dedup_results['manual_adjustments'] = []
            
            st.session_state.dedup_results['manual_adjustments'].append({
                'action': 'remove_member',
                'member_id': member_id,
                'old_master': current_master,
                'new_master': member_id,
                'timestamp': utc_now()
            })
            return True
        else:
            st.error(f"Member {member_id} not found in the data")
            return False
    else:
        st.error("No final results data available to modify")
        return False


def reparent_member_to_group(member_id: str, old_master: str, new_master: str):
    """Move a member from one group to another"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to modify")
        return False
    
    # Update the final results if they exist
    if 'full_final' in st.session_state.dedup_results:
        full_df = st.session_state.dedup_results['full_final']
        
        # Move the member to the new master
        mask = full_df['account_id'] == member_id
        if mask.any():
            full_df.loc[mask, 'master_account_id'] = new_master
            full_df.loc[mask, 'is_master'] = False  # It's not a master anymore
            
            # Recalculate group sizes for both groups
            full_df['group_size'] = full_df.groupby('master_account_id')['account_id'].transform('count')
            full_df['is_dupe'] = full_df['group_size'] > 1
            
            # Update session state
            st.session_state.dedup_results['full_final'] = full_df
            
            # Update masters list
            masters_final = masters_slice(full_df)
            st.session_state.dedup_results['masters_final'] = masters_final
            
            # Track this change
            if 'manual_adjustments' not in st.session_state.dedup_results:
                st.session_state.dedup_results['manual_adjustments'] = []
            
            st.session_state.dedup_results['manual_adjustments'].append({
                'action': 'reparent_member',
                'member_id': member_id,
                'old_master': old_master,
                'new_master': new_master,
                'timestamp': utc_now()
            })
            return True
        else:
            st.error(f"Member {member_id} not found in the data")
            return False
    else:
        st.error("No final results data available to modify")
        return False


def breakup_entire_group(master_id: str):
    """Break up an entire group - make all members standalone"""
    if 'dedup_results' not in st.session_state or not st.session_state.dedup_results:
        st.error("No deduplication results to modify")
        return
    
    # Update the final results if they exist
    if 'full_final' in st.session_state.dedup_results:
        full_df = st.session_state.dedup_results['full_final']
        
        # Get all members of this group
        group_mask = full_df['master_account_id'] == master_id
        group_members = full_df[group_mask]['account_id'].tolist()
        
        # Make each member its own master
        for member_id in group_members:
            member_mask = full_df['account_id'] == member_id
            full_df.loc[member_mask, 'master_account_id'] = member_id
            full_df.loc[member_mask, 'is_master'] = True
        
        # Recalculate group sizes
        full_df['group_size'] = full_df.groupby('master_account_id')['account_id'].transform('count')
        full_df['is_dupe'] = full_df['group_size'] > 1
        
        # Update session state
        st.session_state.dedup_results['full_final'] = full_df
        
        # Update masters list
        masters_final = masters_slice(full_df)
        st.session_state.dedup_results['masters_final'] = masters_final
        
        # Track this change
        if 'manual_adjustments' not in st.session_state.dedup_results:
            st.session_state.dedup_results['manual_adjustments'] = []
        
        st.session_state.dedup_results['manual_adjustments'].append({
            'action': 'breakup_group',
            'master_id': master_id,
            'affected_members': group_members,
            'timestamp': utc_now()
        })


def main():
    st.title("🔍 Account Deduplication Assistant")
    st.markdown("Interactive tool for deduplicating company accounts using AI")

    init_session_state()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "1. Input Management",
        "2. Deduplication Process",
        "3. Review & Results",
        "4. Masters Management"
    ])

    if page == "1. Input Management":
        st.header("📊 Account Input Management")

        st.subheader("Current Accounts")

        # Display editable dataframe
        edited_df = st.data_editor(
            st.session_state.accounts_df,
            num_rows="dynamic",
            width='stretch',
            key="accounts_editor"
        )

        # Update session state with edited data
        st.session_state.accounts_df = edited_df

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add New Account"):
                add_account_row()
                st.rerun()

        with col2:
            if st.button("🗑️ Remove Empty Rows"):
                # Preserve all columns structure but filter empty required fields
                required_cols = ['account_id', 'account_name']
                mask = True
                for col in required_cols:
                    if col in st.session_state.accounts_df.columns:
                        mask = mask & (st.session_state.accounts_df[col].str.strip() != '')
                
                st.session_state.accounts_df = st.session_state.accounts_df[mask]
                st.rerun()

        # Show summary
        valid_accounts = st.session_state.accounts_df[
            (st.session_state.accounts_df['account_id'].str.strip() != '') &
            (st.session_state.accounts_df['account_name'].str.strip() != '')
        ]
        st.info(f"Total valid accounts: {len(valid_accounts)}")

        # Metadata configuration section
        st.subheader("📊 Metadata Configuration")
        
        # Get available metadata columns (excluding required account_id and account_name)
        available_cols = [col for col in st.session_state.accounts_df.columns 
                         if col not in ['account_id', 'account_name']]
        
        if available_cols:
            st.write("**Configure which metadata columns to use:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Similarity Calculation (max 2 columns)**")
                st.caption("These columns will be included in embeddings and fuzzy matching")
                
                similarity_cols = st.multiselect(
                    "Select columns for similarity scoring:",
                    options=available_cols,
                    default=st.session_state.similarity_columns,
                    max_selections=2,
                    key="similarity_cols_select"
                )
                st.session_state.similarity_columns = similarity_cols
                
            with col2:
                st.write("**🤖 LLM Context (max 4 columns)**") 
                st.caption("These columns will be provided as context to the LLM for decisions")
                
                llm_context_cols = st.multiselect(
                    "Select columns for LLM context:",
                    options=available_cols,
                    default=st.session_state.llm_context_columns,
                    max_selections=4,
                    key="llm_context_cols_select"
                )
                st.session_state.llm_context_columns = llm_context_cols
                
            # Show current configuration
            st.info(f"✅ **Current Config:** Similarity: {similarity_cols or 'None'} | LLM Context: {llm_context_cols or 'None'}")
        else:
            st.info("💡 Add metadata columns to configure similarity scoring and LLM context options")

    elif page == "2. Deduplication Process":
        st.header("⚙️ Deduplication Process")

        # Workflow selection
        st.subheader("🔄 Choose Your Workflow")
        workflow_option = st.radio(
            "Select your preferred approach:",
            options=[
                "🎯 Direct Review → Fine-tune all decisions first, then apply",
                "⚡ Quick Run → Run process, then review only uncertain cases"
            ],
            help="Choose how you want to handle the deduplication process"
        )

        if "Direct Review" in workflow_option:
            st.info("""
            **🎯 Direct Review Workflow:**
            1. Run deduplication to get initial LLM decisions
            2. Review and edit ALL LLM decisions (not just uncertain ones)
            3. Add comments to guide LLM understanding
            4. Rerun LLM with your edits + comments as context
            5. Apply final decisions
            
            ✅ **Best for:** When you want to validate/adjust all decisions before applying
            """)
        else:
            st.info("""
            **⚡ Quick Run Workflow:**
            1. Run deduplication to get initial LLM decisions
            2. Apply auto decisions (HIGH confidence YES/NO)
            3. Review only uncertain cases in review queue
            4. Add comments for specific pairs as needed
            
            ✅ **Best for:** When you trust most LLM decisions and only want to review edge cases
            """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Run Deduplication", type="primary"):
                run_deduplication_process()

        with col2:
            if st.button("✅ Apply Decisions & Finalize", type="secondary", disabled=not st.session_state.dedup_results):
                apply_decisions_and_finalize()

        # Show results if available
        if st.session_state.dedup_results:
            st.subheader("📋 Process Results")

            # Show four main tables
            tab1, tab2, tab3, tab4 = st.tabs(["Original Input", "Candidates", "All LLM Decisions", "Review Queue"])

            with tab1:
                st.subheader("Original Input Accounts")
                st.dataframe(st.session_state.dedup_results['accounts_input'], width='stretch')

            with tab2:
                st.subheader("Similarity Candidates")
                candidates_df = st.session_state.dedup_results['candidates']
                # Sort by score descending
                candidates_display = candidates_df.sort_values('score', ascending=False)
                st.dataframe(candidates_display, width='stretch')

                # Show route distribution
                route_counts = candidates_df['route'].value_counts()
                st.bar_chart(route_counts)

            with tab3:
                st.subheader("All LLM Decisions")
                llm_decisions_df = st.session_state.dedup_results.get('llm_decisions', pd.DataFrame())

                if not llm_decisions_df.empty:
                    # Add decision filter
                    decision_filter = st.multiselect(
                        "Filter by LLM Decision:",
                        options=['YES', 'NO', 'NEEDS_CONFIRMATION'],
                        default=['YES', 'NO', 'NEEDS_CONFIRMATION']
                    )

                    filtered_decisions = llm_decisions_df[llm_decisions_df['llm_decision'].isin(decision_filter)]

                    # Sort by confidence descending, then by score
                    filtered_decisions = filtered_decisions.sort_values(['llm_confidence', 'score'], ascending=[False, False])

                    st.markdown("**📝 You can edit the LLM decisions below (click the pencil icon in cells):**")

                    # Make the decisions column editable
                    edited_decisions = st.data_editor(
                        filtered_decisions,
                        width='stretch',
                        column_config={
                            "llm_decision": st.column_config.SelectboxColumn(
                                "LLM Decision",
                                help="Edit the LLM decision",
                                width="small",
                                options=["YES", "NO", "NEEDS_CONFIRMATION"],
                                required=True,
                            ),
                            "llm_confidence": st.column_config.NumberColumn(
                                "LLM Confidence",
                                help="Edit confidence score",
                                min_value=0.0,
                                max_value=1.0,
                                step=0.01,
                                format="%.2f"
                            ),
                            "pair_key": st.column_config.Column("Pair Key", width="small"),
                            "focal_id": st.column_config.Column("Focal ID", width="small"),
                            "candidate_id": st.column_config.Column("Candidate ID", width="small"),
                            "score": st.column_config.NumberColumn("Similarity Score", format="%.3f"),
                        },
                        disabled=["pair_key", "focal_id", "focal_name", "candidate_id", "candidate_name", "score"],
                        key="llm_decisions_editor"
                    )

                    # Update session state with edited decisions
                    if not edited_decisions.equals(filtered_decisions):
                        # Update the full dataframe with changes
                        for idx, edited_row in edited_decisions.iterrows():
                            original_idx = llm_decisions_df[llm_decisions_df['pair_key'] == edited_row['pair_key']].index
                            if len(original_idx) > 0:
                                st.session_state.dedup_results['llm_decisions'].loc[original_idx[0], 'llm_decision'] = edited_row['llm_decision']
                                st.session_state.dedup_results['llm_decisions'].loc[original_idx[0], 'llm_confidence'] = edited_row['llm_confidence']

                    # Show decision distribution
                    decision_counts = llm_decisions_df['llm_decision'].value_counts()
                    st.bar_chart(decision_counts)

                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        yes_count = len(llm_decisions_df[llm_decisions_df['llm_decision'] == 'YES'])
                        st.metric("YES Decisions", yes_count)
                    with col2:
                        no_count = len(llm_decisions_df[llm_decisions_df['llm_decision'] == 'NO'])
                        st.metric("NO Decisions", no_count)
                    with col3:
                        needs_count = len(llm_decisions_df[llm_decisions_df['llm_decision'] == 'NEEDS_CONFIRMATION'])
                        st.metric("Needs Review", needs_count)

                    if st.button("🔄 Preview Changes", help="See what changes would be applied"):
                        # Detect changes using our new system
                        changed_pairs, changes_detail = detect_decision_changes()

                        if changed_pairs:
                            st.info(f"**🔍 {len(changed_pairs)} decision changes detected:**")

                            changes_list = []
                            for pk, detail in changes_detail.items():
                                changes_list.append(f"• {pk}: {detail['original']} → {detail['current']}")

                            st.markdown("\n".join(changes_list))

                            # Performance preview
                            current_decisions = st.session_state.dedup_results['llm_decisions']
                            affected_accounts = find_affected_accounts(changed_pairs, changes_detail, current_decisions)
                            total_masters = len(st.session_state.dedup_results['masters'])

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Changed Pairs", len(changed_pairs))
                            with col2:
                                st.metric("Affected Accounts", len(affected_accounts))
                            with col3:
                                efficiency = (1 - len(affected_accounts) / total_masters) * 100
                                st.metric("Efficiency Gain", f"{efficiency:.1f}%")

                            st.success(f"✨ Smart update will only recalculate {len(affected_accounts)} accounts instead of all {total_masters}!")
                            st.warning("⚠️ Click 'Apply Decisions & Finalize' to execute these changes and see final merged results!")
                        else:
                            st.success("No changes detected.")

                    # Comments section for LLM decision changes
                    st.subheader("💬 Add Comments for LLM Decision Changes")
                    st.caption("Add comments that will be used as context when rerunning the LLM for all pairs")
                    
                    global_llm_comment = st.text_area(
                        "Global Comment (applies to all LLM decision reruns):",
                        value=st.session_state.llm_decision_comments.get('global', ''),
                        key="global_llm_comment_input",
                        help="This comment will be added to context for all pairs when rerunning"
                    )
                    st.session_state.llm_decision_comments['global'] = global_llm_comment

                    # Rerun options with comments
                    st.subheader("🔄 Rerun LLM with Decision Changes & Comments")
                    st.caption("**Important:** This will rerun ALL LLM decisions (not just NEEDS_CONFIRMATION), using your comments and changes as context")
                    
                    # Scope selection
                    rerun_scope = st.radio(
                        "Select rerun scope:",
                        options=["all", "foci", "foci_related"],
                        index=0,
                        format_func=lambda x: {
                            "all": "🌐 ALL LLM pairs",
                            "foci": "🎯 Specific focals only", 
                            "foci_related": "🔗 Related focals (graph-connected)"
                        }[x],
                        help="Choose which pairs to rerun with your context"
                    )
                    
                    # Focal selection for non-all scopes
                    selected_focals = []
                    affected_pairs_preview = []
                    
                    if rerun_scope in ["foci", "foci_related"]:
                        if 'masters' in st.session_state.dedup_results:
                            masters = st.session_state.dedup_results['masters']
                            all_focals = masters['account_id'].tolist()
                            
                            selected_focals = st.multiselect(
                                "Select focal accounts:",
                                options=all_focals,
                                format_func=lambda x: f"{x} ({st.session_state.dedup_results.get('id2name', {}).get(x, 'Unknown')})",
                                help="Choose which focal accounts to use as seeds"
                            )
                            
                            # Show affected pairs preview when focals are selected
                            if selected_focals:
                                candidates = st.session_state.dedup_results['candidates']
                                llm_band = candidates[candidates["route"] == "LLM"]
                                
                                if rerun_scope == "foci":
                                    affected_pairs = llm_band[llm_band["master_a_id"].astype(str).isin(selected_focals)]
                                    scope_detail = f"pairs with focals: {', '.join(selected_focals)}"
                                elif rerun_scope == "foci_related":
                                    # Get graph-connected focals
                                    from example_run import _build_adj_from_pairs, _connected_component_nodes
                                    pairs_df = st.session_state.dedup_results['pairs']
                                    adj = _build_adj_from_pairs(pairs_df)
                                    related_focals = _connected_component_nodes(adj, selected_focals)
                                    affected_pairs = llm_band[llm_band["master_a_id"].astype(str).isin(related_focals)]
                                    scope_detail = f"{len(related_focals)} related focals (seeds: {', '.join(selected_focals)})"
                                
                                affected_pairs_preview = affected_pairs[['pair_key', 'master_a_id', 'master_b_id', 'score']].copy()
                                
                                if not affected_pairs_preview.empty:
                                    st.info(f"📋 **Preview: {len(affected_pairs_preview)} LLM pairs will be rerun** ({scope_detail})")
                                    
                                    # Show preview table with current decisions
                                    if 'llm_decisions' in st.session_state.dedup_results:
                                        llm_decisions = st.session_state.dedup_results['llm_decisions']
                                        preview_with_decisions = affected_pairs_preview.merge(
                                            llm_decisions[['pair_key', 'llm_decision', 'llm_confidence']], 
                                            on='pair_key', 
                                            how='left'
                                        )
                                        
                                        # Add account names for better readability
                                        id2name = st.session_state.dedup_results.get('id2name', {})
                                        preview_with_decisions['focal_name'] = preview_with_decisions['master_a_id'].map(id2name)
                                        preview_with_decisions['candidate_name'] = preview_with_decisions['master_b_id'].map(id2name)
                                        
                                        display_cols = ['pair_key', 'focal_name', 'candidate_name', 'llm_decision', 'llm_confidence', 'score']
                                        st.dataframe(preview_with_decisions[display_cols], width='stretch')
                                    else:
                                        st.dataframe(affected_pairs_preview, width='stretch')
                                else:
                                    st.warning(f"⚠️ No LLM pairs found for {scope_detail}")
                    
                    # Show total count for ALL scope
                    if rerun_scope == "all" and 'candidates' in st.session_state.dedup_results:
                        candidates = st.session_state.dedup_results['candidates']
                        llm_band = candidates[candidates["route"] == "LLM"]
                        st.info(f"📋 **Preview: {len(llm_band)} total LLM pairs will be rerun** (all pairs)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Dynamic button text based on scope
                        if rerun_scope == "all":
                            button_text = "🚀 Rerun ALL with Comments"
                            button_help = "Rerun LLM for ALL pairs using your decision changes and comments as context"
                        elif rerun_scope == "foci":
                            button_text = f"🎯 Rerun {len(selected_focals)} Focals"
                            button_help = f"Rerun LLM for pairs involving the {len(selected_focals)} selected focals"
                        else:  # foci_related
                            button_text = f"🔗 Rerun Related Focals"
                            button_help = f"Rerun LLM for all focals connected to the {len(selected_focals)} selected seeds"
                        
                        # Disable button if focals required but not selected
                        button_disabled = rerun_scope in ["foci", "foci_related"] and not selected_focals
                        
                        if st.button(button_text, type="primary", 
                                   help=button_help, disabled=button_disabled):
                            rerun_llm_with_decision_comments(scope=rerun_scope, focal_ids=selected_focals)
                            st.success("LLM rerun completed! Check results below.")
                            st.rerun()
                        
                        if button_disabled and rerun_scope in ["foci", "foci_related"]:
                            st.warning("⚠️ Please select at least one focal account")
                    
                    with col2:
                        if st.button("🎯 Apply Decisions Now", 
                                   help="Apply current decisions without LLM rerun"):
                            apply_decisions_and_finalize()
                            st.success("Decisions applied!")
                            st.rerun()

                else:
                    st.info("No LLM decisions available - only automatic decisions were made!")

            with tab4:
                st.subheader("LLM Review Queue (Needs Confirmation Only)")
                review_queue = st.session_state.dedup_results['review_queue']

                if not review_queue.empty:
                    st.dataframe(review_queue, width='stretch')
                else:
                    st.info("No items requiring human review - all decisions were automatic!")

        # Show final results if they exist
        if 'masters_final' in st.session_state.dedup_results:
            st.subheader("🎯 Final Results")

            tab_final1, tab_final2, tab_final3 = st.tabs(["Final Masters", "Applied Proposals", "Mapping Changes"])

            with tab_final1:
                st.subheader("Final Deduplicated Masters")
                masters_final = st.session_state.dedup_results['masters_final']
                st.dataframe(masters_final, width='stretch')

                # Summary of changes
                original_count = len(st.session_state.dedup_results['accounts_input'])
                final_count = len(masters_final)
                duplicates_removed = original_count - final_count

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Accounts", original_count)
                with col2:
                    st.metric("Final Masters", final_count)
                with col3:
                    st.metric("Duplicates Merged", duplicates_removed)

            with tab_final2:
                st.subheader("Applied Merge Proposals")
                apply_proposals = st.session_state.dedup_results.get('apply_proposals', pd.DataFrame())
                if not apply_proposals.empty:
                    st.dataframe(apply_proposals, width='stretch')
                else:
                    st.info("No merge proposals were applied")

            with tab_final3:
                st.subheader("Mapping Changes")
                clean_mapping = st.session_state.dedup_results.get('clean_mapping', pd.DataFrame())
                if not clean_mapping.empty:
                    st.dataframe(clean_mapping, width='stretch')
                    st.info("These accounts were merged into their canonical masters")
                else:
                    st.info("No new mappings were created")

    elif page == "3. Review & Results":
        st.header("👀 Review & Results")

        if not st.session_state.dedup_results:
            st.warning("Please run the deduplication process first")
            return

        review_queue = st.session_state.dedup_results.get('review_queue', pd.DataFrame())

        if review_queue.empty:
            st.success("No manual review needed - all decisions were automatic!")
        else:
            st.subheader("Manual Review Required")

            # Show comprehensive decision lineage summary
            st.subheader("📈 Decision History Summary")

            # Collect all decision changes
            lineage_summary = []

            # Human decisions
            human_pairs = review_queue[review_queue['human_decision_from'].notna()] if 'human_decision_from' in review_queue.columns else pd.DataFrame()
            if not human_pairs.empty:
                for _, row in human_pairs.iterrows():
                    original = row.get('human_decision_from', 'Unknown')
                    current = row.get('human_decision', 'Unknown')
                    pair_key = row['pair_key']
                    decision_time = row.get('human_decision_at', 'Unknown')
                    reviewer = row.get('reviewer', 'Unknown')

                    status_icon = "👤✅" if current == 'YES' else "👤❌" if current == 'NO' else "👤🔄"
                    lineage_summary.append(f"{status_icon} **{pair_key}**: {original} → **{current}** (human decision by {reviewer} at {decision_time})")

            # LLM rerun decisions - show ALL reruns, not just changes
            rerun_pairs = review_queue[review_queue['rerun_from'].notna()] if 'rerun_from' in review_queue.columns else pd.DataFrame()
            if not rerun_pairs.empty:
                for _, row in rerun_pairs.iterrows():
                    # Skip if this was a human decision
                    if 'human_decision' in row and pd.notna(row.get('human_decision')):
                        continue

                    original = row.get('rerun_from', 'Unknown')
                    current = row.get('llm_decision', 'Unknown')
                    pair_key = row['pair_key']
                    rerun_time = row.get('rerun_at', 'Unknown')
                    context_used = row.get('context_used', '')

                    # Show ALL reruns, indicate if decision changed or stayed same
                    if original != current:
                        status_icon = "🤖✅" if current == 'YES' else "🤖❌" if current == 'NO' else "🤖🔄"
                        change_indicator = "**CHANGED**"
                    else:
                        status_icon = "🤖🔄"
                        change_indicator = "**CONFIRMED**"

                    context_info = f" (with context)" if context_used and context_used.strip() else ""
                    lineage_summary.append(f"{status_icon} **{pair_key}**: {original} → **{current}** {change_indicator}{context_info} (LLM rerun at {rerun_time})")

            if lineage_summary:
                st.info("**Recent Decision Changes:**\n" + "\n".join(lineage_summary))
            else:
                st.info("No decision changes yet - all pairs are in their original state")

            # Debug section
            with st.expander("🔧 Debug Info (Click to expand)"):
                st.write("**Review Queue Columns:**", list(review_queue.columns) if not review_queue.empty else "Empty")
                if not review_queue.empty:
                    rerun_count = review_queue['rerun_from'].notna().sum() if 'rerun_from' in review_queue.columns else 0
                    human_count = review_queue['human_decision'].notna().sum() if 'human_decision' in review_queue.columns else 0
                    st.write(f"**Pairs with rerun history:** {rerun_count}")
                    st.write(f"**Pairs with human decisions:** {human_count}")
                    st.write(f"**Total pairs in queue:** {len(review_queue)}")

                    if 'rerun_from' in review_queue.columns:
                        # Only select columns that actually exist to avoid KeyError
                        possible_cols = ['pair_key', 'rerun_from', 'llm_decision', 'rerun_at', 'context_used']
                        existing_cols = [c for c in possible_cols if c in review_queue.columns]
                        rerun_details = review_queue[review_queue['rerun_from'].notna()][existing_cols]
                        if not rerun_details.empty:
                            st.write("**Rerun Details:**")
                            st.dataframe(rerun_details, width='stretch')

            # Add comments section
            st.subheader("💬 Add Comments for LLM Context")

            global_comment = st.text_area(
                "Global Comment (applies to all reruns):",
                value=st.session_state.review_comments.get('global', ''),
                key="global_comment_input"
            )
            st.session_state.review_comments['global'] = global_comment

            # Group pairs by current status for better organization
            if 'llm_decisions' in st.session_state.dedup_results:
                # Get the full LLM decisions dataframe which contains ALL processed pairs
                all_llm_decisions = st.session_state.dedup_results['llm_decisions']
                
                # Check for human decisions first
                if 'human_decision' in review_queue.columns:
                    queued_pairs = review_queue[
                        (review_queue['llm_decision'] == 'NEEDS_CONFIRMATION') &
                        (review_queue['human_decision'].isna())
                    ]
                    
                    # Resolved pairs should include:
                    # 1. Pairs with human decisions
                    # 2. All pairs with definitive LLM decisions (YES/NO) from any LLM run
                    # 3. All pairs that have been rerun (regardless of current decision)
                    resolved_pairs = all_llm_decisions[
                        (all_llm_decisions['llm_decision'].isin(['YES', 'NO'])) |
                        (all_llm_decisions['rerun_from'].notna())
                    ].copy()
                    
                    # Add human decision information from review_queue if available
                    if not review_queue.empty and 'human_decision' in review_queue.columns:
                        human_decided = review_queue[review_queue['human_decision'].notna()]
                        if not human_decided.empty:
                            # Merge human decisions into resolved pairs
                            resolved_pairs = resolved_pairs.merge(
                                human_decided[['pair_key', 'human_decision', 'reviewer', 'notes', 
                                              'human_decision_from', 'human_decision_at', 'rerun_from', 'rerun_at']],
                                on='pair_key', how='left'
                            )
                            
                            # Also include any pairs that only have human decisions (not in LLM decisions)
                            human_only = human_decided[~human_decided['pair_key'].isin(resolved_pairs['pair_key'])]
                            if not human_only.empty:
                                resolved_pairs = pd.concat([resolved_pairs, human_only], ignore_index=True)
                elif 'llm_decision' in review_queue.columns:
                    queued_pairs = review_queue[review_queue['llm_decision'] == 'NEEDS_CONFIRMATION']
                    # Include all pairs with definitive decisions from LLM OR that have been rerun
                    resolved_pairs = all_llm_decisions[
                        (all_llm_decisions['llm_decision'].isin(['YES', 'NO'])) |
                        (all_llm_decisions['rerun_from'].notna())
                    ].copy()
                else:
                    # No llm_decision column in review_queue, use all as queued
                    queued_pairs = review_queue.copy()
                    resolved_pairs = all_llm_decisions[
                        (all_llm_decisions['llm_decision'].isin(['YES', 'NO'])) |
                        (all_llm_decisions['rerun_from'].notna())
                    ].copy()
            else:
                # If no llm_decisions, fall back to review_queue only
                if not review_queue.empty and 'llm_decision' in review_queue.columns:
                    # Check for human decisions first
                    if 'human_decision' in review_queue.columns:
                        queued_pairs = review_queue[
                            (review_queue['llm_decision'] == 'NEEDS_CONFIRMATION') &
                            (review_queue['human_decision'].isna())
                        ]
                        resolved_pairs = review_queue[
                            (review_queue['llm_decision'].isin(['YES', 'NO'])) |
                            (review_queue['human_decision'].notna())
                        ]
                    else:
                        queued_pairs = review_queue[review_queue['llm_decision'] == 'NEEDS_CONFIRMATION']
                        resolved_pairs = review_queue[review_queue['llm_decision'].isin(['YES', 'NO'])]
                else:
                    # No llm_decision column available, treat all as queued with no resolved pairs
                    queued_pairs = review_queue.copy()
                    resolved_pairs = pd.DataFrame()

            if not queued_pairs.empty:
                st.subheader("🔍 Pairs Needing Review")
                for idx, row in queued_pairs.iterrows():
                    focal_id = str(row['focal_master_id'])
                    candidate_id = str(row['candidate_master_id'])
                    pair_key_str = row['pair_key']

                    # Show lineage if this pair was rerun
                    lineage_info = ""
                    if 'rerun_from' in row and pd.notna(row['rerun_from']):
                        lineage_info = f" (Originally: {row['rerun_from']})"

                    with st.expander(f"Review Pair: {focal_id} ↔ {candidate_id}{lineage_info}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Focal:** {focal_id}")
                            st.write(f"**Name:** {st.session_state.dedup_results['id2name'].get(focal_id, 'Unknown')}")
                            st.write(f"**Score:** {row.get('score', 'N/A')}")

                        with col2:
                            st.write(f"**Candidate:** {candidate_id}")
                            st.write(f"**Name:** {st.session_state.dedup_results['id2name'].get(candidate_id, 'Unknown')}")
                            st.write(f"**LLM Confidence:** {row.get('llm_confidence', 'N/A')}")

                        st.write(f"**LLM Reason:** {row.get('llm_reason', 'N/A')}")

                        # Show lineage details if available
                        if 'rerun_from' in row and pd.notna(row.get('rerun_from')):
                            current_decision = row.get('llm_decision', 'Unknown')
                            rerun_time = row.get('rerun_at', 'Unknown')
                            st.info(f"📋 **Decision History:** {row['rerun_from']} → {current_decision} (rerun at {rerun_time})")

                        # Individual comment for this pair
                        pair_comment = st.text_area(
                            f"Comment for this pair:",
                            value=st.session_state.review_comments.get(pair_key_str, ''),
                            key=f"comment_{pair_key_str}"
                        )
                        st.session_state.review_comments[pair_key_str] = pair_comment

                        # Accept/Deny buttons
                        st.subheader("👤 Human Decision")
                        col_accept, col_deny = st.columns(2)

                        with col_accept:
                            if st.button(f"✅ Accept Merge", key=f"accept_{pair_key_str}"):
                                handle_human_decision(pair_key_str, focal_id, candidate_id, "YES", pair_comment)
                                st.success(f"✅ Approved merge for {pair_key_str}")
                                st.rerun()

                        with col_deny:
                            if st.button(f"❌ Deny Merge", key=f"deny_{pair_key_str}"):
                                handle_human_decision(pair_key_str, focal_id, candidate_id, "NO", pair_comment)
                                st.success(f"❌ Denied merge for {pair_key_str}")
                                st.rerun()

            if not resolved_pairs.empty:
                st.subheader("✅ Recently Resolved Pairs")
                st.caption("Showing all pairs with definitive decisions: human decisions, direct LLM decisions, and LLM rerun decisions")

                # Show resolved pairs with decision source information
                display_columns = ['pair_key']
                
                # Use focal_id/candidate_id if available, otherwise fall back to focal_master_id/candidate_master_id
                if 'focal_id' in resolved_pairs.columns:
                    display_columns.extend(['focal_id', 'candidate_id'])
                    focal_col, candidate_col = 'focal_id', 'candidate_id'
                else:
                    display_columns.extend(['focal_master_id', 'candidate_master_id']) 
                    focal_col, candidate_col = 'focal_master_id', 'candidate_master_id'

                # Add columns that exist
                for col in ['llm_decision', 'human_decision', 'llm_confidence', 'rerun_from', 'rerun_at', 'human_decision_from', 'human_decision_at', 'reviewer']:
                    if col in resolved_pairs.columns:
                        display_columns.append(col)

                # Only use columns that actually exist
                available_columns = [col for col in display_columns if col in resolved_pairs.columns]
                resolved_display = resolved_pairs[available_columns].copy()

                # Add names for display
                if focal_col in resolved_display.columns:
                    resolved_display['focal_name'] = resolved_display[focal_col].map(st.session_state.dedup_results['id2name'])
                if candidate_col in resolved_display.columns:
                    resolved_display['candidate_name'] = resolved_display[candidate_col].map(st.session_state.dedup_results['id2name'])

                # Enhanced decision source column that shows reruns
                def get_decision_source(row):
                    has_human = pd.notna(row.get('human_decision'))
                    has_rerun = pd.notna(row.get('rerun_from')) or pd.notna(row.get('rerun_at'))
                    
                    if has_human:
                        source = f"👤 Human: {row.get('human_decision', 'N/A')}"
                        if has_rerun:
                            source += f" (after rerun)"
                    else:
                        llm_decision = row.get('llm_decision', 'N/A')
                        if has_rerun:
                            source = f"🔄 LLM Rerun: {llm_decision}"
                        else:
                            source = f"🤖 LLM: {llm_decision}"
                    return source

                resolved_display['decision_source'] = resolved_display.apply(get_decision_source, axis=1)

                # Enhanced lineage information
                def get_lineage(row):
                    has_human = pd.notna(row.get('human_decision'))
                    has_rerun = pd.notna(row.get('rerun_from'))
                    
                    final_decision = row.get('human_decision', row.get('llm_decision', 'Unknown'))
                    
                    if has_human and has_rerun:
                        # Both human decision and rerun
                        human_from = row.get('human_decision_from', 'Unknown')
                        rerun_from = row.get('rerun_from', 'Unknown')
                        if human_from != rerun_from:
                            return f"{rerun_from} → {human_from} → {final_decision}"
                        else:
                            return f"{human_from} → {final_decision} (human override)"
                    elif has_human:
                        # Human decision only
                        human_from = row.get('human_decision_from', 'Original')
                        return f"{human_from} → {final_decision} (human)"
                    elif has_rerun:
                        # Rerun only
                        rerun_from = row.get('rerun_from', 'Original')
                        return f"{rerun_from} → {final_decision} (rerun)"
                    else:
                        # Original decision
                        return f"Original → {final_decision}"

                resolved_display['lineage'] = resolved_display.apply(get_lineage, axis=1)

                # Reorder columns for better display
                final_columns = ['pair_key']
                if 'focal_name' in resolved_display.columns:
                    final_columns.append('focal_name')
                if 'candidate_name' in resolved_display.columns:
                    final_columns.append('candidate_name')
                final_columns.extend(['decision_source', 'lineage'])
                
                if 'llm_confidence' in resolved_display.columns:
                    final_columns.append('llm_confidence')
                if 'reviewer' in resolved_display.columns:
                    final_columns.append('reviewer')

                # Only use columns that actually exist
                available_final_columns = [col for col in final_columns if col in resolved_display.columns]
                st.dataframe(resolved_display[available_final_columns], width='stretch')

                # Show summary
                total_resolved = len(resolved_pairs)
                human_count = len(resolved_pairs[resolved_pairs.get('human_decision', pd.Series()).notna()]) if 'human_decision' in resolved_pairs.columns else 0
                rerun_count = len(resolved_pairs[resolved_pairs.get('rerun_from', pd.Series()).notna()]) if 'rerun_from' in resolved_pairs.columns else 0
                llm_count = total_resolved - human_count
                st.info(f"📊 Showing {total_resolved} resolved pairs: {human_count} human decisions, {llm_count} LLM decisions ({rerun_count} include reruns)")

            # Rerun options
            st.subheader("🔄 Rerun LLM with Comments")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Rerun All"):
                    rerun_llm_with_comments(scope="all")
                    st.success("Rerun completed! Results updated below.")
                    st.rerun()

            with col2:
                focal_options = review_queue['focal_master_id'].unique().tolist()
                selected_focals = st.multiselect("Select focals:", focal_options)
                if st.button("Rerun Selected Focals") and selected_focals:
                    rerun_llm_with_comments(scope="foci", focal_ids=selected_focals)
                    st.success(f"Rerun completed for {len(selected_focals)} focals! Results updated below.")
                    st.rerun()

            with col3:
                if st.button("Rerun Single Account") and selected_focals:
                    rerun_llm_with_comments(scope="foci", focal_ids=selected_focals[:1])
                    st.success(f"Rerun completed for {selected_focals[0]}! Results updated below.")
                    st.rerun()

    elif page == "4. Masters Management":
        st.header("👑 Masters Management")

        if not st.session_state.dedup_results:
            st.warning("Please run the deduplication process first")
            return

        # Add explanation of how group management works
        with st.expander("ℹ️ How Group Management Works", expanded=False):
            st.markdown("""
            **Real-Time Table Updates:**
            - All group management operations update the Final Masters table **in real-time**
            - Changes are immediately reflected in the table above without downloading any files
            - The table shows the current state after all manual adjustments
            
            **Available Operations:**
            - **Remove Member**: Take a member out of a group and make it standalone
            - **Reparent Member**: Move a member from one master group to another  
            - **Break Up Group**: Dissolve an entire group, making all members standalone
            - **Add Parent**: Move a singleton account under an existing master group
            
            **Important Notes:**
            - No CSV files are downloaded automatically - the table updates in-place
            - All changes are tracked in the Manual Adjustments History section
            - Use the "Final Results" tab to export final data when needed
            """)

        # Show summary of manual adjustments if any
        if 'manual_adjustments' in st.session_state.dedup_results and st.session_state.dedup_results['manual_adjustments']:
            adjustments = st.session_state.dedup_results['manual_adjustments']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                remove_count = sum(1 for adj in adjustments if adj['action'] == 'remove_member')
                st.metric("Members Removed", remove_count)
            with col2:
                reparent_count = sum(1 for adj in adjustments if adj['action'] == 'reparent_member')
                st.metric("Members Reparented", reparent_count)
            with col3:
                breakup_count = sum(1 for adj in adjustments if adj['action'] == 'breakup_group')
                st.metric("Groups Broken Up", breakup_count)

        # Show final masters if finalization has been run, otherwise show initial masters
        if 'masters_final' in st.session_state.dedup_results:
            masters_df = st.session_state.dedup_results['masters_final']
            
            # Check for manual adjustments to show last update time
            last_update = None
            total_adjustments = 0
            if 'manual_adjustments' in st.session_state.dedup_results and st.session_state.dedup_results['manual_adjustments']:
                adjustments = st.session_state.dedup_results['manual_adjustments']
                total_adjustments = len(adjustments)
                last_update = max(adj['timestamp'] for adj in adjustments)
            
            if total_adjustments > 0:
                st.subheader(f"Final Masters (After All Merges + {total_adjustments} Manual Adjustments)")
                st.info(f"✅ This table reflects all decisions and manual adjustments. Last updated: {last_update}")
            else:
                st.subheader("Final Masters (After All Merges)")
                st.info("✅ These are the final deduplicated masters after applying all decisions")
        else:
            masters_df = st.session_state.dedup_results.get('masters', pd.DataFrame())
            st.subheader("Initial Masters (Before Applying Decisions)")
            st.warning("⚠️ These are initial masters. Click 'Apply Decisions & Finalize' to see final results.")

        if not masters_df.empty:
            # Add visual indicator for real-time updates
            if 'masters_final' in st.session_state.dedup_results:
                st.success("🔄 This table updates in real-time with all group management changes")
            
            st.dataframe(masters_df, width='stretch')

            # Show summary of merges applied
            if 'masters_final' in st.session_state.dedup_results:
                original_masters = st.session_state.dedup_results.get('masters', pd.DataFrame())
                if not original_masters.empty:
                    original_count = len(original_masters)
                    final_count = len(masters_df)
                    merged_count = original_count - final_count

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Masters", original_count)
                    with col2:
                        st.metric("Final Masters", final_count)
                    with col3:
                        st.metric("Accounts Merged", merged_count, delta=-merged_count if merged_count > 0 else 0)

                    # Show what was merged
                    if 'clean_mapping' in st.session_state.dedup_results:
                        clean_mapping = st.session_state.dedup_results['clean_mapping']
                        if not clean_mapping.empty:
                            st.subheader("🔗 Applied Merges")

                            # Add account names to the mapping
                            full_accounts = st.session_state.dedup_results.get('accounts_input', pd.DataFrame())
                            if not full_accounts.empty:
                                id2name = dict(zip(full_accounts['account_id'], full_accounts['account_name']))

                                merge_display = clean_mapping.copy()
                                merge_display['old_account_name'] = merge_display['old_master_id'].map(id2name)
                                merge_display['canonical_account_name'] = merge_display['canonical_master_id'].map(id2name)

                                st.dataframe(merge_display, width='stretch')
                                st.info(f"📊 {len(clean_mapping)} merge operations were applied")
                            else:
                                st.dataframe(clean_mapping, width='stretch')

            # Show drill-down for each master
            st.subheader("🔍 Drill Down by Master")

            master_ids = masters_df['account_id'].tolist()
            selected_master = st.selectbox("Select master to drill down:", master_ids)

            if selected_master:
                # Show all pairs involving this master
                pairs_df = st.session_state.dedup_results.get('pairs', pd.DataFrame())
                master_pairs = pairs_df[
                    (pairs_df['master_a_id'] == selected_master) |
                    (pairs_df['master_b_id'] == selected_master)
                ]

                if not master_pairs.empty:
                    st.subheader(f"Pairs involving {selected_master}")
                    st.dataframe(master_pairs, width='stretch')

                # Show group members - use final data if available
                if 'full_final' in st.session_state.dedup_results:
                    full_df = st.session_state.dedup_results['full_final']
                    source_label = "Final Group Members (After All Merges)"
                else:
                    full_df = st.session_state.dedup_results.get('full_after_hist', pd.DataFrame())
                    source_label = "Initial Group Members"

                if not full_df.empty:
                    group_members = full_df[full_df['master_account_id'] == selected_master]

                    if len(group_members) > 1:
                        st.subheader(f"{source_label} for {selected_master}")

                        # Add group size information
                        group_size = len(group_members)
                        master_name = st.session_state.dedup_results['id2name'].get(selected_master, 'Unknown')

                        st.info(f"👥 Master: **{selected_master}** ({master_name}) has **{group_size} accounts** in this group")

                        # Enhanced display with management actions
                        display_members = group_members[['account_id', 'account_name', 'is_master']].copy()
                        display_members['role'] = display_members['is_master'].map({True: '👑 Master', False: '📎 Member'})

                        # Show the group members table
                        st.dataframe(display_members[['account_id', 'account_name', 'role']], width='stretch')

                        # Group management actions
                        st.subheader("🛠️ Group Management Actions")
                        
                        # Get member accounts (non-masters) for actions
                        member_accounts = group_members[~group_members['is_master']]['account_id'].tolist()
                        
                        if member_accounts:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**📤 Remove Member from Group**")
                                st.caption("Remove a member and make it a standalone account")
                                
                                remove_member = st.selectbox(
                                    "Select member to remove:",
                                    options=member_accounts,
                                    key=f"remove_member_{selected_master}",
                                    format_func=lambda x: f"{x} ({st.session_state.dedup_results['id2name'].get(x, 'Unknown')})"
                                )
                                
                                if st.button(f"🗑️ Remove {remove_member}", key=f"btn_remove_{selected_master}"):
                                    success = remove_member_from_group(remove_member, selected_master)
                                    if success:
                                        st.success(f"✅ Removed {remove_member} from group {selected_master} - Final table updated!")
                                        st.info("📊 The Final Masters table above has been updated with this change.")
                                        st.rerun()
                            
                            with col2:
                                st.write("**🔄 Reparent Member**")
                                st.caption("Move a member to a different master group")
                                
                                # Get all available masters as reparent options
                                all_masters = masters_df['account_id'].tolist()
                                other_masters = [m for m in all_masters if m != selected_master]
                                
                                if other_masters:
                                    reparent_member = st.selectbox(
                                        "Select member to reparent:",
                                        options=member_accounts,
                                        key=f"reparent_member_{selected_master}",
                                        format_func=lambda x: f"{x} ({st.session_state.dedup_results['id2name'].get(x, 'Unknown')})"
                                    )
                                    
                                    new_master = st.selectbox(
                                        "Select new master:",
                                        options=other_masters,
                                        key=f"new_master_{selected_master}",
                                        format_func=lambda x: f"{x} ({st.session_state.dedup_results['id2name'].get(x, 'Unknown')})"
                                    )
                                    
                                    if st.button(f"🔄 Move to {new_master}", key=f"btn_reparent_{selected_master}"):
                                        success = reparent_member_to_group(reparent_member, selected_master, new_master)
                                        if success:
                                            st.success(f"✅ Moved {reparent_member} from {selected_master} to {new_master} - Final table updated!")
                                            st.info("📊 The Final Masters table above has been updated with this change.")
                                            st.rerun()
                                else:
                                    st.info("No other masters available for reparenting")
                        else:
                            st.info("🏠 No members to manage - this master only contains itself")
                            
                        # Batch operations
                        if len(member_accounts) > 1:
                            st.write("**⚡ Batch Operations**")
                            
                            if st.button(f"💥 Break up entire group", key=f"btn_breakup_{selected_master}"):
                                breakup_entire_group(selected_master)
                                st.success(f"✅ Broke up group {selected_master} - all members are now standalone - Final table updated!")
                                st.info("📊 The Final Masters table above has been updated with this change.")
                                st.rerun()
                                
                    else:
                        st.info(f"🏠 {selected_master} is a singleton (no other accounts merged with it)")
                        
                        # Add parent functionality for singletons
                        st.subheader("👨‍👧‍👦 Add Parent for Singleton")
                        st.caption("Move this singleton account under an existing master group")
                        
                        # Get all other masters as potential parents
                        other_masters = [m for m in masters_df['account_id'].tolist() if m != selected_master]
                        
                        if other_masters:
                            selected_parent = st.selectbox(
                                "Select parent master:",
                                options=other_masters,
                                key=f"parent_select_{selected_master}",
                                format_func=lambda x: f"{x} ({st.session_state.dedup_results['id2name'].get(x, 'Unknown')})"
                            )
                            
                            if st.button(f"👨‍👧‍👦 Add {selected_master} to {selected_parent} group", 
                                       key=f"btn_add_parent_{selected_master}"):
                                add_parent_to_singleton(selected_master, selected_parent)
                                st.success(f"✅ Added {selected_master} to group {selected_parent} - Final table updated!")
                                st.info("📊 The Final Masters table above has been updated with this change.")
                                st.rerun()
                        else:
                            st.info("No other masters available to use as parents")

        else:
            st.warning("No masters data available")

        # Show manual adjustments history if any
        if 'manual_adjustments' in st.session_state.dedup_results and st.session_state.dedup_results['manual_adjustments']:
            st.subheader("📝 Manual Adjustments History")
            adjustments = st.session_state.dedup_results['manual_adjustments']
            
            # Create a readable table of adjustments
            adjustment_rows = []
            for adj in adjustments:
                if adj['action'] == 'remove_member':
                    description = f"Removed {adj['member_id']} from group {adj['old_master']} (made standalone)"
                elif adj['action'] == 'reparent_member':
                    description = f"Moved {adj['member_id']} from {adj['old_master']} to {adj['new_master']}"
                elif adj['action'] == 'breakup_group':
                    member_count = len(adj['affected_members'])
                    description = f"Broke up group {adj['master_id']} ({member_count} members made standalone)"
                else:
                    description = f"Unknown action: {adj['action']}"
                
                adjustment_rows.append({
                    'Timestamp': adj['timestamp'],
                    'Action': adj['action'].replace('_', ' ').title(),
                    'Description': description
                })
            
            adjustments_df = pd.DataFrame(adjustment_rows)
            st.dataframe(adjustments_df, width='stretch')
            
            # Option to export adjustments
            if st.button("📤 Export Adjustments"):
                csv = adjustments_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"manual_adjustments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()