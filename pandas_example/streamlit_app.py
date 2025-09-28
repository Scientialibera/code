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
    _rerun_llm_for_scope, LLM_ARRAY_SIZE, normalize_name, EMB_WEIGHT, FUZZ_WEIGHT
)
from rapidfuzz import fuzz

st.set_page_config(
    page_title="Account Deduplication Assistant",
    page_icon="🔍",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if 'accounts_df' not in st.session_state:
        # Start with demo data
        st.session_state.accounts_df = pd.DataFrame([
            ("a1", "ACME Corp"),
            ("a2", "ACME Corp"),
            ("a3", "Acme Corporation"),
            ("a4", "ACME Corp International"),
            ("f1", "Globex International Logistics"),
            ("b1", "Globex, LLC"),
            ("v1", "Unrelated Co"),
            ("x2", "Loblaws Ltd"),
            ("d1", "Microsoft"),
            ("d2", "Microsoft Corporation"),
            ("y3", "Microsoft International LLC")
        ], columns=["account_id", "account_name"])

    if 'dedup_results' not in st.session_state:
        st.session_state.dedup_results = {}

    if 'context_book' not in st.session_state:
        st.session_state.context_book = load_context_from_notes()

    if 'review_comments' not in st.session_state:
        st.session_state.review_comments = {}

def add_account_row():
    """Add a new empty row to accounts"""
    new_row = pd.DataFrame([("", "")], columns=["account_id", "account_name"])
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
                updated_pairs = incremental_recalculate(affected_accounts, original_masters, original_pairs)
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

        # Filter out empty rows
        accounts_df = accounts_df[
            (accounts_df['account_id'].str.strip() != '') &
            (accounts_df['account_name'].str.strip() != '')
        ]

        if accounts_df.empty:
            st.error("No valid accounts to process")
            return

        with st.spinner("Running deduplication process..."):
            # Step 1: Perfect match base
            full_base = perfect_match_base(accounts_df)

            # Step 2: Apply cumulative mapping
            cum_map_prev = read_csv(CUM_MAPPING_PATH, cols=["old_master_id", "canonical_master_id"])
            full_after_hist = apply_mapping(full_base, cum_map_prev)

            # Step 3: Masters & pairs
            masters = masters_slice(full_after_hist)
            pairs = similarity_pairs(masters)

            # Step 4: Route candidates
            candidates = route_candidates(pairs)

            # Step 5: LLM judgments
            context_book = st.session_state.context_book
            llm_res = llm_results_df(
                candidates, masters,
                array_batch_size=LLM_ARRAY_SIZE,
                context_book=context_book,
                scope="all",
                seed_focals=set(),
                seed_pairs=set(),
                pairs_for_graph=pairs
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

        # Rerun LLM for the specified scope
        review_queue = st.session_state.dedup_results['review_queue']
        id2name = st.session_state.dedup_results['id2name']

        seed_focals = set(focal_ids) if focal_ids else set()

        with st.spinner(f"Rerunning LLM with scope: {scope}..."):
            judgments_by_pk = _rerun_llm_for_scope(
                review_queue, id2name, st.session_state.context_book, scope, seed_focals
            )

            # Update review queue with new judgments
            for pk, judgment in judgments_by_pk.items():
                mask = review_queue['pair_key'] == pk
                if mask.any():
                    idx = review_queue[mask].index[0]
                    review_queue.loc[idx, 'llm_decision'] = judgment.get('llm_decision')
                    review_queue.loc[idx, 'llm_confidence'] = judgment.get('llm_confidence')
                    review_queue.loc[idx, 'llm_reason'] = judgment.get('llm_reason')
                    review_queue.loc[idx, 'context_used'] = judgment.get('context_used')

            st.session_state.dedup_results['review_queue'] = review_queue

        st.success(f"LLM rerun completed for scope: {scope}")

    except Exception as e:
        st.error(f"Error during LLM rerun: {str(e)}")

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
            use_container_width=True,
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
                st.session_state.accounts_df = st.session_state.accounts_df[
                    (st.session_state.accounts_df['account_id'].str.strip() != '') &
                    (st.session_state.accounts_df['account_name'].str.strip() != '')
                ]
                st.rerun()

        # Show summary
        valid_accounts = st.session_state.accounts_df[
            (st.session_state.accounts_df['account_id'].str.strip() != '') &
            (st.session_state.accounts_df['account_name'].str.strip() != '')
        ]
        st.info(f"Total valid accounts: {len(valid_accounts)}")

    elif page == "2. Deduplication Process":
        st.header("⚙️ Deduplication Process")

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
                st.dataframe(st.session_state.dedup_results['accounts_input'], use_container_width=True)

            with tab2:
                st.subheader("Similarity Candidates")
                candidates_df = st.session_state.dedup_results['candidates']
                # Sort by score descending
                candidates_display = candidates_df.sort_values('score', ascending=False)
                st.dataframe(candidates_display, use_container_width=True)

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
                        use_container_width=True,
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

                else:
                    st.info("No LLM decisions available - only automatic decisions were made!")

            with tab4:
                st.subheader("LLM Review Queue (Needs Confirmation Only)")
                review_queue = st.session_state.dedup_results['review_queue']

                if not review_queue.empty:
                    st.dataframe(review_queue, use_container_width=True)
                else:
                    st.info("No items requiring human review - all decisions were automatic!")

        # Show final results if they exist
        if 'masters_final' in st.session_state.dedup_results:
            st.subheader("🎯 Final Results")

            tab_final1, tab_final2, tab_final3 = st.tabs(["Final Masters", "Applied Proposals", "Mapping Changes"])

            with tab_final1:
                st.subheader("Final Deduplicated Masters")
                masters_final = st.session_state.dedup_results['masters_final']
                st.dataframe(masters_final, use_container_width=True)

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
                    st.dataframe(apply_proposals, use_container_width=True)
                else:
                    st.info("No merge proposals were applied")

            with tab_final3:
                st.subheader("Mapping Changes")
                clean_mapping = st.session_state.dedup_results.get('clean_mapping', pd.DataFrame())
                if not clean_mapping.empty:
                    st.dataframe(clean_mapping, use_container_width=True)
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

            # Add comments section
            st.subheader("💬 Add Comments for LLM Context")

            global_comment = st.text_area(
                "Global Comment (applies to all reruns):",
                value=st.session_state.review_comments.get('global', ''),
                key="global_comment_input"
            )
            st.session_state.review_comments['global'] = global_comment

            # Show each review item with individual comment boxes
            for idx, row in review_queue.iterrows():
                focal_id = str(row['focal_master_id'])
                candidate_id = str(row['candidate_master_id'])
                pair_key_str = row['pair_key']

                with st.expander(f"Review Pair: {focal_id} ↔ {candidate_id}"):
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

                    # Individual comment for this pair
                    pair_comment = st.text_area(
                        f"Comment for this pair:",
                        value=st.session_state.review_comments.get(pair_key_str, ''),
                        key=f"comment_{pair_key_str}"
                    )
                    st.session_state.review_comments[pair_key_str] = pair_comment

            # Rerun options
            st.subheader("🔄 Rerun LLM with Comments")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Rerun All"):
                    rerun_llm_with_comments(scope="all")
                    st.rerun()

            with col2:
                focal_options = review_queue['focal_master_id'].unique().tolist()
                selected_focals = st.multiselect("Select focals:", focal_options)
                if st.button("Rerun Selected Focals") and selected_focals:
                    rerun_llm_with_comments(scope="foci", focal_ids=selected_focals)
                    st.rerun()

            with col3:
                if st.button("Rerun Single Account") and selected_focals:
                    rerun_llm_with_comments(scope="foci", focal_ids=selected_focals[:1])
                    st.rerun()

    elif page == "4. Masters Management":
        st.header("👑 Masters Management")

        if not st.session_state.dedup_results:
            st.warning("Please run the deduplication process first")
            return

        masters_df = st.session_state.dedup_results.get('masters', pd.DataFrame())

        if not masters_df.empty:
            st.subheader("Current Masters")
            st.dataframe(masters_df, use_container_width=True)

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
                    st.dataframe(master_pairs, use_container_width=True)

                    # Show group members if any
                    full_df = st.session_state.dedup_results.get('full_after_hist', pd.DataFrame())
                    group_members = full_df[full_df['master_account_id'] == selected_master]

                    if len(group_members) > 1:
                        st.subheader(f"Group Members for {selected_master}")
                        st.dataframe(group_members[['account_id', 'account_name', 'is_master']], use_container_width=True)
        else:
            st.warning("No masters data available")

if __name__ == "__main__":
    main()