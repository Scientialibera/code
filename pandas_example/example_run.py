#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-cycle account deduper with Azure OpenAI (no fallbacks).

New in this version
-------------------
• Scope controls for LLM judging & re-runs:
  - --scope {all,foci,foci_related}
  - --focals a1,a3 (seed focals for foci/foci_related)
  - --seed_pairs a1|a2,a3|a6 (optional extra seeds for foci_related graph expansion)
• Rerun scope inside the interactive reviewer:
  - When you choose to rerun after adding notes, it will use --rerun_scope {all,foci,foci_related}
    with seeds defaulting to the current focal (plus any --focals/--seed_pairs provided).
• Prompt metadata end-to-end and explicit echo of the appended reviewer context via `context_used`.

Flow:
1) Perfect match → base table with master pointers
2) Apply cumulative mapping from previous runs (CDC)
3) Masters slice → compute embeddings → all pairwise similarities
4) Route (AUTO_95 / LLM / AUTO_NO)
5) LLM tool-calling to decide YES / NO / NEEDS_CONFIRMATION for LLM band
6) Interactive prompts ONLY for NEEDS_CONFIRMATION (+ optional rerun-with-notes)
   - Reviewer notes persist per focal and are applied to subsequent prompts
   - Rerun can rejudge ALL / FOCI / FOCI+RELATED queued pairs, based on --rerun_scope
   - Optional auto-approve pairs that flip to YES above a confidence threshold
7) (Optional --admin_review) Confirm/override ALL suggested merges (AUTO_95 + LLM_YES)
8) Build apply_proposals (AUTO_95 + LLM_YES + HUMAN_APPROVED + ADMIN_APPROVED) minus admin rejections
9) Clean into mapping with union-find, update ledgers, re-apply mapping
10) Write snapshots for this run and print summaries
"""
from __future__ import annotations

import os, re, json, math, argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable, Set

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv

# ----------------------------
# Config and paths
# ----------------------------
T_AUTO = float(os.getenv("T_AUTO", "0.95"))
T_LLM_LOW = float(os.getenv("T_LLM_LOW", "0.60"))
LLM_TOP_N = int(os.getenv("LLM_TOP_N", "3"))
LLM_ARRAY_SIZE = int(os.getenv("LLM_ARRAY_SIZE", str(LLM_TOP_N)))  # chunk size per LLM request

# Auto-approve behavior for reruns with notes
AUTO_APPROVE_RERUN = os.getenv("AUTO_APPROVE_RERUN_YES", "true").lower() == "true"
AUTO_APPROVE_RERUN_YES_CONF = float(os.getenv("AUTO_APPROVE_RERUN_YES_CONF", "0.9"))

ADMIN_REVIEW_DEFAULT = os.getenv("ADMIN_REVIEW", "false").lower() == "true"

# Combined similarity weights: embedding and fuzzy (0-1 each, sum should usually be 1)
EMB_WEIGHT = float(os.getenv("SIM_EMB_WEIGHT", "0.95"))
FUZZ_WEIGHT = float(os.getenv("SIM_FUZZ_WEIGHT", "0.05"))

ROOT = Path("./output/azure_cicd_cli").resolve()
ROOT.mkdir(parents=True, exist_ok=True)

DECISIONS_PATH = ROOT / "decisions_history.csv"              # pair_key, decision, source, decided_at, score?, reason?
CUM_MAPPING_PATH = ROOT / "clean_proposals_accumulated.csv"  # old_master_id, canonical_master_id
NOTES_PATH = ROOT / "reviewer_notes.csv"        # scope, focal_master_id, note, added_at
DEMO_ACCOUNTS_PATH = ROOT / "demo_accounts.csv" # persistent demo input across runs


# ----------------------------
# Utilities
# ----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_name(s: str) -> str:
    s2 = (s or "").lower().strip()
    s2 = re.sub(r"[^a-z0-9\s]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def cosine(a, b) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a) or 1.0
    nb = np.linalg.norm(b) or 1.0
    return float(np.dot(a, b) / (na * nb))

def pair_key(a, b) -> str:
    a, b = sorted([str(a), str(b)])
    return f"{a}|{b}"

def read_csv(path: Path, cols=None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=cols or [])

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def show(df: pd.DataFrame, title: str, n: int = 15):
    print(f"\n--- {title} (rows={len(df)}) ---")
    if df.empty:
        print("<empty>")
    else:
        print(df.head(n).to_string(index=False))

def get_pair_score(candidates: pd.DataFrame, m1: str, m2: str) -> float:
    pk = pair_key(m1, m2)
    row = candidates.loc[candidates["pair_key"]==pk]
    if not row.empty:
        try:
            return float(row["score"].iloc[0])
        except Exception:
            return 0.0
    return 0.0

def compress_mapping_df(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given rows (old_master_id -> canonical_master_id), collapse transitive chains so
    each old maps directly to the ultimate root. Returns 2-col df (old, root).
    """
    if mapping_df is None or mapping_df.empty:
        return pd.DataFrame(columns=["old_master_id","canonical_master_id"])
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
    # Build unions across all edges
    for _, r in mapping_df.iterrows():
        a = str(r["old_master_id"]); b = str(r["canonical_master_id"])
        union(a, b)
    # Emit direct-to-root for all nodes except the root itself
    rows = []
    for node in list(parent.keys()):
        root = find(node)
        if node != root:
            rows.append((node, root))
    out = pd.DataFrame(rows, columns=["old_master_id","canonical_master_id"]).drop_duplicates()
    return out

def _parse_add_accounts_arg(s: str) -> List[Tuple[str, str]]:
    out = []
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "|" not in p:
            raise ValueError(f"--add_accounts entry missing '|': {p}")
        aid, aname = p.split("|", 1)
        aid = aid.strip()
        aname = aname.strip()
        if not aid or not aname:
            raise ValueError(f"--add_accounts invalid entry: {p}")
        out.append((aid, aname))
    return out
# ----------------------------
# Graph helpers for scope = foci_related
# ----------------------------

def _build_adj_from_pairs(pairs_df: pd.DataFrame) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {}
    for _, r in pairs_df.iterrows():
        a, b = str(r["master_a_id"]), str(r["master_b_id"]) 
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _connected_component_nodes(adj: Dict[str, Set[str]], seeds: Iterable[str]) -> Set[str]:
    seeds = {str(s) for s in seeds if str(s)}
    if not seeds:
        return set()
    seen: Set[str] = set(seeds)
    stack: List[str] = list(seeds)
    while stack:
        v = stack.pop()
        for nb in adj.get(v, ()): 
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen

def load_context_from_notes() -> ContextBook:
    """Load persisted reviewer notes into a ContextBook (global + per-focal)."""
    cb = ContextBook()
    df = read_csv(NOTES_PATH, cols=["scope","focal_master_id","note","added_at"])
    if df.empty:
        return cb
    for _, r in df.iterrows():
        scope = (r.get("scope") or "").upper().strip()
        note  = (r.get("note") or "").strip()
        fid   = str(r.get("focal_master_id") or "").strip()
        if not note:
            continue
        if scope == "GLOBAL":
            cb.add_global(note)
        elif scope == "FOCAL" and fid:
            cb.add_focal(fid, note)
    return cb

def persist_note(scope: str, note: str, focal_id: str | None = None):
    """Append a reviewer note to the persistent CSV (de-duplicated)."""
    scope_up = "GLOBAL" if (scope or "").upper() == "GLOBAL" else "FOCAL"
    if not (note or "").strip():
        return
    row = {
        "scope": scope_up,
        "focal_master_id": str(focal_id) if scope_up == "FOCAL" else None,
        "note": note.strip(),
        "added_at": utc_now()
    }
    df = read_csv(NOTES_PATH, cols=["scope","focal_master_id","note","added_at"])
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # dedupe exact same (scope,focal,note)
    df = df.drop_duplicates(subset=["scope","focal_master_id","note"], keep="last")
    write_csv(df, NOTES_PATH)

# ----------------------------
# Reviewer context (persists for the run)
# ----------------------------
class ContextBook:
    """Holds global and per-focal notes to append to prompts."""
    def __init__(self):
        self.global_notes: List[str] = []
        self.focal_notes: Dict[str, List[str]] = {}

    def add_global(self, note: str):
        note = (note or "").strip()
        if note:
            self.global_notes.append(note)

    def add_focal(self, focal_id: str, note: str):
        note = (note or "").strip()
        if not note:
            return
        self.focal_notes.setdefault(str(focal_id), []).append(note)

    def render_context(self, focal_id: str | None = None) -> str:
        parts = []
        if self.global_notes:
            parts.append("Global guidance: " + " | ".join(self.global_notes))
        if focal_id and self.focal_notes.get(str(focal_id)):
            parts.append(f"Focal {focal_id} guidance: " + " | ".join(self.focal_notes[str(focal_id)]))
        return "\n".join(parts)


# ----------------------------
# Azure OpenAI (no fallback)
# ----------------------------
class AOAI:
    def __init__(self):
        load_dotenv()  # read .env
        self.endpoint = os.getenv("AOAI_ENDPOINT")
        self.chat_deploy = os.getenv("AOAI_CHAT_DEPLOYMENT")
        self.emb_deploy = os.getenv("AOAI_EMBEDDING_DEPLOYMENT")
        self.api_version = os.getenv("AOAI_API_VERSION", "2024-08-01-preview")

        if not (self.endpoint and self.chat_deploy and self.emb_deploy):
            raise RuntimeError("AOAI_* env vars missing. Please configure AOAI_ENDPOINT, AOAI_CHAT_DEPLOYMENT, AOAI_EMBEDDING_DEPLOYMENT.")

        from openai import AzureOpenAI
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        scope = "https://cognitiveservices.azure.com/.default"
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), scope)

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider
        )

    # Embeddings (batched)
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            out: List[List[float]] = []
            B = 64
            for i in range(0, len(texts), B):
                chunk = texts[i:i+B]
                resp = self.client.embeddings.create(model=self.emb_deploy, input=chunk)
                for d in resp.data:
                    out.append(d.embedding)
            return out
        except Exception as e:
            raise RuntimeError(f"Embedding error: {e}")

    # Tool-calling LLM
    def judge_matches(
        self,
        focal: Dict[str, str],
        candidates: List[Dict[str, Any]],
        extra_context: str | None = None,
        prompt_version: str = "v2_with_context"
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of:
        {candidate_master_id, score, llm_decision, llm_confidence, llm_reason, context_used}
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
                                    "score": {"type": "number", "description": "Cosine similarity score [0,1]"},
                                    "decision": {"type": "string", "enum": ["YES", "NO", "NEEDS_CONFIRMATION"]},
                                    "confidence": {"type": "number"}
                                },
                                "required": ["reason", "candidate_master_id", "decision", "score", "confidence"]
                            }
                        }
                    },
                    "required": ["decisions"]
                }
            }
        }]

        system = {
            "role": "system",
            "content": (
                "You are an account deduper. Decide if each candidate is the SAME real-world company as the focal. "
                "Normalize abbreviations and legal suffixes (LLC/L.L.C., Ltd/Limited, Intl/International, Log/Logistics, etc.). "
                "Subsidiaries/hold corps are NOT the same as parents. "
                "Return ONLY the function call with a decision for EACH candidate: YES, NO, or NEEDS_CONFIRMATION. "
                "Provide a short reason and a confidence 0.0–1.0."
                "If reviewer notes are provided, consider them carefully in your decisions and auto approve if certain - no need to reconfirm with user."
            )
        }

        context_text = f"\nSee below reviewer notes to consider in your matching: {extra_context.strip()}" if (extra_context and extra_context.strip()) else ""
        user = {
            "role": "user",
            "content": (
                f"Focal: {json.dumps(focal, ensure_ascii=False)}\n"
                f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
                f"{context_text}"
            )
        }

        resp = self.client.chat.completions.create(
            model=self.chat_deploy,
            messages=[system, user],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_match_judgments"}},
            temperature=0.1,
            top_p=0.1,
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            raise RuntimeError("Model did not return a tool call; ensure tool_choice is enforced and deployment is correct.")

        args = json.loads(tool_calls[0].function.arguments or "{}")
        decisions = args.get("decisions", [])
        out = []
        cand_map = {c["id"]: c for c in candidates}
        for d in decisions:
            cid = d.get("candidate_master_id")
            if not cid:
                continue
            sc = float(d.get("score", cand_map.get(cid, {}).get("score", 0.0)))
            dec = d.get("decision", "NEEDS_CONFIRMATION")
            conf = float(d.get("confidence", 0.6))
            rea = d.get("reason", "model_decision")
            out.append({
                "candidate_master_id": cid,
                "score": sc,
                "llm_decision": dec,
                "llm_confidence": conf,
                "llm_reason": rea,
                "prompt_version": prompt_version,
                "context_used": (extra_context or "")
            })
        return out


_AOAI_CLIENT: AOAI | None = None

def get_aoai_client() -> AOAI:
    global _AOAI_CLIENT
    if _AOAI_CLIENT is None:
        _AOAI_CLIENT = AOAI()
    return _AOAI_CLIENT


# ----------------------------
# Pipeline steps
# ----------------------------

def perfect_match_base(accounts_df: pd.DataFrame) -> pd.DataFrame:
    df = accounts_df.copy()
    df["normalized_name"] = df["account_name"].map(normalize_name)
    masters = df.groupby("normalized_name")["account_id"].min().rename("master_account_id")
    full = df.merge(masters, on="normalized_name", how="left")
    full["group_size"] = full.groupby("master_account_id")["account_id"].transform("count")
    full["is_master"] = full["account_id"] == full["master_account_id"]
    full["is_dupe"] = full["group_size"] > 1
    full["stage_rule"] = np.where(full["is_dupe"], "PERFECT", None)
    return full


def apply_mapping(full_base: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    if mapping_df is None or mapping_df.empty:
        return full_base.copy()
    mp = dict(zip(mapping_df["old_master_id"], mapping_df["canonical_master_id"]))
    out = full_base.copy()
    out["master_account_id"] = out["master_account_id"].map(lambda v: mp.get(str(v), str(v)))
    out["group_size"] = out.groupby("master_account_id")["account_id"].transform("count")
    out["is_master"] = out["account_id"] == out["master_account_id"]
    out["is_dupe"] = out["group_size"] > 1
    return out


def masters_slice(full_df: pd.DataFrame) -> pd.DataFrame:
    mo = full_df[full_df["is_master"]][["account_id","account_name","master_account_id","is_master","group_size"]].copy()
    mo["is_dupe"] = mo["group_size"] > 1
    return mo


def similarity_pairs(masters_only: pd.DataFrame) -> pd.DataFrame:
    mo = masters_only.copy()
    mo["normalized_name"] = mo["account_name"].map(normalize_name)
    try:
        client = get_aoai_client()
        vectors = client.embed_batch(mo["normalized_name"].tolist())
    except Exception:
        vectors = []
        for n in mo["normalized_name"].tolist():
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

            name_a = mo.loc[mo["account_id"]==a, "normalized_name"].iloc[0]
            name_b = mo.loc[mo["account_id"]==b, "normalized_name"].iloc[0]
            fuzz_score = float(fuzz.token_sort_ratio(name_a, name_b) / 100.0)

            combined = EMB_WEIGHT * emb_score + FUZZ_WEIGHT * fuzz_score
            rows.append((pair_key(a,b), a, b, emb_score, fuzz_score, combined))
    return pd.DataFrame(rows, columns=["pair_key","master_a_id","master_b_id","emb_score","fuzz_score","score"])


def route_candidates(pairs_df: pd.DataFrame) -> pd.DataFrame:
    cand = pairs_df.copy()
    def route(s):
        if s >= T_AUTO: return "AUTO_YES_95"
        if s >= T_LLM_LOW: return "LLM"
        return "AUTO_NO"
    cand["route"] = cand["score"].map(route)
    cand["proposed_keep_master"] = np.where(cand["route"]=="AUTO_YES_95", cand["master_a_id"], None)
    cand["proposed_merge_master"] = np.where(cand["route"]=="AUTO_YES_95", cand["master_b_id"], None)
    return cand


# --- scope filter utilities ---

def _parse_seed_focals(seed_focals: Iterable[str] | None) -> Set[str]:
    return {str(x).strip() for x in (seed_focals or []) if str(x).strip()}


def _parse_seed_pairs(seed_pairs: Iterable[str] | None) -> Set[str]:
    out: Set[str] = set()
    for p in (seed_pairs or []):
        p = str(p).strip()
        if not p:
            continue
        if "|" in p:
            a, b = p.split("|", 1)
            out.add(a.strip()); out.add(b.strip())
    return out


def filter_llm_band_by_scope(llm_band: pd.DataFrame, scope: str, seed_focals: Set[str], seed_pairs: Set[str], pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict LLM band to focals per scope: all | foci | foci_related."""
    scope = (scope or "all").lower()
    if scope == "all" or llm_band.empty:
        return llm_band

    seeds = set(seed_focals)
    seeds |= _parse_seed_pairs(seed_pairs)

    if scope == "foci":
        return llm_band[llm_band["master_a_id"].astype(str).isin(seeds)].copy()

    if scope == "foci_related":
        # Build graph from ALL candidate pairs (so relation can hop across focals)
        adj = _build_adj_from_pairs(pairs_df)
        comp_nodes = _connected_component_nodes(adj, seeds)
        return llm_band[llm_band["master_a_id"].astype(str).isin(comp_nodes)].copy()

    return llm_band


# --- batched LLM calls with array size control + optional context

def llm_results_df(
    candidates: pd.DataFrame,
    masters_only: pd.DataFrame,
    array_batch_size: int | None = None,
    context_book: ContextBook | None = None,
    scope: str = "all",
    seed_focals: Set[str] | None = None,
    seed_pairs: Set[str] | None = None,
    pairs_for_graph: pd.DataFrame | None = None,
) -> pd.DataFrame:
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

    id2name = dict(zip(masters_only["account_id"], masters_only["account_name"]))
    results = []
    client = get_aoai_client()

    for focal, grp in llm_band.groupby("master_a_id"):
        focal_payload = {"id": str(focal), "name": id2name.get(focal, "")}
        grp_sorted = grp.sort_values("score", ascending=False)
        # chunk into arrays of size array_batch_size
        chunked = [grp_sorted.iloc[i:i+array_batch_size] for i in range(0, len(grp_sorted), array_batch_size)]
        judgments_all: List[Dict[str, Any]] = []
        extra_ctx = context_book.render_context(str(focal)) if context_book else None
        for ch in chunked:
            candidates_payload = [
                {"id": str(r["master_b_id"]), "name": id2name.get(r["master_b_id"], ""), "score": float(r["score"])}
                for _, r in ch.iterrows()
            ]
            judgments = client.judge_matches(focal_payload, candidates_payload, extra_context=extra_ctx)
            judgments_all.extend(judgments)

        results.append({
            "focal_master_id": str(focal),
            "results": judgments_all,
            "model_name": os.getenv("AOAI_CHAT_DEPLOYMENT","gpt-4.1"),
            "prompt_version": f"v1_batch{array_batch_size}",
            "decided_at": utc_now()
        })
    return pd.DataFrame(results)


def build_review_queue(llm_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in llm_results.iterrows():
        focal = r["focal_master_id"]
        pv = r.get("prompt_version", None)
        mn = r.get("model_name", None)
        dt = r.get("decided_at", None)
        for it in r["results"]:
            if it["llm_decision"] == "NEEDS_CONFIRMATION":
                rows.append({
                    "pair_key": pair_key(focal, it["candidate_master_id"]),
                    "focal_master_id": focal,
                    "candidate_master_id": it["candidate_master_id"],
                    "score": it.get("score", None),
                    "llm_confidence": it.get("llm_confidence", None),
                    "llm_reason": it.get("llm_reason", None),
                    "context_used": it.get("context_used"),
                    "status": "QUEUED",
                    # stage breadcrumbs
                    "prompt_version": pv,
                    "model_name": mn,
                    "decided_at": dt,
                })
    return pd.DataFrame(rows)


def _rerun_llm_for_focal_batch(
    focal_id: str,
    focal_name: str,
    rows_for_focal: pd.DataFrame,
    id2name: Dict[str,str],
    context_book: ContextBook,
) -> Dict[str, Dict[str, Any]]:
    """
    Rejudge all candidate rows for a focal using accumulated context notes.
    Returns dict[candidate_id] -> judgment.
    """
    client = get_aoai_client()
    focal_payload = {"id": str(focal_id), "name": focal_name}
    candidates_payload = [{
        "id": str(r["candidate_master_id"]),
        "name": id2name.get(str(r["candidate_master_id"]), ""),
        "score": float(r["score"])
    } for _, r in rows_for_focal.iterrows()]
    extra_ctx = context_book.render_context(str(focal_id))
    judgments = client.judge_matches(focal=focal_payload, candidates=candidates_payload, extra_context=extra_ctx)
    return {j["candidate_master_id"]: j for j in judgments}


def _component_focals_from_queue(review_queue: pd.DataFrame, seed_focals: Set[str]) -> Set[str]:
    q = review_queue[review_queue["status"] == "QUEUED"]
    edges = q[["focal_master_id","candidate_master_id"]].astype(str).drop_duplicates()
    adj: Dict[str, Set[str]] = {}
    for _, r in edges.iterrows():
        a, b = r["focal_master_id"], r["candidate_master_id"]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    comp_nodes = _connected_component_nodes(adj, seed_focals)
    focals_in_queue = set(q["focal_master_id"].astype(str).unique())
    return comp_nodes & focals_in_queue


def _rerun_llm_for_scope(
    review_queue: pd.DataFrame,
    id2name: Dict[str, str],
    context_book: ContextBook,
    scope: str,
    seed_focals: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """Rejudge subset of queued pairs per scope; returns map pair_key -> judgment payload."""
    scope = (scope or "all").lower()
    queued = review_queue[review_queue["status"] == "QUEUED"].copy()
    if queued.empty:
        return {}

    # choose focals to process
    if scope == "all":
        focals_to_run = set(queued["focal_master_id"].astype(str))
    elif scope == "foci":
        focals_to_run = set(x for x in queued["focal_master_id"].astype(str) if x in seed_focals)
    elif scope == "foci_related":
        focals_to_run = _component_focals_from_queue(queued, seed_focals)
    else:
        focals_to_run = set(queued["focal_master_id"].astype(str))

    if not focals_to_run:
        return {}

    stamps = {
        "prompt_version": "v2_with_context",
        "model_name": os.getenv("AOAI_CHAT_DEPLOYMENT", "gpt-4.1"),
        "decided_at": utc_now(),
    }

    out: Dict[str, Dict[str, Any]] = {}
    for focal in sorted(focals_to_run):
        grp = queued[queued["focal_master_id"].astype(str) == focal][["pair_key","focal_master_id","candidate_master_id","score"]]
        judgments_map = _rerun_llm_for_focal_batch(
            focal_id=str(focal),
            focal_name=id2name.get(str(focal), ""),
            rows_for_focal=grp,
            id2name=id2name,
            context_book=context_book,
        )
        for _, r in grp.iterrows():
            cid = str(r["candidate_master_id"])
            pk  = str(r["pair_key"])
            j = judgments_map.get(cid)
            if not j:
                continue
            out[pk] = {
                "llm_decision":   j.get("llm_decision"),
                "llm_confidence": j.get("llm_confidence"),
                "llm_reason":     j.get("llm_reason"),
                "context_used":   j.get("context_used"),
                **stamps,
            }
    return out


def interactive_review(
    review_queue: pd.DataFrame,
    id2name: Dict[str,str],
    candidates_df: pd.DataFrame,
    context_book: ContextBook,
    llm_array_size: int | None = None,
    rerun_scope: str = "all",
    seed_focals: Set[str] | None = None,
):
    """
    NEEDS_CONFIRMATION only.
    - Allows adding notes (saved to context_book).
    - Rerun can rejudge ALL / FOCI / FOCI+RELATED queued pairs using --rerun_scope.
    - Auto-approve any pairs (except the current one) that flip to YES above a confidence threshold.
    Returns APPROVED subset.
    """
    seed_focals = _parse_seed_focals(seed_focals)

    cols = list(review_queue.columns)
    for extra in ["decision", "reviewer", "notes", "updated_at", "prompt_version", "model_name", "decided_at", "context_used"]:
        if extra not in cols:
            cols.append(extra)

    if review_queue.empty:
        return pd.DataFrame(columns=cols)

    if "llm_decision" not in review_queue.columns:
        review_queue["llm_decision"] = "NEEDS_CONFIRMATION"

    approved_rows = []
    auto_approved_keys: set[str] = set()

    # stable row-index lookup by pair_key (we only edit in place)
    idx_by_pk = {pk: i for i, pk in enumerate(review_queue["pair_key"].tolist())}

    print("\n>>> HUMAN REVIEW (only for LLM 'NEEDS_CONFIRMATION'):")
    for idx in review_queue.index.tolist():
        # re-check live status so auto-approved rows aren’t prompted later
        if str(review_queue.at[idx, "status"]) != "QUEUED":
            continue
        r = review_queue.loc[idx, :]  # fresh read after any in-loop updates
        focal = str(r["focal_master_id"])
        cand  = str(r["candidate_master_id"])
        pkcur = str(r["pair_key"])
        score = float(r["score"]) if r.get("score") is not None else float("nan")
        llm_conf   = r.get("llm_confidence", None)
        llm_reason = r.get("llm_reason", None)
        stage_pv   = r.get("prompt_version", "n/a")
        stage_m    = r.get("model_name", "n/a")
        stage_t    = r.get("decided_at", "n/a")

        if math.isnan(score):
            score_str = "nan"
        else:
            score_str = f"{score:.3f}"
        q = (
            f"\nApprove MERGE?\n"
            f"  Focal [{focal}]  {id2name.get(focal,'')}\n"
            f"  With  [{cand}]   {id2name.get(cand,'')}\n"
            f"  Similarity score: {score_str}\n"
            f"  LLM confidence: {llm_conf if llm_conf is not None else 'n/a'}\n"
            f"  LLM reason: {llm_reason if llm_reason is not None else 'n/a'}\n"
            f"  Stage: prompt_version={stage_pv}, model={stage_m}, decided_at={stage_t}\n"
            f"  Enter 'y' to approve, 'n' to reject, or 's' to skip: "
        )
        ans = input(q).strip().lower()

        notes = input("Add notes for the LLM/context (optional, press Enter to skip): ").strip()
        if notes:
            txt = notes[1:].strip() if notes.startswith("!") else notes.strip()
            # In-memory (current run)
            context_book.add_global(txt)
            context_book.add_focal(focal, txt)
            # Persist for future runs (auto-applied next time without any trigger)
            persist_note("GLOBAL", txt)
            persist_note("FOCAL",  txt, focal_id=focal)


        do_rerun = input(
            f"Rerun now with these notes? (scope={rerun_scope} — all/foci/foci_related) (y/n): "
        ).strip().lower() == "y"
        if do_rerun:
            # seeds: current focal + any external seeds provided
            seeds = set(seed_focals) | {focal}
            judgments_by_pk = _rerun_llm_for_scope(review_queue, id2name, context_book, rerun_scope, seeds)

            # Apply updates & auto-approve strong YES (except the current row, which awaits explicit answer)
            for pk2, j in judgments_by_pk.items():
                i2 = idx_by_pk.get(pk2)
                if i2 is None:
                    continue
                review_queue.at[i2, "context_used"]  = j.get("context_used")
                review_queue.at[i2, "llm_decision"]   = j.get("llm_decision")
                review_queue.at[i2, "llm_confidence"] = j.get("llm_confidence")
                review_queue.at[i2, "llm_reason"]     = j.get("llm_reason")
                review_queue.at[i2, "prompt_version"] = j.get("prompt_version")
                review_queue.at[i2, "model_name"]     = j.get("model_name")
                review_queue.at[i2, "decided_at"]     = j.get("decided_at")

                # Auto-approve if strong YES and not the current row
                if (
                    pk2 != pkcur
                    and AUTO_APPROVE_RERUN
                    and j.get("llm_decision") == "YES"
                    and (j.get("llm_confidence") is not None and float(j["llm_confidence"]) >= AUTO_APPROVE_RERUN_YES_CONF)
                ):
                    apr = review_queue.loc[i2, :].to_dict()
                    apr["status"]     = "APPROVED"
                    apr["decision"]   = "HUMAN_APPROVED_AFTER_NOTES"
                    apr["reviewer"]   = os.getenv("USERNAME") or os.getenv("USER") or "cli_user"
                    apr["notes"]      = (notes or None)
                    apr["updated_at"] = utc_now()
                    approved_rows.append(apr)
                    auto_approved_keys.add(pk2)
                    review_queue.at[i2, "status"] = "APPROVED"
                    print(f"→ Auto-approved after context: {pk2} (conf={j.get('llm_confidence')}, reason={j.get('llm_reason')})")

            j_self = judgments_by_pk.get(pkcur)
            if j_self:
                print(
                    f"→ LLM re-run decision (current pair): {j_self.get('llm_decision')} "
                    f"(conf={j_self.get('llm_confidence')}, reason={j_self.get('llm_reason')})"
                )

        # Explicit approval/rejection for the current pair
        if ans == "y":
            row = review_queue.loc[idx, :].to_dict()
            row["status"]     = "APPROVED"
            row["decision"]   = "HUMAN_APPROVED"
            row["reviewer"]   = os.getenv("USERNAME") or os.getenv("USER") or "cli_user"
            row["notes"]      = (notes or None)
            row["updated_at"] = utc_now()
            if "llm_confidence" not in row: row["llm_confidence"] = None
            if "llm_reason"     not in row: row["llm_reason"]     = None
            approved_rows.append(row)
            review_queue.at[idx, "status"] = "APPROVED"

    if not approved_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(approved_rows, columns=cols)


# --- Optional admin review of ALL suggestions (AUTO_95 + LLM_YES)

def admin_review_all(candidates: pd.DataFrame, llm_results: pd.DataFrame, id2name: Dict[str,str]) -> Tuple[pd.DataFrame, set]:
    to_review_rows = []

    # AUTO suggestions
    for _, r in candidates[candidates["route"]=="AUTO_YES_95"].iterrows():
        pk = r["pair_key"]
        m1, m2, score = r["master_a_id"], r["master_b_id"], float(r["score"])
        to_review_rows.append({
            "pair_key": pk,
            "focal_master_id": m1,
            "candidate_master_id": m2,
            "score": score,
            "suggested_source": "AUTO_95",
            "llm_confidence": None,
            "llm_reason": "auto_threshold"
        })

    # LLM YES suggestions
    if not llm_results.empty:
        for _, rr in llm_results.iterrows():
            focal = rr["focal_master_id"]
            for it in rr["results"]:
                if it.get("llm_decision") == "YES":
                    pk = pair_key(focal, it["candidate_master_id"])
                    to_review_rows.append({
                        "pair_key": pk,
                        "focal_master_id": focal,
                        "candidate_master_id": it["candidate_master_id"],
                        "score": float(it.get("score", 0.0)),
                        "suggested_source": "LLM_YES",
                        "llm_confidence": float(it.get("llm_confidence", 0.6)),
                        "llm_reason": it.get("llm_reason", "llm_decision")
                    })

    if not to_review_rows:
        return pd.DataFrame(columns=["pair_key","focal_master_id","candidate_master_id","score","llm_confidence","llm_reason","status","decision","reviewer","notes","updated_at"]), set()

    admin_approved = []
    admin_rejected: set = set()

    print("\n>>> ADMIN REVIEW (ALL suggested merges):")
    for row in to_review_rows:
        focal = str(row["focal_master_id"]); cand = str(row["candidate_master_id"])
        score = float(row["score"])
        src = row["suggested_source"]
        conf = row.get("llm_confidence", "n/a")
        reason = row.get("llm_reason", "n/a")

        prompt = (
            f"\nSuggested MERGE ({src}):\n"
            f"  Focal [{focal}]  {id2name.get(focal,'')}\n"
            f"  With  [{cand}]   {id2name.get(cand,'')}\n"
            f"  Score: {score:.3f}  LLM conf: {conf}  Reason: {reason}\n"
            f"  Accept? (y = approve, n = reject, s = skip): "
        )
        ans = input(prompt).strip().lower()
        notes = input("Optional admin notes (Enter to skip): ").strip()

        if ans == "y":
            admin_approved.append({
                "pair_key": row["pair_key"],
                "focal_master_id": focal,
                "candidate_master_id": cand,
                "score": score,
                "llm_confidence": None if src=="AUTO_95" else row.get("llm_confidence"),
                "llm_reason": reason,
                "status": "APPROVED",
                "decision": "ADMIN_APPROVED",
                "reviewer": os.getenv("USERNAME") or os.getenv("USER") or "cli_admin",
                "notes": notes if notes else None,
                "updated_at": utc_now()
            })
        elif ans == "n":
            admin_rejected.add(row["pair_key"])

    admin_approved_df = pd.DataFrame(admin_approved) if admin_approved else pd.DataFrame(columns=["pair_key","focal_master_id","candidate_master_id","score","llm_confidence","llm_reason","status","decision","reviewer","notes","updated_at"])
    return admin_approved_df, admin_rejected



def build_apply(
    candidates: pd.DataFrame,
    llm_results: pd.DataFrame,
    human_approved: pd.DataFrame,
    decisions_hist: pd.DataFrame,
    admin_approved: pd.DataFrame | None = None,
    admin_rejected: set | None = None
) -> pd.DataFrame:
    decided = set(decisions_hist["pair_key"].tolist()) if not decisions_hist.empty else set()
    admin_rejected = admin_rejected or set()
    rows = []

    # AUTO 95
    for _, r in candidates[candidates["route"]=="AUTO_YES_95"].iterrows():
        pk = r["pair_key"]
        if pk in decided or pk in admin_rejected: 
            continue
        rows.append({
            "pair_key": pk,
            "m1": r["master_a_id"],
            "m2": r["master_b_id"],
            "source": "AUTO_95",
            "score": float(r["score"]),
            "confidence": 1.0,
            "reason": "auto_threshold",
            # stage fields (deterministic auto)
            "prompt_version": "auto_threshold",
            "model_name": None,
            "decided_at": utc_now(),
        })

    # LLM YES
    for _, rr in llm_results.iterrows():
        focal = rr["focal_master_id"]
        pv = rr.get("prompt_version", None)
        mn = rr.get("model_name", None)
        dt = rr.get("decided_at", None)
        for it in rr["results"]:
            if it["llm_decision"] == "YES":
                pk = pair_key(focal, it["candidate_master_id"])
                if pk in decided or pk in admin_rejected: 
                    continue
                rows.append({
                    "pair_key": pk,
                    "m1": focal,
                    "m2": it["candidate_master_id"],
                    "source": "LLM_YES",
                    "score": float(it.get("score", 0.0)),
                    "confidence": float(it.get("llm_confidence", it.get("confidence", 0.6) or 0.6)),
                    "reason": it.get("llm_reason", it.get("reason", "llm_decision")),
                    # stage fields
                    "prompt_version": pv,
                    "model_name": mn,
                    "decided_at": dt,
                })

    # HUMAN APPROVED (NEEDS)
    if not human_approved.empty:
        for _, r in human_approved.iterrows():
            pk = r["pair_key"]
            if pk in decided or pk in admin_rejected:
                continue
            src = "HUMAN_NOTES_AUTO_APPROVED" if r.get("decision") == "HUMAN_APPROVED_AFTER_NOTES" else "HUMAN_APPROVED"
            rows.append({
                "pair_key": pk,
                "m1": r["focal_master_id"],
                "m2": r["candidate_master_id"],
                "source": src,
                "score": float(r["score"]),
                "confidence": float(r.get("llm_confidence") or 1.0),
                "reason": (r.get("llm_reason") or "human_approved"),
                "prompt_version": r.get("prompt_version") or ("v2_with_context" if src=="HUMAN_NOTES_AUTO_APPROVED" else "human_review"),
                "model_name": r.get("model_name"),
                "decided_at": r.get("updated_at") or utc_now(),
            })

    # ADMIN APPROVED
    if admin_approved is not None and not admin_approved.empty:
        for _, r in admin_approved.iterrows():
            pk = r["pair_key"]
            if pk in decided: 
                continue
            rows.append({
                "pair_key": pk,
                "m1": r["focal_master_id"],
                "m2": r["candidate_master_id"],
                "source": "ADMIN_APPROVED",
                "score": float(r["score"]),
                "confidence": float(r.get("llm_confidence") or 1.0),
                "reason": (r.get("llm_reason") or "admin_approved"),
                "prompt_version": "admin_review",
                "model_name": None,
                "decided_at": r.get("updated_at") or utc_now(),
            })

    return pd.DataFrame(rows)


def clean_proposals(apply_df: pd.DataFrame) -> pd.DataFrame:
    if apply_df is None or apply_df.empty:
        return pd.DataFrame(columns=["old_master_id","canonical_master_id"])
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
    for _, e in apply_df[["m1","m2"]].drop_duplicates().iterrows():
        union(str(e["m1"]), str(e["m2"]))
    rows = []
    for node in list(parent.keys()):
        root = find(node)
        if node != root:
            rows.append((node, root))
    return pd.DataFrame(rows, columns=["old_master_id","canonical_master_id"]).drop_duplicates()


# ----------------------------
# Main one-cycle runner
# ----------------------------

def run_cycle(
    accounts_df: pd.DataFrame,
    cycle_name: str | None = None,
    admin_review: bool = False,
    llm_array_size: int | None = None,
    scope: str = "all",
    seed_focals: Set[str] | None = None,
    seed_pairs: Set[str] | None = None,
    rerun_scope: str = "all",
):
    seed_focals = _parse_seed_focals(seed_focals)
    seed_pairs = _parse_seed_focals(seed_pairs)

    cycle_dir = ROOT / (cycle_name or datetime.now().strftime("cycle_%Y%m%d_%H%M%S"))
    cycle_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== START CYCLE:", cycle_dir.name, "=====")

    # 1) Base
    full_base = perfect_match_base(accounts_df)
    show(full_base, "full_base")

    # 2) Apply cumulative mapping from previous runs
    cum_map_prev = read_csv(CUM_MAPPING_PATH, cols=["old_master_id","canonical_master_id"])
    full_after_hist = apply_mapping(full_base, cum_map_prev)
    show(full_after_hist, "full_after_hist_mapping")

    # 3) Masters & pairs
    masters = masters_slice(full_after_hist)
    id2name = dict(zip(masters["account_id"].astype(str), masters["account_name"]))
    show(masters, "masters_slice (pre-merge deduped)")
    pairs = similarity_pairs(masters)
    show(pairs.sort_values("score", ascending=False), "pairs (top)")

    # 4) Route
    candidates = route_candidates(pairs)
    show(candidates.sort_values("score", ascending=False), "candidates (routed)")

    # 5) LLM judgments (batched + context; initially context empty)
    context_book = load_context_from_notes()
    llm_res = llm_results_df(
        candidates,
        masters,
        array_batch_size=llm_array_size or LLM_ARRAY_SIZE,
        context_book=context_book,
        scope=scope,
        seed_focals=seed_focals,
        seed_pairs=seed_pairs,
        pairs_for_graph=pairs,
    )
    show(pd.json_normalize(llm_res["results"].explode()) if not llm_res.empty else pd.DataFrame(), "llm_results (flattened)")

    # 6) Review queue (NEEDS only) + interactive approvals (with context propagation & multi-pair rerun)
    queue = build_review_queue(llm_res)
    show(queue, "review_queue (NEEDS)")
    human_approved = interactive_review(
        queue, id2name, candidates, context_book, llm_array_size or LLM_ARRAY_SIZE,
        rerun_scope=rerun_scope,
        seed_focals=seed_focals,
    )
    if not human_approved.empty:
        print("\nApproved by human (incl. any auto-approved after notes):\n", human_approved.to_string(index=False))

    # 6b) Optional ADMIN review of ALL suggested merges
    admin_approved = pd.DataFrame()
    admin_rejected: set = set()
    if admin_review:
        admin_approved, admin_rejected = admin_review_all(candidates, llm_res, id2name)
        if not admin_approved.empty:
            print("\nAdmin approved merges:\n", admin_approved.to_string(index=False))
        if admin_rejected:
            print("\nAdmin rejected pair_keys:", sorted(list(admin_rejected)))

    # 7) Apply proposals (respect decisions history + admin rejects)
    decisions_hist = read_csv(DECISIONS_PATH, cols=["pair_key","decision","source","decided_at","score","reason"])
    apply_df = build_apply(candidates, llm_res, human_approved, decisions_hist, admin_approved, admin_rejected)
    show(apply_df, "apply_proposals (AUTO_95 + LLM_YES + HUMAN/ADMIN_APPROVED)")

    # 8) Clean proposals → delta mapping; append to ledgers
    clean_delta = clean_proposals(apply_df)
    show(clean_delta, "clean_proposals_delta")

    # decisions ledger (YES merges) — append-only, never overwrite
    if not apply_df.empty:
        dec = apply_df[["pair_key","source","score","confidence","reason","prompt_version","model_name","decided_at"]].copy()
        dec["decision"] = "YES"
        decisions_hist2 = pd.concat([decisions_hist, dec], ignore_index=True, sort=False) \
                            .drop_duplicates(subset=["pair_key","decision"], keep="last")
    else:
        decisions_hist2 = decisions_hist.copy()

    # also log ADMIN rejects as NO (for audit) — append-only
    if admin_rejected:
        rows = []
        for pk in admin_rejected:
            try:
                m1, m2 = pk.split("|")
            except ValueError:
                m1 = m2 = None
            sc = get_pair_score(candidates, m1, m2) if (m1 and m2) else None
            rows.append({
                "pair_key": pk,
                "source": "ADMIN_REJECT",
                "score": sc,
                "reason": "admin_reject",
                "decision": "NO",
                "decided_at": utc_now(),
                "confidence": None,
                "prompt_version": "admin_review",
                "model_name": None,
            })
        rej_df = pd.DataFrame(rows)
        decisions_hist2 = pd.concat([decisions_hist2, rej_df], ignore_index=True, sort=False) \
                            .drop_duplicates(subset=["pair_key","decision"], keep="last")

    write_csv(decisions_hist2, DECISIONS_PATH)

    # 9) Update cumulative mapping & re-apply to produce final snapshots
    cum_map_now = read_csv(CUM_MAPPING_PATH, cols=["old_master_id","canonical_master_id"])
    if not clean_delta.empty:
        cum_concat = pd.concat([cum_map_now, clean_delta], ignore_index=True).drop_duplicates()
        cum_map_now = compress_mapping_df(cum_concat)
        write_csv(cum_map_now, CUM_MAPPING_PATH)
    else:
        # still compress in case prior file existed with transitive chains
        if not cum_map_now.empty:
            cum_map_now = compress_mapping_df(cum_map_now)
            write_csv(cum_map_now, CUM_MAPPING_PATH)

    full_post = apply_mapping(full_base, cum_map_now)
    masters_gold = masters_slice(full_post)
    show(full_post, "full_postapply (all rows remapped)")
    show(masters_gold, "masters_gold (FINAL deduped for this cycle)")

    # 10) Persist all artifacts for this cycle — append/emit only, never overwrite ledgers
    write_csv(accounts_df, cycle_dir/"accounts_input.csv")
    write_csv(full_base, cycle_dir/"full_base.csv")
    write_csv(full_after_hist, cycle_dir/"full_after_hist_mapping.csv")
    write_csv(masters, cycle_dir/"masters_slice.csv")
    write_csv(pairs, cycle_dir/"pairs.csv")
    write_csv(candidates, cycle_dir/"candidates.csv")
    (cycle_dir/"llm_results.json").write_text(json.dumps(llm_res.to_dict(orient="records"), indent=2))
    write_csv(queue, cycle_dir/"review_queue.csv")
    write_csv(human_approved, cycle_dir/"human_approved.csv")
    write_csv(admin_approved, cycle_dir/"admin_approved.csv")
    write_csv(apply_df, cycle_dir/"apply_proposals.csv")
    write_csv(clean_delta, cycle_dir/"clean_proposals_delta.csv")
    write_csv(cum_map_now, cycle_dir/"clean_proposals_cumulative.csv")
    write_csv(full_post, cycle_dir/"full_postapply.csv")
    write_csv(masters_gold, cycle_dir/"masters_gold.csv")

    print("\nArtifacts saved under:", cycle_dir.as_posix())
    print("Ledgers:")
    print(" -", CUM_MAPPING_PATH.as_posix())
    print(" -", DECISIONS_PATH.as_posix())
    print("\n===== END CYCLE =====\n")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Account deduper (Azure OpenAI, interactive + admin review) with scope controls.")
    parser.add_argument("--input_csv", type=str, default="",
                        help="CSV with columns: account_id,account_name. If omitted, a demo sample is used.")
    parser.add_argument("--cycle_name", type=str, default="",
                        help="Name for the output folder of this run.")
    parser.add_argument("--admin_review", action="store_true",
                        help="If set, after NEEDS review you will be asked to confirm/override ALL suggested merges (AUTO_95 + LLM_YES).")
    parser.add_argument("--llm_array_size", type=int, default=None,
                        help="How many candidates to send per LLM request for each focal (independent of LLM_TOP_N).")

    # Scope for initial LLM judging
    parser.add_argument("--scope", type=str, default=os.getenv("SCOPE", "all"), choices=["all","foci","foci_related"],
                        help="Restrict initial LLM judging to: all focals | only --focals | graph-connected focals.")
    parser.add_argument("--focals", type=str, default=os.getenv("FOCALS", ""),
                        help="Comma-separated focal IDs for scope=foci/foci_related (e.g., a1,a3).")
    parser.add_argument("--seed_pairs", type=str, default=os.getenv("SEED_PAIRS", ""),
                        help="Comma-separated pair_keys for graph seeding in scope=foci_related (e.g., a1|a2,a3|a6).")

    # Scope for reruns triggered during interactive review
    parser.add_argument("--rerun_scope", type=str, default=os.getenv("RERUN_SCOPE", "all"), choices=["all","foci","foci_related"],
                        help="When rerunning after notes: rejudge all queued, only seed focals, or seeds + graph-connected focals.")
    parser.add_argument("--demo_stateful", action="store_true",
                        help="Persist and reuse a demo dataset at ROOT/demo_accounts.csv across runs.")
    parser.add_argument("--add_accounts", type=str, default="",
                        help="Comma-separated id|name entries to append (demo_stateful only), e.g. 'a6|ACME Int Corp,d1|Microsoft Corporation'.")

    args = parser.parse_args()

    if args.input_csv:
        accounts_df = pd.read_csv(args.input_csv, dtype=str)
        if not {"account_id","account_name"}.issubset(accounts_df.columns):
            raise ValueError("input_csv must contain columns: account_id, account_name")
    else:
        # Demo mode
        if args.demo_stateful:
            # Create or reuse the persistent demo file
            if DEMO_ACCOUNTS_PATH.exists():
                accounts_df = pd.read_csv(DEMO_ACCOUNTS_PATH, dtype=str)
            else:
                accounts_df = pd.DataFrame([
                ("a1", "ACME Corp"),
                ("a2", "ACME Corp"),
                ("a3", "Acme Corporation"),
                ("f1", "Globex International Logistics"),
                ("b1", "Globex, LLC"),
                ("v1", "Unrelated Co"),
                ("x2", "Loblaws Ltd"),
                ("d1", "Microsoft"),
                ("d2", "Microsoft Corporation"),
                ], columns=["account_id","account_name"])
                write_csv(accounts_df, DEMO_ACCOUNTS_PATH)

            # Optionally append new rows from CLI
            new_rows = _parse_add_accounts_arg(args.add_accounts) if hasattr(args, "add_accounts") else []
            if new_rows:
                to_add = pd.DataFrame(new_rows, columns=["account_id","account_name"])
                accounts_df = pd.concat([accounts_df, to_add], ignore_index=True)
                # de-dupe by id (keep last occurrence so updates replace prior)
                accounts_df = accounts_df.drop_duplicates(subset=["account_id"], keep="last")
                write_csv(accounts_df, DEMO_ACCOUNTS_PATH)

        else:
            # Ephemeral one-shot demo (original behavior)
            accounts_df = pd.DataFrame([
                ("a1", "ACME Corp"),
                ("a2", "ACME Corp"),
                ("a3", "Acme Corporation"),
                ("f1", "Globex International Logistics"),
                ("b1", "Globex, LLC"),
                ("v1", "Unrelated Co"),
                ("x2", "Loblaws Ltd"),
                ("d1", "Microsoft"),
                ("d2", "Microsoft Corporation"),
            ], columns=["account_id","account_name"])

    # prefer CLI > env defaults
    admin_flag = args.admin_review or ADMIN_REVIEW_DEFAULT
    llm_arr = args.llm_array_size if args.llm_array_size is not None else LLM_ARRAY_SIZE

    seed_focals = [s.strip() for s in args.focals.split(",") if s.strip()] if args.focals else []
    seed_pairs  = [s.strip() for s in args.seed_pairs.split(",") if s.strip()] if args.seed_pairs else []

    run_cycle(
        accounts_df,
        cycle_name=args.cycle_name or None,
        admin_review=admin_flag,
        llm_array_size=llm_arr,
        scope=args.scope,
        seed_focals=set(seed_focals),
        seed_pairs=set(seed_pairs),
        rerun_scope=args.rerun_scope,
    )

if __name__ == "__main__":
    main()
