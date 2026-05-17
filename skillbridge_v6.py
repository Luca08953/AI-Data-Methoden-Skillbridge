#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillBridge v6 — Robert Half Benchmark Edition
HSG Capstone | AI Methoden | FS2026
"""

import streamlit as st
import pandas as pd
import os
import ast
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SkillBridge v6", page_icon="🎓", layout="wide")

BASE_PATH = "/Users/lucakamber/Library/Mobile Documents/com~apple~CloudDocs/HSG/MBI/FS2026/AI Methoden/09_Gruppenarbeit/Skill Bridge/Data"

DATASETS = {
    "Consulting & Strategy": "ConsultingData_clustered.csv",
    "Finance & Banking":     "Banking_clustered.csv",
    "Legal & Compliance":    "Legal_clustered.csv",
}

EXCLUDE_CLUSTERS = {
    "Retail & Service", "Retail Operations",
    "Physical Mobility & Material Handling", "Formal Qualifications & Degrees",
    "Teamwork", "Customer Service",
}

EXCLUDE_SKILL_KEYWORDS = [
    "degree", "diploma", "license", "lifting", "standing", "vaccination",
    "dental insurance", "commission", "gmc registration", "psychiatry",
    "clinical governance", "remote work", "follow", "teaching",
    "travel", "driving", "driver's license", "vehicle", "commute", "relocation",
    "english", "german", "language", "bilingual",
    "written", "verbal", "written and verbal", "passionate", "enthusiastic",
    "reliable", "punctual", "hard working", "fast-paced environment",
    "equal opportunity", "benefits", "salary", "bonus",
]

BASIC_SKILL_EXCLUDE_FROM_SALARY = {
    "microsoft office", "powerpoint", "excel", "outlook", "word",
    "microsoft excel", "microsoft word", "microsoft powerpoint",
    "communication", "teamwork", "organization", "multitasking",
    "time management", "customer service", "problem solving",
}

HARD_KEYWORDS = {
    "python", "sql", "java", "javascript", "excel", "tableau", "power bi",
    "aws", "cloud", "sap", "crm", "microsoft office", "powerpoint", "word",
    "data analysis", "data visualization", "data security", "machine learning",
    "software", "financial", "accounting", "budgeting", "forecasting",
    "auditing", "risk management", "risk assessment", "compliance",
    "investment", "wealth management", "lead generation", "account management",
    "legal", "regulatory", "contract", "governance", "project management",
    "business analysis", "business development", "change management",
    "supply chain", "procurement", "logistics", "marketing", "recruiting",
    "research", "reporting", "analytics", "cyber", "security", "information security",
    "threat", "saas", "container", "cissp", "azure", "inventory", "computer",
    "retail", "recruitment",
}

SOFT_KEYWORDS = {
    "communication", "leadership", "teamwork", "problem solving",
    "time management", "organizational", "interpersonal", "analytical",
    "attention to detail", "collaboration", "adaptability", "flexibility",
    "multitasking", "self-motivation", "selfmotivation", "negotiation",
    "presentation", "relationship", "mentoring", "training", "coaching",
    "customer service", "sales", "networking", "critical thinking",
    "decision making", "creativity", "innovation", "organization",
    "motivation", "initiative", "resilience", "empathy", "proactive",
}

# ── Robert Half Daten ──────────────────────────────────────────
RH_PROFILES = [
    "General Accountant General Ledger",
    "Accounts Payable Accounts Receivable Clerk",
    "Payroll Accountant",
    "Accounting Manager",
    "External Auditor",
    "Internal Auditor",
    "Business Analyst Manager",
    "Financial Planning Analysis Manager FPA",
    "Controller Manager Controlling",
    "Treasury Analyst Manager",
    "Credit Risk Manager",
    "Compliance Officer Manager",
    "Tax Specialist Manager",
    "Finance Manager",
    "Head of Finance Accounting",
    "Chief Financial Officer CFO",
    "Security Specialist",
    "IT Project Manager",
    "Business Data Analyst",
    "Data Manager Analyst",
    "Data Scientist",
    "Chief Security Officer CSO",
]

RH_SALARY_DATA = {
    "General Accountant General Ledger":          {"entry": 102000, "mid": 117250, "senior": 132500},
    "Accounts Payable Accounts Receivable Clerk":  {"entry":  71500, "mid":  86750, "senior":  96750},
    "Payroll Accountant":                          {"entry":  91750, "mid": 117250, "senior": 132500},
    "Accounting Manager":                          {"entry": 122500, "mid": 137750, "senior": 153000},
    "External Auditor":                            {"entry": 100750, "mid": 122250, "senior": 141500},
    "Internal Auditor":                            {"entry": 112000, "mid": 132500, "senior": 152750},
    "Business Analyst Manager":                    {"entry": 112500, "mid": 125000, "senior": 150000},
    "Financial Planning Analysis Manager FPA":     {"entry": 122500, "mid": 147750, "senior": 183000},
    "Controller Manager Controlling":              {"entry":  98750, "mid": 127250, "senior": 152750},
    "Treasury Analyst Manager":                    {"entry": 100750, "mid": 132500, "senior": 168000},
    "Credit Risk Manager":                         {"entry": 127250, "mid": 141500, "senior": 169750},
    "Compliance Officer Manager":                  {"entry": 114500, "mid": 132500, "senior": 168000},
    "Tax Specialist Manager":                      {"entry": 112000, "mid": 132500, "senior": 163000},
    "Finance Manager":                             {"entry": 127250, "mid": 147750, "senior": 168000},
    "Head of Finance Accounting":                  {"entry": 143000, "mid": 159000, "senior": 190750},
    "Chief Financial Officer CFO":                 {"entry": 183250, "mid": 234500, "senior": 254500},
    "Security Specialist":                         {"entry": 127500, "mid": 148250, "senior": 173000},
    "IT Project Manager":                          {"entry": 130500, "mid": 159750, "senior": 171000},
    "Business Data Analyst":                       {"entry": 102000, "mid": 122250, "senior": 145500},
    "Data Manager Analyst":                        {"entry":  90500, "mid": 110250, "senior": 118000},
    "Data Scientist":                              {"entry": 100000, "mid": 122000, "senior": 142500},
    "Chief Security Officer CSO":                  {"entry": 179750, "mid": 208250, "senior": 209750},
}

RH_GLOBAL_MEDIAN = sum(v["mid"] for v in RH_SALARY_DATA.values()) / len(RH_SALARY_DATA)


def classify_skill(skill):
    s = skill.lower()
    if any(h in s for h in HARD_KEYWORDS): return "hard"
    if any(soft in s for soft in SOFT_KEYWORDS): return "soft"
    return "hard"

def is_valid_skill(skill):
    return not any(kw in skill.lower() for kw in EXCLUDE_SKILL_KEYWORDS)

# ── Daten laden ────────────────────────────────────────────────
@st.cache_data
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    if "job_dimension" in df.columns and "job_cluster" not in df.columns:
        df = df.rename(columns={"job_dimension": "job_cluster"})
    def parse(entry):
        try: return ast.literal_eval(str(entry))
        except: return []
    df["skills_parsed"] = df["skills_filtered"].apply(parse)
    return df

@st.cache_data
def get_top_skills(filepath, top_n=100):
    df = load_dataset(filepath)
    counter = Counter()
    total = len(df)
    for skills in df["skills_parsed"]: counter.update(set(skills))
    return {
        skill: round(count / total * 100, 1)
        for skill, count in counter.most_common(top_n)
        if skill and len(skill) > 1 and is_valid_skill(skill)
    }

@st.cache_data
def get_cluster_profiles(filepath):
    df = load_dataset(filepath)
    df = df[~df["job_cluster"].isin(EXCLUDE_CLUSTERS)]
    clusters = {}
    for cluster in df["job_cluster"].unique():
        subset = df[df["job_cluster"] == cluster]
        counter = Counter()
        for skills in subset["skills_parsed"]: counter.update(skills)
        top = [s for s, _ in counter.most_common(10) if is_valid_skill(s)]
        clusters[cluster] = {
            "top_skills": top,
            "job_count":  len(subset),
            "pct":        round(len(subset) / len(df) * 100, 1),
        }
    return clusters

@st.cache_data
def train_random_forest(filepath, top_n=100):
    df = load_dataset(filepath)
    df = df[~df["job_cluster"].isin(EXCLUDE_CLUSTERS)].copy()
    counter = Counter()
    for skills in df["skills_parsed"]: counter.update(set(skills))
    feature_skills = [s for s, _ in counter.most_common(top_n) if is_valid_skill(s)]
    X = pd.DataFrame([
        {s: int(s in set(skills)) for s in feature_skills}
        for skills in df["skills_parsed"]
    ])
    le = LabelEncoder()
    y  = le.fit_transform(df["job_cluster"])
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.DataFrame({
        "skill":      feature_skills,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

SALARY_FILE = os.path.join(BASE_PATH, "salary_with_skills.csv")
SOURCE_MAP  = {"Consulting & Strategy": "Consulting", "Finance & Banking": "Banking", "Legal & Compliance": "Legal"}

@st.cache_data
def compute_salary_impact(primary_cat):
    import warnings
    try:
        df = pd.read_csv(SALARY_FILE)
        source = SOURCE_MAP.get(primary_cat, "Consulting")
        df = df[df["_source"] == source].copy()
        def parse(entry):
            try: return ast.literal_eval(str(entry))
            except: return []
        df["skills_parsed"] = df["skills_filtered"].apply(parse)
        results = []
        for cluster in df["job_dimension"].unique():
            cluster_df = df[df["job_dimension"] == cluster]
            if len(cluster_df) < 15: continue
            cluster_median = cluster_df["salary_mid"].median()
            counter = Counter()
            for skills in cluster_df["skills_parsed"]:
                counter.update([s for s in skills if is_valid_skill(s)])
            for skill, count in counter.most_common():
                if count < 10: continue
                if skill.lower() in BASIC_SKILL_EXCLUDE_FROM_SALARY: continue
                jobs_with = cluster_df[cluster_df["skills_parsed"].apply(lambda x: skill in x)]["salary_mid"]
                if len(jobs_with) < 10: continue
                if jobs_with.std() < 1: continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p_val = stats.ttest_1samp(jobs_with, cluster_median)
                diff = jobs_with.median() - cluster_median
                if p_val < 0.05 and abs(diff) > 5000:
                    results.append({"skill": skill, "cluster": cluster,
                                    "cluster_median": round(cluster_median),
                                    "salary_diff": round(diff), "p_val": round(p_val, 3),
                                    "n": count, "significant": True})
        if not results: return pd.DataFrame()
        return pd.DataFrame(results).sort_values("salary_diff", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def compute_cluster_salary(primary_cat):
    try:
        df = pd.read_csv(SALARY_FILE)
        source = SOURCE_MAP.get(primary_cat, "Consulting")
        df = df[df["_source"] == source].copy()
        result = df.groupby("job_dimension")["salary_mid"].agg(median_salary="median", n="count").reset_index()
        result.columns = ["cluster", "median_salary", "n"]
        return result.sort_values("median_salary", ascending=False)
    except:
        return pd.DataFrame()

@st.cache_data
def compute_skill_value_score(filepath, primary_cat):
    df = load_dataset(filepath)
    df = df[~df["job_cluster"].isin(EXCLUDE_CLUSTERS)].copy()
    try:
        sal = pd.read_csv(SALARY_FILE)
        source = SOURCE_MAP.get(primary_cat, "Consulting")
        sal_sub = sal[sal["_source"] == source]
        cluster_median_map = sal_sub.groupby("job_dimension")["salary_mid"].median()
        global_median = sal_sub["salary_mid"].median()
    except:
        cluster_median_map = {}; global_median = 100000.0
    df["salary_imputed"] = df["job_cluster"].map(cluster_median_map).fillna(global_median)
    sig_df = compute_salary_impact(primary_cat)
    significant_skills = set()
    if not sig_df.empty:
        significant_skills = set(sig_df[sig_df["salary_diff"] > 0]["skill"].unique())
    skill_counts = Counter(); skill_senior_counts = Counter(); skill_salary_sum = {}
    for _, row in df.iterrows():
        level = str(row.get("job_level", "")).lower()
        is_senior = any(kw in level for kw in ["senior", "director", "executive", "lead", "manager"])
        proxy_sal = row.get("salary_imputed", global_median)
        for s in row.get("skills_parsed", []):
            if not is_valid_skill(s): continue
            skill_counts[s] += 1
            if is_senior: skill_senior_counts[s] += 1
            skill_salary_sum.setdefault(s, [])
            if proxy_sal > 0: skill_salary_sum[s].append(proxy_sal)
    max_count = max(skill_counts.values()) if skill_counts else 1
    rows = []
    for skill, count in skill_counts.items():
        if count < 5: continue
        demand_score = (count / max_count) * 100
        senior_ratio = skill_senior_counts.get(skill, 0) / count
        career_score = senior_ratio * 100
        salaries = skill_salary_sum.get(skill, [])
        avg_sal = sum(salaries) / len(salaries) if salaries else global_median
        rows.append({"skill": skill, "demand_score": demand_score,
                     "career_score": career_score, "avg_salary": avg_sal, "count": count})
    res_df = pd.DataFrame(rows)
    if res_df.empty: return res_df
    min_s, max_s = res_df["avg_salary"].min(), res_df["avg_salary"].max()
    res_df["salary_score"] = (res_df["avg_salary"] - min_s) / (max_s - min_s) * 100 if max_s > min_s else 50.0
    final_scores = []
    for _, row in res_df.iterrows():
        skill = row["skill"]
        base = row["demand_score"] * 0.30 + row["career_score"] * 0.40 + row["salary_score"] * 0.30
        skill_type = classify_skill(skill)
        if skill.lower() in BASIC_SKILL_EXCLUDE_FROM_SALARY: base *= 0.2; modifier = "basic_penalty"
        elif skill_type == "soft": base *= 0.4; modifier = "soft_penalty"
        elif skill_type == "hard" and skill in significant_skills: base *= 1.25; modifier = "sig_boost"
        else: modifier = "none"
        final_scores.append({"score": round(min(base, 100), 1), "modifier": modifier,
                              "demand_pct": round(row["demand_score"], 1),
                              "career_ratio": round(row["career_score"] / 100, 2),
                              "salary_ratio": round(row["avg_salary"] / global_median, 2)})
    scores_df = pd.DataFrame(final_scores)
    for col in ["score", "modifier", "demand_pct", "career_ratio", "salary_ratio"]:
        res_df[col] = scores_df[col].values
    return res_df.sort_values("score", ascending=False).reset_index(drop=True)

@st.cache_data
def get_skill_primary_dimension(filepath):
    """Cached: skill → (primary_dimension, count)."""
    df = load_dataset(filepath)
    df = df[~df["job_cluster"].isin(EXCLUDE_CLUSTERS)]
    skill_dim_freq = {}
    for dim in df["job_cluster"].unique():
        counter = Counter()
        for skills in df[df["job_cluster"] == dim]["skills_parsed"]: counter.update(skills)
        for skill, count in counter.items():
            if skill not in skill_dim_freq or count > skill_dim_freq[skill][1]:
                skill_dim_freq[skill] = (dim, count)
    return skill_dim_freq

# ── Schritt 6: RH-Benchmark-Berechnungen ──────────────────────
@st.cache_data
def compute_rh_similarity(filepath):
    """Top-3 RH matches per dimension (used for dimension detail tab and MWS)."""
    df = load_dataset(filepath)
    dimensions = [d for d in df["job_cluster"].unique() if d not in EXCLUDE_CLUSTERS]
    dim_texts = {dim: " ".join(df[df["job_cluster"] == dim]["job_title"].dropna().tolist())
                 for dim in dimensions}
    if not dim_texts: return {}
    all_texts = list(dim_texts.values()) + RH_PROFILES
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(all_texts)
    n = len(dim_texts)
    sim_matrix = cosine_similarity(tfidf[:n], tfidf[n:])
    results = {}
    for i, dim_name in enumerate(dim_texts.keys()):
        sims = sim_matrix[i]
        top_idx = sims.argsort()[-3:][::-1]
        results[dim_name] = [
            {"profile": RH_PROFILES[j], "similarity": round(float(sims[j]), 3),
             "salary": RH_SALARY_DATA[RH_PROFILES[j]]}
            for j in top_idx if float(sims[j]) >= 0.05
        ]
    return results


@st.cache_data
def compute_full_profile_similarities(filepath):
    """Full similarity matrix: ALL 22 RH profiles × ALL dimensions.
    Needed so that profiles not in the top-3 per dimension still get requirements."""
    df = load_dataset(filepath)
    dimensions = [d for d in df["job_cluster"].unique() if d not in EXCLUDE_CLUSTERS]
    dim_texts = {dim: " ".join(df[df["job_cluster"] == dim]["job_title"].dropna().tolist())
                 for dim in dimensions}
    if not dim_texts: return {}
    all_texts = list(dim_texts.values()) + RH_PROFILES
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(all_texts)
    n = len(dim_texts)
    sim_matrix = cosine_similarity(tfidf[:n], tfidf[n:])
    # {profile: {dim: similarity}}
    result = {}
    for j, profile in enumerate(RH_PROFILES):
        result[profile] = {
            dim: round(float(sim_matrix[i, j]), 4)
            for i, dim in enumerate(dim_texts.keys())
        }
    return result


def compute_best_fit_roles(rh_sim_map, cluster_coverage_lookup, cluster_gaps):
    """Ranks RH job profiles by how ready the user is (weighted coverage × similarity)."""
    profile_data = {}
    for dim, matches in rh_sim_map.items():
        if not matches: continue
        top = matches[0]; profile = top["profile"]; sim = top["similarity"]
        cov = cluster_coverage_lookup.get(dim, 0)
        if profile not in profile_data:
            profile_data[profile] = {"salary": top["salary"], "dims": []}
        profile_data[profile]["dims"].append((dim, cov, sim))
    result = []
    for profile, data in profile_data.items():
        total_sim = sum(s for _, _, s in data["dims"])
        if total_sim == 0: continue
        weighted_cov = sum(c * s for _, c, s in data["dims"]) / total_sim
        # For each matching dimension, get the user's owned skills
        dims_detail = []
        for dim, cov, sim in sorted(data["dims"], key=lambda x: x[2], reverse=True)[:2]:
            gap = next((g for g in cluster_gaps if g["cluster"] == dim), None)
            has_skills  = [s.title() for s in (gap["has"]    if gap else [])[:4]]
            miss_skills = [s.title() for s in (gap["missing"] if gap else [])[:2]]
            dims_detail.append({"dim": dim, "cov": cov, "sim": sim,
                                 "has": has_skills, "missing": miss_skills})
        result.append({"profile": profile, "fit_pct": round(weighted_cov),
                       "salary": data["salary"], "dims_detail": dims_detail})
    return sorted(result, key=lambda x: x["fit_pct"], reverse=True)


def compute_global_markt_wert_score(skill_score_df, rh_sim_map, skill_dim_map, importance_df):
    """
    Markt-Wert-Score: gleiche Formel wie Skill-Value-Score,
    aber Gehaltskomponente = Robert Half CH-Median (statt LinkedIn-Gehaltsschätzung).
    Enthält zusätzlich die RF-Feature-Importance für Lernwert-Ranking.
    """
    # Build importance lookup
    imp_lookup = dict(zip(importance_df["skill"], importance_df["importance"]))
    max_imp = importance_df["importance"].max() if not importance_df.empty else 1.0

    rows = []
    for _, row in skill_score_df.iterrows():
        skill = row["skill"]
        if skill not in skill_dim_map: continue
        dim, _ = skill_dim_map[skill]
        matches = rh_sim_map.get(dim, [])
        if not matches: continue
        rh_mid = matches[0]["salary"]["mid"]
        rh_salary_score = min((rh_mid / RH_GLOBAL_MEDIAN) * 50, 100)

        base = row["demand_pct"] * 0.30 + row["career_ratio"] * 100 * 0.40 + rh_salary_score * 0.30
        modifier = row.get("modifier", "none")
        if modifier == "basic_penalty":  base *= 0.2
        elif modifier == "soft_penalty": base *= 0.4
        elif modifier == "sig_boost":    base *= 1.25

        imp_norm = (imp_lookup.get(skill, 0) / max_imp) * 100
        lernwert = round((rh_salary_score * 0.5 + imp_norm * 0.5), 1)

        rows.append({
            "skill":        skill,
            "svs":          row["score"],
            "mws":          round(min(base, 100), 1),
            "demand_pct":   row["demand_pct"],
            "career_ratio": row["career_ratio"],
            "modifier":     modifier,
            "rh_mid":       rh_mid,
            "rh_profile":   matches[0]["profile"],
            "rf_importance":round(imp_norm, 1),
            "lernwert":     lernwert,
            "dim":          dim,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df["diff"] = (df["mws"] - df["svs"]).round(1)
    return df.sort_values("mws", ascending=False).reset_index(drop=True)


def get_missing_skills_by_lernwert(missing_set, mws_df):
    """Missing skills sorted by Lernwert (RF-Importance × RH-Salary)."""
    if mws_df.empty: return []
    result = mws_df[mws_df["skill"].isin(missing_set)].copy()
    result = result.sort_values("lernwert", ascending=False)
    return result.to_dict("records")


def estimate_level(coverage_pct, salary):
    if coverage_pct >= 70:   return "Senior Level", salary["senior"]
    elif coverage_pct >= 40: return "Mid Level",    salary["mid"]
    else:                    return "Entry Level",  salary["entry"]


def make_radar_chart(cluster_gaps):
    top = cluster_gaps[:10]
    labels = [g["cluster"] for g in top]
    coverages = [g["coverage"] for g in top]
    lc = labels + [labels[0]]; cc = coverages + [coverages[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100]*len(lc), theta=lc, fill="toself",
        fillcolor="rgba(255,255,255,0.04)", line=dict(color="#2A4060", width=1)))
    fig.add_trace(go.Scatterpolar(
        r=cc, theta=lc, fill="toself", name="Deine Coverage",
        fillcolor="rgba(74,230,138,0.18)", line=dict(color="#4AE68A", width=2),
        hovertemplate="%{theta}<br>Coverage: %{r}%<extra></extra>"))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], color="#444",
                            tickcolor="#444", gridcolor="#2A3040", tickfont=dict(size=9)),
            angularaxis=dict(color="#888", gridcolor="#2A3040"),
            bgcolor="rgba(0,0,0,0)"),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", size=11), height=370,
        margin=dict(l=55, r=55, t=30, b=30))
    return fig


def compute_dim_requirements(all_sims):
    """For each RH profile: {dimension: requirement_pct (0–100)}.
    all_sims: {profile: {dim: similarity}} — full matrix from compute_full_profile_similarities.
    Normalises per profile so the top dimension = 100%."""
    requirements = {}
    for profile, dim_sims in all_sims.items():
        if not dim_sims:
            requirements[profile] = {}
            continue
        max_s = max(dim_sims.values()) or 1.0
        requirements[profile] = {dim: round(s / max_s * 100) for dim, s in dim_sims.items()}
    return requirements


def make_interactive_radar(cluster_gaps, requirements, selected_profiles):
    """Radar: user coverage (green) + up to 2 profile requirement rings (dashed)."""
    top    = cluster_gaps[:10]
    labels = [g["cluster"] for g in top]
    cov    = [g["coverage"] for g in top]
    lc = labels + [labels[0]]; cc = cov + [cov[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100]*len(lc), theta=lc, fill="toself",
        fillcolor="rgba(255,255,255,0.03)",
        line=dict(color="#1E2A3A", width=1),
        showlegend=False, hoverinfo="skip"))

    profile_styles = [
        ("107,159,212", "#6B9FD4", "dot"),
        ("241,196,15",  "#F1C40F", "dot"),
    ]
    for idx, profile in enumerate(selected_profiles[:2]):
        reqs    = requirements.get(profile, {})
        req_v   = [reqs.get(dim, 0) for dim in labels]
        req_vc  = req_v + [req_v[0]]
        rgb, hex_c, dash = profile_styles[idx]
        fig.add_trace(go.Scatterpolar(
            r=req_vc, theta=lc, fill="toself",
            name=f"🎯 {profile}",
            fillcolor=f"rgba({rgb},0.10)",
            line=dict(color=hex_c, width=2, dash=dash),
            hovertemplate="%{theta}<br>Anforderung: %{r}%<extra>" + profile + "</extra>"))

    fig.add_trace(go.Scatterpolar(
        r=cc, theta=lc, fill="toself", name="✅ Deine Skills",
        fillcolor="rgba(74,230,138,0.22)",
        line=dict(color="#4AE68A", width=2.5),
        hovertemplate="%{theta}<br>Deine Coverage: %{r}%<extra></extra>"))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#444",
                            tickcolor="#444", gridcolor="#2A3040", tickfont=dict(size=9)),
            angularaxis=dict(color="#888", gridcolor="#2A3040"),
            bgcolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(0,0,0,0.35)", font=dict(size=11),
                    orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", size=11), height=430,
        margin=dict(l=60, r=60, t=30, b=50))
    return fig


def analyse_profile_gaps(profile, requirements, cluster_gaps):
    """Dimension-by-dimension gap analysis for a selected RH profile."""
    reqs = requirements.get(profile, {})
    result = []
    for gap in cluster_gaps:
        dim = gap["cluster"]
        req = reqs.get(dim, 0)
        if req < 5: continue
        cov     = gap["coverage"]
        gap_pct = max(0, req - cov)
        result.append({"dim": dim, "requirement": req, "coverage": cov,
                        "gap": gap_pct, "missing": gap["missing"][:4],
                        "has": gap["has"][:4]})
    return sorted(result, key=lambda x: x["gap"], reverse=True)


def generate_rapport(selected_skills, missing_set, best_fit_roles,
                     selected_profiles, cluster_gaps, requirements, primary_cat):
    """Generate a plain-text career rapport for download."""
    import datetime
    today = datetime.date.today().strftime("%d.%m.%Y")
    lines = [
        "=" * 62,
        "SKILLBRIDGE — PERSÖNLICHER KARRIERE-RAPPORT",
        f"Bereich: {primary_cat}",
        f"Datum:   {today}",
        "=" * 62, "",
        "DEINE AKTUELLEN SKILLS",
        "-" * 40,
    ]
    for s in sorted(selected_skills):
        lines.append(f"  ✅ {s.title()}")
    lines += ["", "FEHLENDE SKILLS (Top 10 nach Lernwert)", "-" * 40]
    for s in sorted(list(missing_set))[:10]:
        lines.append(f"  🔴 {s.title()}")
    lines += ["", "TOP PASSENDE JOBPROFILE", "-" * 40]
    for role in best_fit_roles[:5]:
        sal = role["salary"]
        lines.append(f"  {role['fit_pct']:3d}%  {role['profile']}")
        lines.append(f"        Entry CHF {sal['entry']:,} · Mid CHF {sal['mid']:,} · Senior CHF {sal['senior']:,}")
    lines.append("")

    for profile in selected_profiles:
        sal      = RH_SALARY_DATA.get(profile, {})
        dim_gaps = analyse_profile_gaps(profile, requirements, cluster_gaps)
        tot_req  = sum(d["requirement"] for d in dim_gaps) or 1
        w_gap    = round(sum(d["gap"] * d["requirement"] for d in dim_gaps) / tot_req)
        lines += [f"PROFIL-ANALYSE: {profile}", "-" * 40]
        if sal:
            lines.append(f"  Marktgehalt CH 2024 (Robert Half):")
            lines.append(f"    Entry CHF {sal['entry']:,} | Mid CHF {sal['mid']:,} | Senior CHF {sal['senior']:,}")
        lines.append(f"  Gesamtlücke zu diesem Profil: {w_gap}%")
        lines += ["", "  PRIORITÄRE LERNBEREICHE:"]
        for d in dim_gaps[:6]:
            if d["gap"] < 3: continue
            miss_str = ", ".join(s.title() for s in d["missing"][:3])
            lines.append(f"  [{d['gap']:3d}% Lücke]  {d['dim']}")
            lines.append(f"              Anforderung: {d['requirement']}% | Deine Coverage: {d['coverage']}%")
            if miss_str:
                lines.append(f"              Fehlende Skills: {miss_str}")
        lines.append("")

    lines += [
        "=" * 62,
        "DATENQUELLEN",
        "  LinkedIn Finance & Banking — 49'000 Stellenanzeigen",
        "  Robert Half Salary Guide Schweiz 2024",
        "  Keine Gewähr auf Vollständigkeit oder Aktualität.",
        "=" * 62,
    ]
    return "\n".join(lines)


def make_svs_mws_comparison_chart(mws_df, selected_set, top_n=15):
    """Grouped bar chart: Skill-Value-Score vs Markt-Wert-Score für Top-N Skills."""
    df = mws_df.copy()
    df["Status"] = df["skill"].apply(lambda s: "Vorhanden" if s in selected_set else "Fehlend")
    top = df.nlargest(top_n, "mws")
    top = top.sort_values("mws", ascending=True)

    fig = go.Figure()
    colors_svs = ["#4A7CC0" if s == "Fehlend" else "#2A5A8A" for s in top["Status"]]
    colors_mws = ["#4AE68A" if s == "Fehlend" else "#27AE60" for s in top["Status"]]

    fig.add_trace(go.Bar(
        name="Skill-Value-Score (LinkedIn)", y=top["skill"].str.title(),
        x=top["svs"], orientation="h",
        marker_color=colors_svs, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>SVS: %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Bar(
        name="Markt-Wert-Score (RH CH 2024)", y=top["skill"].str.title(),
        x=top["mws"], orientation="h",
        marker_color=colors_mws, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>MWS: %{x:.1f}<extra></extra>"))

    fig.update_layout(
        barmode="group", height=max(350, top_n * 32),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(gridcolor="#2A3040", range=[0, 100]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=20, t=40, b=10))
    return fig


def make_scatter_svs_mws(mws_df, selected_set):
    """Scatter: SVS (x) vs Markt-Wert-Score (y), Quadranten-Labels."""
    df = mws_df.copy()
    df["Status"] = df["skill"].apply(lambda s: "Fehlend" if s in selected_set else "Vorhanden")
    df = df.head(60)
    if df.empty: return None
    med_x, med_y = df["svs"].median(), df["mws"].median()
    fig = px.scatter(
        df, x="svs", y="mws", color="Status", size="demand_pct",
        hover_name="skill",
        color_discrete_map={"Fehlend": "#FF6B6B", "Vorhanden": "#4AE68A"},
        labels={"svs": "Skill-Value-Score (0–100, LinkedIn)", "mws": "Markt-Wert-Score (0–100, RH CH)"},
        hover_data={"demand_pct": True, "Status": False, "rh_mid": True},
    )
    fig.add_hline(y=med_y, line_dash="dash", line_color="#2A4060",
                  annotation_text="Ø MWS", annotation_font_color="#6B9FD4",
                  annotation_position="bottom right")
    fig.add_vline(x=med_x, line_dash="dash", line_color="#2A4060",
                  annotation_text="Ø SVS", annotation_font_color="#6B9FD4",
                  annotation_position="top right")
    x_max, y_max = df["svs"].max(), df["mws"].max()
    for txt, x, y, col in [
        ("🏆 Jetzt lernen<br><i>Hoch nachgefragt + top Gehalt</i>",  x_max*0.97, y_max*0.97, "#4AE68A"),
        ("💡 Strategisch<br><i>Weniger verbreitet, gut bezahlt</i>", med_x*0.1,  y_max*0.97, "#F1C40F"),
        ("🔄 Grundlage<br><i>Nötig, kein starker Gehaltseffekt</i>", x_max*0.97, med_y*0.86, "#888888"),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(color=col, size=10), xanchor="right",
                           bgcolor="rgba(0,0,0,0.45)", borderpad=4)
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="#222")))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)", title=None))
    return fig


# ── Step 5: Dimensions-Karte rendern ──────────────────────────
def render_dimension_card(gap, skill_score_df, selected_set, global_sal_median):
    cluster  = gap["cluster"]
    coverage = gap["coverage"]
    if coverage == 0:   color, badge = "#C0392B", "🔴"
    elif coverage < 50: color, badge = "#E67E22", "🟠"
    elif coverage < 80: color, badge = "#F1C40F", "🟡"
    else:               color, badge = "#27AE60", "🟢"

    miss_hard = [s for s in gap["missing"] if classify_skill(s) == "hard"]
    miss_soft = [s for s in gap["missing"] if classify_skill(s) == "soft"]
    has_hard  = [s for s in gap["has"]     if classify_skill(s) == "hard"]
    has_soft  = [s for s in gap["has"]     if classify_skill(s) == "soft"]

    def get_score(s):
        row = skill_score_df[skill_score_df["skill"] == s]
        return row["score"].values[0] if not row.empty else 0

    def skill_tag(s, missing):
        score_row = skill_score_df[skill_score_df["skill"] == s]
        if not score_row.empty:
            row = score_row.iloc[0]
            score = int(row["score"]); demand = row["demand_pct"]
            career = row["career_ratio"]; salary = row["salary_ratio"]
            modifier = row.get("modifier", "none"); score_str = f" · {score}"
            if modifier == "soft_penalty":    mod_str = "⚠️ Soft-Skill Penalty × 0.4"
            elif modifier == "basic_penalty": mod_str = "📉 Basic-Skill Penalty × 0.2"
            elif modifier == "sig_boost":     mod_str = "✅ Signifikanz-Boost × 1.25"
            else:                             mod_str = "➖ Kein Modifier"
            tooltip = (f"Skill-Value-Score: {score}/100&#10;"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━&#10;"
                       f"📊 Nachfrage (30%): {demand}% der Jobs&#10;"
                       f"🚀 Karriere  (40%): {career}x Senior-Anteil&#10;"
                       f"💰 Gehalt    (30%): {salary:.2f}x Median&#10;"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━&#10;{mod_str}")
        else:
            score_str = ""; tooltip = "Kein Score"
        _miss_bg = "#FDEAEA" if T.get("bg2") == "#FFFFFF" else "#3D1010"
        _has_bg  = "#E8F8EE" if T.get("bg2") == "#FFFFFF" else "#0D2E1A"
        if missing:
            return (f'<span title="{tooltip}" style="background:{_miss_bg};color:{T["red"]};'
                    f'padding:2px 8px;border-radius:12px;font-size:0.82em;margin:2px;'
                    f'display:inline-block;cursor:help">{s.title()}{score_str}</span>')
        return (f'<span title="{tooltip}" style="background:{_has_bg};color:{T["green"]};'
                f'padding:2px 8px;border-radius:12px;font-size:0.82em;margin:2px;'
                f'display:inline-block;cursor:help">{s.title()}{score_str}</span>')

    ch1, ch2 = st.columns([2, 5])
    with ch1:
        sal_median = gap["sal_median"]; has_salary = sal_median != global_sal_median
        priority_pct = round(gap["priority_score"] * 100)
        gap_pct = 100 - coverage
        priority_tooltip = (
            f"Priorität = Gehalts-Potenzial × Verbleibende Lücke&#10;"
            f"💰 {round(gap['sal_norm']*100)}% des Kategorie-Medians&#10;"
            f"Coverage = {gap['has_score']:.0f} ÷ {gap['total_score']:.0f} = {coverage}%&#10;"
            f"Score = {gap['sal_norm']:.2f} × {gap_pct/100:.2f} = {gap['priority_score']:.3f}"
        )
        st.markdown(
            f'<div style="margin-top:8px"><b style="color:{T["text"]}">{cluster}</b> '
            f'<span style="color:{T["muted"]};font-size:0.85em">{gap["pct"]}% der Stellen</span></div>',
            unsafe_allow_html=True)
        if has_salary:
            st.markdown(
                f'<div style="color:{T["yellow"]};font-size:0.82em;margin:2px 0">💰 ${sal_median:,.0f} Median</div>',
                unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin:3px 0"><span title="{priority_tooltip}" '
            f'style="color:{T["yellow"]};font-size:0.82em;font-weight:600;cursor:help">'
            f'Priorität: {priority_pct}</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{T["progress"]};border-radius:4px;margin:4px 0 2px 0">'
            f'<div style="background:{color};width:{coverage}%;height:8px;border-radius:4px"></div>'
            f'</div><span style="color:{T["muted"]};font-size:0.78em">{coverage}% Coverage</span>',
            unsafe_allow_html=True)
    with ch2:
        if miss_hard or has_hard:
            all_hard = sorted([(s, True) for s in miss_hard]+[(s, False) for s in has_hard],
                              key=lambda x: get_score(x[0]), reverse=True)
            st.markdown(
                f'<div style="margin-bottom:4px">'
                f'<span style="font-size:0.78em;color:{T["muted"]};font-weight:600;margin-right:4px">🔧 HARD</span>'
                + "".join(skill_tag(s, m) for s, m in all_hard) + '</div>',
                unsafe_allow_html=True)
        if miss_soft or has_soft:
            all_soft = sorted([(s, True) for s in miss_soft]+[(s, False) for s in has_soft],
                              key=lambda x: get_score(x[0]), reverse=True)
            st.markdown(
                f'<div style="margin-bottom:4px">'
                f'<span style="font-size:0.78em;color:{T["muted"]};font-weight:600;margin-right:4px">💬 SOFT</span>'
                + "".join(skill_tag(s, m) for s, m in all_soft) + '</div>',
                unsafe_allow_html=True)
    st.markdown(f'<hr style="margin:6px 0;border-color:{T["border"]}">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# UI START  (Button zuerst → T danach mit aktuellem Wert)
# ═══════════════════════════════════════════════════════════════

if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

_hdr_l, _hdr_r = st.columns([9, 1])
with _hdr_l:
    st.markdown("### HSG CAPSTONE")
with _hdr_r:
    _lbl = "☀️ Hell" if not st.session_state.light_mode else "🌙 Dunkel"
    if st.button(_lbl, key="theme_toggle", use_container_width=True):
        st.session_state.light_mode = not st.session_state.light_mode
        # Kein st.rerun() – T wird direkt darunter mit dem neuen Wert berechnet

# ═══════════════════════════════════════════════════════════════
# THEME  (nach Button-Handler, damit T immer den aktuellen Wert hat)
# ═══════════════════════════════════════════════════════════════

if st.session_state.light_mode:
    T = dict(
        bg="#F0F4FA", bg2="#FFFFFF", bg3="#E8EEF8",
        border="#C5D5E8", border2="#9AAFC4",
        text="#1A2A40", muted="#5A6A7A", dim="#8A9AAA",
        accent="#2862A8", green="#1A9050", yellow="#9A7A00", red="#CC3333",
        progress="#D0DCE8", lv_bg="#EEF8DC", lv_border="#7AAB2A",
        skill_bg="#FFFFFF", rank1="#5A8A10", rank2="#2862A8", rank3="#7A8A9A",
        plot_bg="rgba(248,250,253,0)", plot_paper="rgba(248,250,253,0)",
        font_clr="#333333", grid="#C0CEDC",
    )
    st.markdown("""<style>
    /* ── Seitenhintergrund ── */
    .stApp, .stAppViewContainer, section.main,
    .block-container, [data-testid="stHeader"] {
        background: #F8FAFD !important;
    }
    /* ── Gesamter Text dunkel ── */
    .stApp p, .stApp span, .stApp div, .stApp li,
    .stApp label, .stApp small, .stApp strong, .stApp em,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stCheckbox"] p,
    [data-testid="stToggle"] p,
    [data-baseweb="checkbox"] span,
    [class*="st-emotion"] p, [class*="st-emotion"] span {
        color: #31333F !important;
    }
    /* ── Überschriften ── */
    h1, h2, h3, h4, h5, h6 { color: #1A2A40 !important; }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { color: #555 !important; }
    .stTabs [aria-selected="true"] { color: #1A2A40 !important; font-weight:700 !important; }
    /* ── Expander ── */
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span { color: #1A2A40 !important; }
    /* ── Eingabefelder / Dropdowns ── */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div,
    [data-baseweb="input"],
    [data-baseweb="input"] > div { background: #FFFFFF !important; color: #31333F !important; }
    [data-baseweb="select"] *,
    [data-baseweb="input"] * { color: #31333F !important; }
    [data-baseweb="popover"],
    [data-baseweb="popover"] * { background: #FFFFFF !important; color: #31333F !important; }
    /* ── Buttons ── */
    [data-testid="stButton"] > button,
    [data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #1A2A40 !important;
        border: 1px solid #C5D5E8 !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #E8EEF8 !important;
        border-color: #6B9FD4 !important;
    }
    /* ── Hinweise / Alerts ── */
    [data-testid="stAlert"] p { color: #1A2A40 !important; }
    [data-testid="stSidebar"] { background: #EEF2F8 !important; }
    [data-testid="stDownloadButton"] button { background: #E8EEF8 !important; color: #1A2A40 !important; border-color: #9AAFC4 !important; }
    /* ── Container-Rahmen im Hellmodus (Emotion-Klasse direkt angesprochen) ── */
    /* st-emotion-cache-1gz5zxc ist die Klasse die Streamlit für border=True Container setzt */
    html body .st-emotion-cache-1gz5zxc,
    html body [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1.5px solid #6090B8 !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
    }
    /* ── Toggle-Widget: ganzer Bereich sichtbar machen ── */
    [data-testid="stToggle"] {
        border: 1.5px solid #A0BACE !important;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        background-color: rgba(220, 234, 248, 0.45) !important;
    }
    </style>""", unsafe_allow_html=True)
else:
    T = dict(
        bg="#131C2B", bg2="#0D1520", bg3="#1A2840",
        border="#1E3050", border2="#2A4060",
        text="#D0D8E8", muted="#888888", dim="#555555",
        accent="#6B9FD4", green="#4AE68A", yellow="#F1C40F", red="#FF6B6B",
        progress="#1E2A3A", lv_bg="#1A2800", lv_border="#3A5010",
        skill_bg="#0D1520", rank1="#C5E828", rank2="#6B9FD4", rank3="#555555",
        plot_bg="rgba(0,0,0,0)", plot_paper="rgba(0,0,0,0)",
        font_clr="#aaaaaa", grid="#2A3040",
    )
st.title("SkillBridge")
st.caption("Finde heraus, welche Skills du für deinen Traumjob brauchst.")
st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

# ── Schritt 1 ─────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Schritt 1: In welchem Bereich möchtest du arbeiten?")
    selected_categories = []
    cols = st.columns(len(DATASETS))
    for i, cat in enumerate(DATASETS.keys()):
        if cols[i].toggle(cat, key=f"cat_{cat}"): selected_categories.append(cat)
    if not selected_categories:
        st.info("👆 Wähle oben mindestens einen Jobbereich aus.")
if not selected_categories:
    st.stop()

primary_cat = selected_categories[0]
filepath    = os.path.join(BASE_PATH, DATASETS[primary_cat])

with st.spinner("⏳ Datensatz und Modell werden geladen..."):
    top_skills_dict   = get_top_skills(filepath)
    cluster_profiles  = get_cluster_profiles(filepath)
    importance_df     = train_random_forest(filepath)
    salary_impact_df  = compute_salary_impact(primary_cat)
    cluster_salary_df = compute_cluster_salary(primary_cat)
    skill_score_df    = compute_skill_value_score(filepath, primary_cat)

# ── Schritt 2: Skill-Auswahl ──────────────────────────────────
st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 2: Welche Skills bringst du bereits mit?")
    st.caption("Wähle alle Skills aus die auf dich zutreffen.")

    all_skills_set = set()
    for profile in cluster_profiles.values():
        for s in profile["top_skills"]:
            if is_valid_skill(s): all_skills_set.add(s)

    hard_list = sorted([s for s in all_skills_set if classify_skill(s) == "hard"])
    soft_list  = sorted([s for s in all_skills_set if classify_skill(s) == "soft"])

    st.markdown(
        f'<div style="background:{T["bg3"]};border:1px solid {T["border2"]};border-radius:10px;'
        f'padding:12px 20px 4px 20px;margin-bottom:12px">'
        f'<span style="font-size:1em;font-weight:700;color:{T["accent"]}">🔧 Hard Skills</span></div>',
        unsafe_allow_html=True)
    n_hard = len(hard_list); chunk = max(1, -(-n_hard // 3)); h_cols = st.columns(3)
    for i, skill in enumerate(hard_list):
        with h_cols[i // chunk]: st.checkbox(skill.title(), key=f"sk2_{skill}")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{T["lv_bg"]};border:1px solid {T["lv_border"]};border-radius:10px;'
        f'padding:12px 20px 4px 20px;margin-bottom:12px">'
        f'<span style="font-size:1em;font-weight:700;color:{T["green"]}">💬 Soft Skills</span></div>',
        unsafe_allow_html=True)
    n_soft = len(soft_list); n_scols = 2 if n_soft <= 12 else 3
    schunk = max(1, -(-n_soft // n_scols)); s_cols = st.columns(n_scols)
    for i, skill in enumerate(soft_list):
        with s_cols[i // schunk]: st.checkbox(skill.title(), key=f"sk2_{skill}")

    selected_skills = list(set(
        key.replace("sk2_", "", 1)
        for key, val in st.session_state.items()
        if key.startswith("sk2_") and val
    ))
    if not selected_skills:
        st.warning("Bitte wähle mindestens einen Skill aus.")
    else:
        st.success(f"**{len(selected_skills)} Skills ausgewählt:** "
                   f"{', '.join(s.title() for s in sorted(selected_skills))}")
    if st.button("Weiter zur Analyse →", type="primary"):
        st.session_state["analyse_ready"] = True

if not selected_skills:
    st.stop()
if not st.session_state.get("analyse_ready", False):
    st.stop()

cluster_skills_all = set(
    s for profile in cluster_profiles.values()
    for s in profile["top_skills"] if is_valid_skill(s)
)
top_skills_dict = {k: v for k, v in top_skills_dict.items() if k in cluster_skills_all}
selected_set    = set(selected_skills)
all_top_set     = set(top_skills_dict.keys())
missing_skills  = sorted(all_top_set - selected_set)
missing_set     = set(missing_skills)

# ── Schritt 3: Skill-Vergleich ────────────────────────────────
st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 3: Skill-Vergleich")
    st.caption("Wie häufig wird jeder Skill in Stellenanzeigen verlangt? Deine Skills sind mit ✅ markiert.")

    def render_skill_row(skill, pct):
        is_own = skill in selected_set
        label  = f"✅ **{skill.title()}**" if is_own else f"　 {skill.title()}"
        color  = "#1A5C38" if pct >= 60 else ("#4A8C63" if pct >= 30 else "#7BB899")
        c1, c2, c3 = st.columns([3, 5, 1])
        c1.markdown(label)
        c2.markdown(
            f'<div style="background:{color};width:{min(pct*1.2,100):.0f}%;'
            f'height:14px;border-radius:4px;margin-top:5px"></div>', unsafe_allow_html=True)
        c3.markdown(f"**{pct}%**")

    sorted_skills   = sorted(top_skills_dict.items(), key=lambda x: x[1], reverse=True)
    top5            = sorted_skills[:5]; rest = sorted_skills[5:]
    own_not_in_top5 = [(s, p) for s, p in rest if s in selected_set]
    own_top5 = own_not_in_top5[:5]; own_rest = own_not_in_top5[5:]
    rest_not_own = [(s, p) for s, p in rest if s not in selected_set]

    s3_col1, s3_col2 = st.columns(2)
    with s3_col1:
        st.markdown(f'<span style="color:{T["yellow"]};font-size:0.78em;font-weight:600">⭐ TOP 5 IN DER BRANCHE</span>', unsafe_allow_html=True)
        for skill, pct in top5: render_skill_row(skill, pct)
        with st.expander(f"Alle weiteren Skills ({len(rest_not_own)})"):
            for skill, pct in rest_not_own: render_skill_row(skill, pct)
    with s3_col2:
        st.markdown(f'<span style="color:{T["green"]};font-size:0.78em;font-weight:600">✅ DEINE TOP SKILLS</span>', unsafe_allow_html=True)
        if own_top5:
            for skill, pct in own_top5: render_skill_row(skill, pct)
            if own_rest:
                with st.expander(f"Alle weiteren eigenen Skills ({len(own_rest)})"):
                    for skill, pct in own_rest: render_skill_row(skill, pct)
        else:
            st.caption("Keine deiner Skills in den Top-Skills gefunden.")
    st.caption("🟢 ≥ 60% sehr wichtig &nbsp;|&nbsp; 🟡 30–60% relevant &nbsp;|&nbsp; ⚪ < 30% selten")

# ── Schritt 4: Mein Gap ───────────────────────────────────────
st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 4: Mein Gap")
    st.caption(
        "Fehlende Skills priorisiert nach **Skill-Value-Score** (SVS) — "
        "basierend auf LinkedIn-Daten: Karrierewachstum (40%) + Nachfrage (30%) + Gehalts-Premium (30%). "
        "Soft Skills × 0.4 · Basic Skills × 0.2 · Hard Skills mit stat. Gehaltseffekt × 1.25."
    )
    total = len(top_skills_dict); matched = sum(1 for s in selected_skills if s in top_skills_dict)
    match_pct = round(matched / total * 100) if total > 0 else 0
    st.markdown(
        f'<div style="background:{T["bg3"]};border:1px solid {T["border"]};border-radius:8px;padding:10px 16px;margin-bottom:12px">'
        f'<span style="font-size:1.4em;font-weight:700;color:{T["green"]}">{match_pct}%</span>'
        f'<span style="color:{T["muted"]};font-size:0.9em;margin-left:12px">Match · {primary_cat}</span>'
        f'<span style="color:{T["text"]};font-size:0.9em;margin-left:24px">'
        f'✅ {matched} von {total} Skills · 🔴 {total - matched} fehlend</span></div>',
        unsafe_allow_html=True)
    ci1, ci2, ci3 = st.columns(3)
    ci1.info("📊 **Nachfrage (30%)** — Wie oft wird der Skill in Stellenanzeigen verlangt?")
    ci2.info("🚀 **Karrierewachstum (40%)** — Wie stark korreliert der Skill mit Senior-Positionen?")
    ci3.info("💰 **Gehalts-Premium (30%)** — Geschätztes Gehalt basierend auf LinkedIn-Jobdaten")
    st.markdown("#### 📊 Skill-Value Matrix (Interaktiv)")
    st.caption(
        "X-Achse: Wie oft wird der Skill nachgefragt? "
        "Y-Achse: Skill-Value-Score — kombiniert Nachfrage, Karrierewachstum und Gehaltseffekt. "
        "Fahre über Punkte für Details."
    )
    plot_df = skill_score_df.copy()
    plot_df["Typ"] = plot_df["skill"].apply(lambda s: "Soft Skill" if classify_skill(s) == "soft" else "Hard Skill")
    plot_df["Status"] = plot_df["skill"].apply(lambda s: "✅ Vorhanden" if s in selected_set else "🔴 Fehlend")
    fig = px.scatter(
        plot_df.head(40), x="demand_pct", y="score", color="Typ", symbol="Status", hover_name="skill",
        hover_data={"score": True, "demand_pct": True, "Typ": False, "Status": False},
        labels={"demand_pct": "Nachfrage (% der Stellen)", "score": "Skill-Value-Score (0–100)"},
        color_discrete_map={"Hard Skill": "#4AE68A", "Soft Skill": "#F1C40F"})
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=T["font_clr"]))
    st.plotly_chart(fig, use_container_width=True)

    scored_missing = skill_score_df[skill_score_df["skill"].isin(missing_set)].reset_index(drop=True)
    scored_own     = skill_score_df[skill_score_df["skill"].isin(selected_set)].reset_index(drop=True)

    def render_score_row(rank, row, max_score, owned=False):
        skill = row["skill"]; score = row["score"]; demand = row["demand_pct"]
        career = row["career_ratio"]; salary = row["salary_ratio"]
        modifier = row.get("modifier", "none"); bar = int((score / max_score) * 100)
        if owned:
            color, prefix = "#1A5C38", "✅"
        else:
            color  = "#C0392B" if rank < 3 else ("#E67E22" if rank < 8 else "#7BB899")
            badge  = "🔴" if rank < 3 else ("🟠" if rank < 8 else "⚪")
            prefix = f"{badge} **{rank+1}.**" if rank < 3 else f"{badge} {rank+1}."
        _boost_bg  = "#E8F8EE" if T.get("bg2") == "#FFFFFF" else "#1A5C38"
        _badge_bg  = T["bg3"]
        if modifier == "sig_boost":
            mod_badge = f' <span style="background:{_boost_bg};color:{T["green"]};font-size:0.7em;padding:1px 5px;border-radius:8px">✅ sig. Gehaltsboost</span>'
        elif modifier == "soft_penalty":
            mod_badge = f' <span style="background:{_badge_bg};color:{T["muted"]};font-size:0.7em;padding:1px 5px;border-radius:8px">⚠️ Soft-Skill</span>'
        elif modifier == "basic_penalty":
            mod_badge = f' <span style="background:{_badge_bg};color:{T["muted"]};font-size:0.7em;padding:1px 5px;border-radius:8px">📉 Basic-Skill</span>'
        else:
            mod_badge = ""
        demand_c = round(demand / 100 * 0.30 * 100, 1)
        career_c = round(career * 100 * 0.40, 1)
        salary_c = round(min((salary - 1) * 50 + 50, 100) * 0.30, 1) if salary else 0
        base_pre = round(demand_c + career_c + salary_c, 1)
        if modifier == "soft_penalty":    mod_f = f"× 0.4 = {score}"
        elif modifier == "basic_penalty": mod_f = f"× 0.2 = {score}"
        elif modifier == "sig_boost":     mod_f = f"× 1.25 = {score}"
        else:                             mod_f = f"= {score}"
        st.markdown(
            f'{prefix} **{skill.title()}**{mod_badge}'
            f'<br><span style="color:{T["muted"]};font-size:0.75em;font-family:monospace">'
            f'({demand}%×0.3) + ({career}x×0.4×100) + (Gehalt×0.3) → {base_pre} {mod_f}</span>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{color};width:{bar}%;height:10px;border-radius:4px;margin-bottom:6px"></div>',
            unsafe_allow_html=True)

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown("**🎯 Fehlende Skills — nach Skill-Value-Score**")
        if scored_missing.empty:
            st.success("Keine fehlenden Skills!")
        else:
            max_score = scored_missing["score"].max()
            for rank, row in scored_missing.head(3).iterrows(): render_score_row(rank, row, max_score)
            if len(scored_missing) > 3:
                with st.expander(f"Alle weiteren fehlenden Skills ({len(scored_missing) - 3})"):
                    for rank, row in scored_missing.iloc[3:].iterrows(): render_score_row(rank, row, max_score)
    with cg2:
        st.markdown("**✅ Deine Skills — nach Skill-Value-Score**")
        if scored_own.empty:
            st.warning("Keine deiner Skills im Datensatz gefunden.")
        else:
            max_score2 = scored_own["score"].max()
            for _, row in scored_own.head(3).iterrows(): render_score_row(0, row, max_score2, owned=True)
            if len(scored_own) > 3:
                with st.expander(f"Alle weiteren eigenen Skills ({len(scored_own) - 3})"):
                    for _, row in scored_own.iloc[3:].iterrows(): render_score_row(0, row, max_score2, owned=True)

# ── Schritt 5: Lernpfad ───────────────────────────────────────
st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 5: Lernpfad")
    st.caption(
        "Dimensionen priorisiert nach Gehalts-Potenzial × verbleibende Lücke. "
        "Dimensionen mit vielen fehlenden Skills und hohem Gehaltsniveau kommen zuerst."
    )

    with st.expander("ℹ️ Was sind Skill-Dimensionen?"):
        st.markdown("""
**Skill-Dimensionen** sind Gruppen von Skills, die häufig zusammen in Stellenanzeigen auftreten —
sie beschreiben typische Tätigkeitsfelder wie *Auditing*, *Data Analysis* oder *Risk & Compliance*.

Diese Dimensionen wurden automatisch aus über **49'000 LinkedIn-Stellenanzeigen** im Bereich
Finance & Banking extrahiert. Das Verfahren heisst **NMF (Non-negative Matrix Factorization)**
und erkennt, welche Skills immer wieder gemeinsam gesucht werden — ohne dass man das manuell
definieren muss.

Jede Dimension steht also für ein reales, am Markt nachgefragtes Tätigkeitsprofil.
        """)

    ci1, ci2, ci3 = st.columns(3)
    ci1.info("🧩 **Methode**\nMaschinelles Lernen auf 49k Stellenanzeigen (NMF)")
    ci2.info("🎯 **Coverage**\nWie viel % der Dimension deckst du ab?")
    ci3.info("🔧💬 **Hard & Soft**\nSeparat innerhalb jeder Dimension")

    global_sal_median = 100000.0; cluster_sal_lookup = {}
    if not cluster_salary_df.empty:
        global_sal_median = cluster_salary_df["median_salary"].median()
        cluster_sal_lookup = dict(zip(cluster_salary_df["cluster"], cluster_salary_df["median_salary"]))

    cluster_gaps = []
    for cluster_name, profile in cluster_profiles.items():
        top = profile["top_skills"]
        if not top: continue
        has = [s for s in top if s in selected_set]
        missing = [s for s in top if s not in selected_set]
        def get_skill_score(s):
            row = skill_score_df[skill_score_df["skill"] == s]
            return float(row["score"].values[0]) if not row.empty else 10.0
        total_score = sum(get_skill_score(s) for s in top)
        has_score   = sum(get_skill_score(s) for s in has)
        coverage    = round(has_score / total_score * 100) if total_score > 0 else 0
        sal_median  = cluster_sal_lookup.get(cluster_name, global_sal_median)
        sal_norm    = sal_median / global_sal_median
        priority_score = sal_norm * ((100 - coverage) / 100)
        cluster_gaps.append({
            "cluster": cluster_name, "coverage": coverage, "has": has, "missing": missing,
            "pct": profile["pct"], "sal_median": sal_median, "sal_norm": round(sal_norm, 2),
            "priority_score": round(priority_score, 3),
            "total_score": round(total_score, 1), "has_score": round(has_score, 1),
        })
    cluster_gaps.sort(key=lambda x: x["priority_score"], reverse=True)

    for gap in cluster_gaps[:3]:
        render_dimension_card(gap, skill_score_df, selected_set, global_sal_median)

    if len(cluster_gaps) > 3:
        with st.expander(f"Alle weiteren {len(cluster_gaps) - 3} Dimensionen anzeigen"):
            for gap in cluster_gaps[3:]:
                render_dimension_card(gap, skill_score_df, selected_set, global_sal_median)

# ── Schritt 6: Schweizer Gehalts-Benchmark ────────────────────
st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 6: Schweizer Gehalts-Benchmark")
    st.caption(
        "Basierend auf dem **Robert Half Salary Guide Schweiz 2024**. "
        "Robert Half ist ein führendes Personalvermittlungsunternehmen, das jährlich Schweizer "
        "Marktgehälter für Finance & Banking veröffentlicht. "
        "Die Zuordnung deiner Skill-Dimensionen zu konkreten Jobprofilen erfolgt automatisch via "
        "**TF-IDF Cosine Similarity** — ein Textähnlichkeits-Verfahren, das LinkedIn-Jobtitel "
        "mit den RH-Jobprofilbezeichnungen vergleicht."
    )

    if primary_cat != "Finance & Banking":
        st.info(
            "📊 **Benchmark für diese Kategorie folgt.**\n\n"
            "Der Robert Half Gehalts-Benchmark ist aktuell nur für **Finance & Banking** verfügbar.")
        st.stop()

    with st.spinner("🔍 Jobprofil-Matching wird berechnet..."):
        rh_sim_map    = compute_rh_similarity(filepath)
        skill_dim_map = get_skill_primary_dimension(filepath)

    cluster_coverage_lookup = {gap["cluster"]: gap["coverage"] for gap in cluster_gaps}
    best_fit_roles = compute_best_fit_roles(rh_sim_map, cluster_coverage_lookup, cluster_gaps)

    with st.spinner("📊 Markt-Wert-Score wird berechnet..."):
        mws_df = compute_global_markt_wert_score(skill_score_df, rh_sim_map, skill_dim_map, importance_df)

    missing_by_lernwert = get_missing_skills_by_lernwert(missing_set, mws_df)

    # ── Persönlicher Zusammenfassungs-Banner ──────────────────────
    overall_readiness = round(sum(g["coverage"] for g in cluster_gaps) / len(cluster_gaps)) if cluster_gaps else 0
    best_role = best_fit_roles[0] if best_fit_roles else None
    level_label, level_salary = "–", 0
    if best_role:
        level_label, level_salary = estimate_level(best_role["fit_pct"], best_role["salary"])

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(
            f'<div style="background:{T["bg"]};border:1px solid {T["border2"]};border-radius:10px;'
            f'padding:14px 18px;text-align:center">'
            f'<div style="color:{T["muted"]};font-size:0.78em;margin-bottom:4px">BESTES JOBPROFIL</div>'
            f'<div style="color:{T["accent"]};font-weight:700;font-size:0.95em">'
            f'{best_role["profile"] if best_role else "–"}</div>'
            f'<div style="color:{T["green"]};font-size:1.3em;font-weight:700;margin-top:6px">'
            f'{best_role["fit_pct"] if best_role else 0}% Fit-Score</div>'
            f'<div style="color:{T["dim"]};font-size:0.72em">Anteil deiner Skills, die dieses Profil abdecken</div>'
            f'</div>', unsafe_allow_html=True)
    with col_r2:
        st.markdown(
            f'<div style="background:{T["bg"]};border:1px solid {T["border2"]};border-radius:10px;'
            f'padding:14px 18px;text-align:center">'
            f'<div style="color:{T["muted"]};font-size:0.78em;margin-bottom:4px">GESCHÄTZTES MARKTNIVEAU</div>'
            f'<div style="color:{T["yellow"]};font-weight:700;font-size:0.95em">{level_label}</div>'
            f'<div style="color:{T["green"]};font-size:1.3em;font-weight:700;margin-top:6px">'
            f'CHF {level_salary:,}</div>'
            f'<div style="color:{T["dim"]};font-size:0.72em">/ Jahr · Robert Half CH 2024</div></div>',
            unsafe_allow_html=True)
    with col_r3:
        readiness_color = "#27AE60" if overall_readiness >= 70 else ("#F1C40F" if overall_readiness >= 40 else "#E67E22")
        st.markdown(
            f'<div style="background:{T["bg"]};border:1px solid {T["border2"]};border-radius:10px;'
            f'padding:14px 18px;text-align:center">'
            f'<div style="color:{T["muted"]};font-size:0.78em;margin-bottom:4px">MARKT-READINESS</div>'
            f'<div style="color:{T["text"]};font-size:0.82em">Ø Coverage über alle Dimensionen</div>'
            f'<div style="color:{readiness_color};font-size:1.6em;font-weight:700;margin-top:6px">'
            f'{overall_readiness}%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # ── 4 Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Jetzt bewerben",
        "📈 Skills entwickeln",
        "📊 Score-Vergleich",
        "🔬 Dimensionen im Detail",
    ])

    # ── TAB 1: Jetzt bewerben ─────────────────────────────────────
    with tab1:
        # Pre-compute requirements (needed for radar + rapport)
        full_sims    = compute_full_profile_similarities(filepath)
        requirements = compute_dim_requirements(full_sims)

        # ── Kompakte Top-3 Fit-Karten ─────────────────────────────
        st.markdown(
            '<span style="font-size:0.9em;color:#6B9FD4;font-weight:700">'
            'Top Jobprofile — passend zu deinen heutigen Skills</span>',
            unsafe_allow_html=True)
        st.caption(
            "Der **Fit-Score** zeigt, wie viel % der typischen Anforderungen eines Jobprofils "
            "du bereits abdeckst — abgeleitet aus deiner Coverage je Dimension, "
            "gewichtet nach der Ähnlichkeit (Similarity) des Profils."
        )

        top3_cols = st.columns(3)
        for idx, role in enumerate(best_fit_roles[:3]):
            fit  = role["fit_pct"]; sal = role["salary"]; name = role["profile"]
            fit_color = "#27AE60" if fit >= 50 else ("#F1C40F" if fit >= 25 else "#E67E22")
            reason_parts = []
            for dd in role["dims_detail"]:
                has_str  = " · ".join(f"✅ {s}" for s in dd["has"][:2])  if dd["has"]  else ""
                miss_str = " · ".join(f"🔴 {s}" for s in dd["missing"][:1]) if dd["missing"] else ""
                skills_str = " &nbsp; ".join(filter(None, [has_str, miss_str]))
                reason_parts.append(
                    f'<div style="margin-top:4px">'
                    f'<span style="color:#6B9FD4;font-size:0.72em;font-weight:600">'
                    f'{dd["dim"]} ({dd["cov"]}%)</span>'
                    f'<div style="color:#777;font-size:0.7em">{skills_str}</div></div>'
                )
            with top3_cols[idx]:
                st.markdown(
                    f'<div style="background:{T["bg"]};border:1px solid {T["border"]};border-radius:8px;'
                    f'padding:10px 12px;height:100%">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                    f'<span style="color:{T["text"]};font-size:0.82em;font-weight:700;line-height:1.3">'
                    f'{name}</span>'
                    f'<span style="color:{fit_color};font-weight:700;font-size:1.15em;margin-left:8px">'
                    f'{fit}%</span></div>'
                    f'<div style="background:{T["progress"]};border-radius:3px;margin:6px 0 4px">'
                    f'<div style="background:{fit_color};width:{min(fit*2,100)}%;height:4px;border-radius:3px"></div></div>'
                    f'{"".join(reason_parts)}'
                    f'<div style="color:{T["dim"]};font-size:0.7em;margin-top:7px;border-top:1px solid {T["border"]};padding-top:5px">'
                    f'Entry {sal["entry"]//1000}k · Mid {sal["mid"]//1000}k · Senior {sal["senior"]//1000}k CHF'
                    f'</div></div>',
                    unsafe_allow_html=True)

        if len(best_fit_roles) > 3:
            with st.expander(f"Alle weiteren {len(best_fit_roles)-3} passenden Jobprofile"):
                more_cols = st.columns(2)
                for idx, role in enumerate(best_fit_roles[3:]):
                    fit = role["fit_pct"]; sal = role["salary"]; name = role["profile"]
                    fit_color = "#27AE60" if fit >= 50 else ("#F1C40F" if fit >= 25 else "#E67E22")
                    with more_cols[idx % 2]:
                        st.markdown(
                            f'<div style="background:{T["bg2"]};border:1px solid {T["bg3"]};border-radius:6px;'
                            f'padding:8px 12px;margin-bottom:6px">'
                            f'<div style="display:flex;justify-content:space-between">'
                            f'<span style="color:{T["text"]};font-size:0.83em">{name}</span>'
                            f'<span style="color:{fit_color};font-weight:700">{fit}%</span></div>'
                            f'<div style="color:{T["dim"]};font-size:0.72em;margin-top:3px">'
                            f'Entry {sal["entry"]//1000}k · Mid {sal["mid"]//1000}k · Senior {sal["senior"]//1000}k CHF'
                            f'</div></div>', unsafe_allow_html=True)

        # ── Interaktiver Profil-Vergleich ─────────────────────────
        st.markdown("---")
        st.markdown(
            '<span style="font-size:0.9em;color:#6B9FD4;font-weight:700">'
            'Interaktiver Profil-Vergleich</span>', unsafe_allow_html=True)
        st.caption(
            "Wähle bis zu zwei Jobprofile aus dem Robert Half Salary Guide. "
            "Der gepunktete Ring zeigt, wie wichtig jede Dimension für dieses Profil ist — "
            "die grüne Fläche zeigt, wo du heute stehst. Die Lücke dazwischen = was du noch lernen solltest."
        )

        d1_col, d2_col = st.columns(2)
        default_p1 = best_fit_roles[0]["profile"] if best_fit_roles else RH_PROFILES[0]
        with d1_col:
            sel_p1 = st.selectbox(
                "🎯 Zielprofil 1",
                RH_PROFILES,
                index=RH_PROFILES.index(default_p1),
                key="radar_p1",
                help="Wähle das Jobprofil, auf das du hinarbeitest.")
        with d2_col:
            no_comp = "— kein zweites Profil —"
            sel_p2_raw = st.selectbox(
                "🔵 Zum Vergleich (optional)",
                [no_comp] + RH_PROFILES,
                index=0,
                key="radar_p2",
                help="Optional: zweites Profil zum direkten Vergleich.")
        sel_p2 = sel_p2_raw if sel_p2_raw != no_comp else None
        active_profiles = [sel_p1] + ([sel_p2] if sel_p2 else [])

        st.plotly_chart(
            make_interactive_radar(cluster_gaps, requirements, active_profiles),
            use_container_width=True)

        # ── Lückenanalyse pro Profil ──────────────────────────────
        gap_cols = st.columns(len(active_profiles))
        for col_idx, profile in enumerate(active_profiles):
            sal      = RH_SALARY_DATA.get(profile, {})
            dim_gaps = analyse_profile_gaps(profile, requirements, cluster_gaps)
            tot_req  = sum(d["requirement"] for d in dim_gaps) or 1
            w_gap    = round(sum(d["gap"] * d["requirement"] for d in dim_gaps) / tot_req)
            w_cov    = 100 - w_gap
            prof_col = "#6B9FD4" if col_idx == 0 else "#F1C40F"
            lv_lbl, lv_sal = estimate_level(w_cov, sal) if sal else ("–", 0)

            with gap_cols[col_idx]:
                st.markdown(
                    f'<div style="background:{T["bg2"]};border:2px solid {prof_col}55;'
                    f'border-radius:10px;padding:14px 16px;margin-top:8px">'
                    f'<div style="color:{prof_col};font-size:0.85em;font-weight:700;margin-bottom:8px">'
                    f'{"🎯" if col_idx==0 else "🔵"} {profile}</div>'
                    f'<div style="display:flex;gap:16px;margin-bottom:10px">'
                    f'<div style="text-align:center">'
                    f'<div style="color:{T["muted"]};font-size:0.72em">GESAMTLÜCKE</div>'
                    f'<div style="color:{T["red"]};font-size:1.4em;font-weight:700">{w_gap}%</div></div>'
                    f'<div style="text-align:center">'
                    f'<div style="color:{T["muted"]};font-size:0.72em">DEINE COVERAGE</div>'
                    f'<div style="color:{T["green"]};font-size:1.4em;font-weight:700">{w_cov}%</div></div>'
                    f'<div style="text-align:center">'
                    f'<div style="color:{T["muted"]};font-size:0.72em">MARKTNIVEAU (CH)</div>'
                    f'<div style="color:{T["yellow"]};font-size:0.88em;font-weight:600">{lv_lbl}</div>'
                    f'<div style="color:{T["green"]};font-size:0.9em">CHF {lv_sal:,}</div></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True)

                # Progress bar
                st.markdown(
                    f'<div style="background:{T["progress"]};border-radius:4px;margin:-6px 0 10px">'
                    f'<div style="background:{T["green"]};width:{w_cov}%;height:6px;border-radius:4px"></div>'
                    f'</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<span style="font-size:0.78em;color:#6B9FD4;font-weight:600">'
                    f'Top Lernbereiche für dieses Profil:</span>', unsafe_allow_html=True)

                for d in dim_gaps[:5]:
                    if d["gap"] < 3: continue
                    gap_color = "#FF6B6B" if d["gap"] > 40 else ("#F1C40F" if d["gap"] > 15 else "#888")
                    miss_str = " · ".join(s.title() for s in d["missing"][:3])
                    has_str  = " · ".join(s.title() for s in d["has"][:3])
                    bar_req = d["requirement"]; bar_cov = d["coverage"]
                    st.markdown(
                        f'<div style="margin-bottom:8px">'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="color:#D0D8E8;font-size:0.82em">{d["dim"]}</span>'
                        f'<span style="color:{gap_color};font-size:0.8em;font-weight:600">'
                        f'Lücke {d["gap"]}%</span></div>'
                        f'<div style="position:relative;height:8px;background:#1E2A3A;border-radius:4px;margin:3px 0">'
                        f'<div style="position:absolute;background:{prof_col};opacity:0.3;'
                        f'width:{bar_req}%;height:8px;border-radius:4px"></div>'
                        f'<div style="position:absolute;background:#4AE68A;'
                        f'width:{bar_cov}%;height:8px;border-radius:4px"></div></div>'
                        f'<div style="color:#555;font-size:0.7em">'
                        f'Profil braucht: {bar_req}% · Du hast: {bar_cov}%</div>'
                        + (f'<div style="color:#4AE68A;font-size:0.72em;margin-top:2px">'
                           f'✅ {has_str}</div>' if has_str else '')
                        + (f'<div style="color:#FF6B6B;font-size:0.72em;margin-top:1px">'
                           f'🔴 Fehlend: {miss_str}</div>' if miss_str else '')
                        + '</div>',
                        unsafe_allow_html=True)

        # ── Download Rapport ──────────────────────────────────────
        st.markdown("---")
        rapport_txt = generate_rapport(
            selected_skills, missing_set, best_fit_roles,
            active_profiles, cluster_gaps, requirements, primary_cat)
        dl_col, info_col = st.columns([2, 5])
        with dl_col:
            st.download_button(
                label="📥 Rapport herunterladen (.txt)",
                data=rapport_txt,
                file_name="skillbridge_karriere_rapport.txt",
                mime="text/plain",
                use_container_width=True)
        with info_col:
            st.caption(
                "Der Rapport enthält: deine Skills, Top-Jobprofile, Lückenanalyse "
                "je Zielprofil mit fehlenden Skills und Gehaltsbänder. "
                "Ideal als Grundlage für ein Gespräch mit Career Services oder Mentor:innen.")

        with st.expander("ℹ️ Wie funktioniert das Jobprofil-Matching?"):
            st.markdown("""
**Woher kommen die Jobprofile?**
Die 22 Jobprofile stammen aus dem Robert Half Salary Guide Schweiz 2024 — einem jährlichen
Bericht über Marktgehälter für Finance & Banking (25./50./75. Perzentile = Entry/Mid/Senior).

**Wie wird die Anforderung je Dimension berechnet?**
Für jedes Jobprofil misst das Tool, wie ähnlich die LinkedIn-Jobtitel in jeder Dimension
zur Profilbezeichnung sind (TF-IDF Cosine Similarity). Die Dimension mit der höchsten
Ähnlichkeit erhält 100% Anforderung — alle anderen skalieren proportional dazu.

**Was zeigt der gepunktete Ring im Radar?**
Der Ring zeigt, wie wichtig jede Dimension für das gewählte Zielprofil ist.
Deine grüne Coverage-Fläche sollte idealerweise den Ring ausfüllen.
Je grösser die Lücke zwischen Ring und Fläche, desto mehr Lernbedarf in dieser Dimension.

**Was bedeutet der Fit-Score?**
Er zeigt, wie viel % der typischen Anforderungen du bereits abdeckst.
64% Fit = mehr als die Hälfte der Kompetenzen vorhanden — mit gezieltem Lernen marktreif.
            """)

    # ── TAB 2: Skills entwickeln ──────────────────────────────────
    with tab2:
        # Soft-Skill-Dimensionen (heuristisch)
        SOFT_DIMS_T2 = {
            "leadership", "communication", "collaboration", "teamwork",
            "problem solving", "adaptability", "time management",
            "attention to detail", "customer service", "networking",
            "interpersonal", "presentation",
        }

        st.markdown("##### 🎯 Welche Skills brauchst du für deinen Wunschjob?")
        st.caption(
            "Wähle dein Ziel-Jobprofil — die App zeigt dir, welche konkreten **Hard Skills** "
            "dir noch fehlen und wie du sie priorisieren solltest."
        )

        # ── Profil-Selektor ───────────────────────────────────────
        all_profiles_t2 = [r["profile"] for r in best_fit_roles]
        selected_target = st.selectbox(
            "🎯 Ziel-Jobprofil:",
            options=all_profiles_t2,
            index=0,
            key="target_profile_tab2",
        )

        target_role_t2 = next((r for r in best_fit_roles if r["profile"] == selected_target), None)
        fit_pct_t2 = target_role_t2["fit_pct"] if target_role_t2 else 0
        salary_t2  = target_role_t2["salary"]   if target_role_t2 else 0
        level_lbl_t2, level_sal_t2 = estimate_level(fit_pct_t2, salary_t2)

        # ── Fehlende Skills für gewähltes Profil ermitteln ────────
        t2_gaps = analyse_profile_gaps(selected_target, requirements, cluster_gaps)
        skill_map_t2: dict = {}
        for g in t2_gaps:
            if g["gap"] <= 0:
                continue
            priority = g["requirement"] * g["gap"] / 100
            is_soft  = any(sd in g["dim"].lower() for sd in SOFT_DIMS_T2)
            for skill in g["missing"]:
                if skill not in skill_map_t2 or priority > skill_map_t2[skill]["priority"]:
                    skill_map_t2[skill] = {
                        "skill":    skill,
                        "dim":      g["dim"],
                        "priority": priority,
                        "req":      g["requirement"],
                        "gap":      g["gap"],
                        "is_soft":  is_soft,
                    }

        hard_t2 = sorted([s for s in skill_map_t2.values() if not s["is_soft"]],
                         key=lambda x: x["priority"], reverse=True)
        soft_t2 = sorted([s for s in skill_map_t2.values() if s["is_soft"]],
                         key=lambda x: x["priority"], reverse=True)

        # ── Summary-Banner ────────────────────────────────────────
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(
                f'<div style="background:{T["bg"]};border:1px solid {T["border2"]};border-radius:10px;'
                f'padding:12px 16px;text-align:center">'
                f'<div style="color:{T["muted"]};font-size:0.75em;margin-bottom:4px">AKTUELLER FIT-SCORE</div>'
                f'<div style="color:{T["green"]};font-size:1.35em;font-weight:700">{fit_pct_t2}%</div>'
                f'<div style="color:{T["dim"]};font-size:0.7em">deiner heutigen Skills passen</div>'
                f'</div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(
                f'<div style="background:{T["bg"]};border:1px solid {T["border2"]};border-radius:10px;'
                f'padding:12px 16px;text-align:center">'
                f'<div style="color:{T["muted"]};font-size:0.75em;margin-bottom:4px">ZIELGEHALT</div>'
                f'<div style="color:{T["accent"]};font-size:1.35em;font-weight:700">CHF {level_sal_t2:,}</div>'
                f'<div style="color:{T["dim"]};font-size:0.7em">{level_lbl_t2} · Robert Half CH 2024</div>'
                f'</div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(
                f'<div style="background:{T["bg"]};border:1px solid {T["border2"]};border-radius:10px;'
                f'padding:12px 16px;text-align:center">'
                f'<div style="color:{T["muted"]};font-size:0.75em;margin-bottom:4px">FEHLENDE HARD SKILLS</div>'
                f'<div style="color:{T["red"]};font-size:1.35em;font-weight:700">{len(hard_t2)}</div>'
                f'<div style="color:{T["dim"]};font-size:0.7em">lernbar durch Kurse & Praxis</div>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

        # ── Hard Skills (links) + Soft Skills (rechts) ────────────
        col_hard, col_soft = st.columns([3, 2])

        with col_hard:
            st.markdown(
                f'<div style="color:{T["accent"]};font-size:0.88em;font-weight:700;margin-bottom:8px">'
                f'🔧 Fehlende Hard Skills — lernbar durch Kurse & Praxis</div>',
                unsafe_allow_html=True)
            if not hard_t2:
                st.success("Alle Hard-Skill-Dimensionen dieses Profils sind bereits abgedeckt! 🎉")
            else:
                for i, s in enumerate(hard_t2[:8]):
                    rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    rank_col  = T["rank1"] if i < 3 else (T["rank2"] if i < 6 else T["rank3"])
                    req_bar   = min(s["req"], 100)
                    gap_bar   = min(s["gap"], 100)
                    st.markdown(
                        f'<div style="margin-bottom:10px;padding:9px 12px;background:{T["bg2"]};'
                        f'border-radius:8px;border-left:3px solid {rank_col}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="color:{rank_col};font-size:0.9em;font-weight:700">'
                        f'{rank_icon} {s["skill"].title()}</span>'
                        f'<span style="background:{rank_col}22;color:{rank_col};font-size:0.7em;'
                        f'padding:2px 8px;border-radius:10px;font-weight:600">'
                        f'Priorität {s["priority"]:.0f}</span>'
                        f'</div>'
                        f'<div style="color:{T["muted"]};font-size:0.72em;margin:3px 0 7px 0">'
                        f'📐 {s["dim"]}</div>'
                        f'<div style="display:flex;gap:4px;align-items:center;margin-bottom:3px">'
                        f'<span style="color:{T["dim"]};font-size:0.68em;width:120px">Profil-Anforderung</span>'
                        f'<div style="flex:1;background:{T["progress"]};border-radius:3px;height:5px">'
                        f'<div style="background:{T["accent"]};width:{req_bar}%;height:5px;border-radius:3px">'
                        f'</div></div>'
                        f'<span style="color:{T["accent"]};font-size:0.68em;width:35px;text-align:right">'
                        f'{s["req"]}%</span>'
                        f'</div>'
                        f'<div style="display:flex;gap:4px;align-items:center">'
                        f'<span style="color:{T["dim"]};font-size:0.68em;width:120px">Deine Lücke</span>'
                        f'<div style="flex:1;background:{T["progress"]};border-radius:3px;height:5px">'
                        f'<div style="background:{T["red"]};width:{gap_bar}%;height:5px;border-radius:3px">'
                        f'</div></div>'
                        f'<span style="color:{T["red"]};font-size:0.68em;width:35px;text-align:right">'
                        f'{s["gap"]}%</span>'
                        f'</div></div>',
                        unsafe_allow_html=True)

        with col_soft:
            st.markdown(
                f'<div style="color:{T["muted"]};font-size:0.88em;font-weight:700;margin-bottom:4px">'
                f'💬 Soft Skills</div>'
                f'<div style="color:{T["dim"]};font-size:0.72em;margin-bottom:10px">'
                f'Wichtig, aber schwerer formell zu lernen — entwickelt sich durch '
                f'Erfahrung, Teamarbeit und Führungsverantwortung.</div>',
                unsafe_allow_html=True)
            if not soft_t2:
                st.success("Alle Soft-Skill-Dimensionen abgedeckt!")
            else:
                for i, s in enumerate(soft_t2[:6]):
                    rank_col = T["accent"] if i < 2 else T["muted"]
                    gap_bar  = min(s["gap"], 100)
                    st.markdown(
                        f'<div style="margin-bottom:8px;padding:8px 11px;background:{T["bg2"]};'
                        f'border-radius:8px;border-left:2px solid {rank_col}">'
                        f'<div style="color:{rank_col};font-size:0.85em;font-weight:600">'
                        f'{i+1}. {s["skill"].title()}</div>'
                        f'<div style="color:{T["muted"]};font-size:0.7em;margin:2px 0 5px 0">'
                        f'{s["dim"]}</div>'
                        f'<div style="display:flex;gap:4px;align-items:center">'
                        f'<span style="color:{T["dim"]};font-size:0.66em;width:55px">Lücke</span>'
                        f'<div style="flex:1;background:{T["progress"]};border-radius:3px;height:4px">'
                        f'<div style="background:{rank_col};width:{gap_bar}%;height:4px;border-radius:3px">'
                        f'</div></div>'
                        f'<span style="color:{rank_col};font-size:0.66em;width:35px;text-align:right">'
                        f'{s["gap"]}%</span>'
                        f'</div></div>',
                        unsafe_allow_html=True)

        # ── Lern-Roadmap ──────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"##### 🗺️ Deine Lern-Roadmap für: *{selected_target}*")
        st.caption(
            "Die fehlenden Hard Skills in drei Phasen eingeteilt — "
            "nach Priorität (Profil-Anforderung × deine Lücke)."
        )

        roadmap_skills = hard_t2[:10]
        if roadmap_skills:
            phase1_rm = roadmap_skills[:3]
            phase2_rm = roadmap_skills[3:6]
            phase3_rm = roadmap_skills[6:10]
            phase_defs = [
                ("🚀 Phase 1 — Sofort starten",   "Höchste Priorität. Zahlen sich am schnellsten aus.", T["rank1"],  T["lv_bg"], T["lv_border"], phase1_rm),
                ("📚 Phase 2 — Nächste 3 Monate", "Ergänzt und vertieft Phase 1.",                      T["rank2"],  T["bg3"],   T["border"],    phase2_rm),
                ("🎯 Phase 3 — Mittelfristig",    "Spezialisierung & Differenzierung.",                  T["accent"], T["bg2"],   T["border2"],   phase3_rm),
            ]
            road_cols = st.columns(3)
            icons_rm  = ["🥇", "🥈", "🥉", "4.", "5."]
            for col_i, (label, desc, color, bg, border, skills) in enumerate(phase_defs):
                with road_cols[col_i]:
                    items_html = "".join(
                        f'<div style="padding:7px 0;border-bottom:1px solid {T["border"]}">'
                        f'<div style="color:{T["text"]};font-size:0.86em;font-weight:600">'
                        f'{icons_rm[i]} {s["skill"].title()}</div>'
                        f'<div style="color:{T["muted"]};font-size:0.7em;margin-top:1px">'
                        f'📐 {s["dim"]}</div>'
                        f'<div style="color:{T["dim"]};font-size:0.68em">'
                        f'Anforderung {s["req"]}% · Lücke {s["gap"]}%</div>'
                        f'</div>'
                        for i, s in enumerate(skills)
                    )
                    st.markdown(
                        f'<div style="background:{bg};border:2px solid {border};'
                        f'border-radius:8px;padding:12px 14px">'
                        f'<div style="color:{color};font-size:0.88em;font-weight:700;margin-bottom:4px">'
                        f'{label}</div>'
                        f'<div style="color:{T["muted"]};font-size:0.72em;margin-bottom:10px">'
                        f'{desc}</div>'
                        + items_html + '</div>',
                        unsafe_allow_html=True)
        else:
            st.success("Alle Hard Skills für dieses Profil sind bereits abgedeckt! 🎉")

    # ── TAB 3: Score-Vergleich ────────────────────────────────────
    with tab3:
        st.markdown("##### Skill-Value-Score vs. Markt-Wert-Score — was ist der Unterschied?")
        st.caption(
            "Beide Scores messen den Wert eines Skills — aber mit unterschiedlicher Datenbasis. "
            "Wo SVS und MWS stark abweichen, lohnt sich ein genauer Blick: "
            "Das sind Skills, die LinkedIn anders bewertet als der Schweizer Gehaltsmarkt."
        )

        if mws_df.empty:
            st.warning("Score-Vergleich nicht verfügbar — bitte RH-Matching prüfen.")
        else:
            mws_display = mws_df.copy()
            mws_display["Status"] = mws_display["skill"].apply(
                lambda s: "✅ Vorhanden" if s in selected_set else "🔴 Fehlend")

            bar_col, table_col = st.columns([5, 4])

            with bar_col:
                st.markdown(
                    '<span style="font-size:0.82em;color:#6B9FD4;font-weight:600">'
                    'Top 15 Skills: SVS vs. Markt-Wert-Score (MWS)</span>',
                    unsafe_allow_html=True)
                st.caption(
                    "Blau = Skill-Value-Score (LinkedIn-Daten) · Grün = Markt-Wert-Score (RH CH 2024). "
                    "Helle Farbe = Skill fehlt dir · Dunkle Farbe = Skill hast du bereits."
                )
                bar_fig = make_svs_mws_comparison_chart(mws_df, selected_set, top_n=15)
                st.plotly_chart(bar_fig, use_container_width=True)

            with table_col:
                st.markdown(
                    '<span style="font-size:0.82em;color:#6B9FD4;font-weight:600">'
                    'Grösste Abweichungen zwischen SVS und MWS</span>',
                    unsafe_allow_html=True)
                st.caption(
                    "Positive Differenz: CH-Markt bezahlt mehr als LinkedIn schätzt. "
                    "Negative Differenz: LinkedIn-Nachfrage übertrifft das CH-Gehaltsignal."
                )
                top_diff = mws_display.reindex(
                    mws_display["diff"].abs().sort_values(ascending=False).index
                ).head(12).reset_index(drop=True)

                for _, row in top_diff.iterrows():
                    diff = row["diff"]
                    diff_color = T["green"] if diff > 5 else (T["red"] if diff < -5 else T["muted"])
                    diff_icon  = "▲" if diff > 5 else ("▼" if diff < -5 else "━")
                    status_dot = "🟢" if "Vorhanden" in row["Status"] else "🔴"
                    rh_k = row["rh_mid"] / 1000
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:5px 8px;margin-bottom:3px;background:{T["bg2"]};border-radius:5px;'
                        f'border:1px solid {T["border"]}">'
                        f'<div>'
                        f'<span style="color:{T["text"]};font-size:0.85em">{status_dot} {row["skill"].title()}</span>'
                        f'<div style="color:{T["muted"]};font-size:0.7em">'
                        f'SVS {row["svs"]:.0f} · MWS {row["mws"]:.0f} · CHF {rh_k:.0f}k</div>'
                        f'</div>'
                        f'<span style="color:{diff_color};font-weight:700;font-size:0.9em">'
                        f'{diff_icon} {abs(diff):.0f}</span>'
                        f'</div>',
                        unsafe_allow_html=True)

            # Interpretations-Expander
            with st.expander("ℹ️ Wie soll ich die Abweichungen interpretieren?"):
                st.markdown("""
**MWS > SVS (grüne Pfeile ▲):**
Der Schweizer Markt bezahlt für diesen Skill mehr, als LinkedIn-Daten vermuten lassen.
→ Strategischer Vorteil: Weniger Konkurrenz, aber hohe Entlohnung. Besonders interessant für
   Kandidaten, die sich vom Mainstream abheben wollen.

**SVS > MWS (rote Pfeile ▼):**
LinkedIn zeigt hohe Nachfrage, aber das CH-Gehalts-Potenzial ist moderat.
→ Hygiene-Skill: Wird erwartet, aber macht dich nicht einzigartig. Trotzdem unbedingt vorhanden.

**Wenig Abweichung (━):**
Beide Quellen sind konsistent — robustes Signal.
→ Diese Skills sind sowohl nachgefragt als auch gut bezahlt. Klar priorisieren.

*Hinweis: Die Differenz kann auch durch unterschiedliche geographische Abdeckung entstehen
(LinkedIn global vs. RH Schweiz).*
                """)

    # ── TAB 4: Jobprofile im Detail ───────────────────────────────
    with tab4:
        st.markdown("##### Jobprofile und ihre Skill-Dimensionen")
        st.caption(
            "Für jedes Jobprofil aus dem Robert Half Salary Guide siehst du, welche Skill-Dimensionen "
            "entscheidend sind — und wie gut du sie bereits abdeckst. Sortiert nach deinem Fit-Score."
        )

        fit_by_profile = {r["profile"]: r["fit_pct"] for r in best_fit_roles}
        profiles_sorted_t4 = sorted(RH_PROFILES, key=lambda p: -fit_by_profile.get(p, 0))

        for profile_t4 in profiles_sorted_t4:
            sal_t4      = RH_SALARY_DATA.get(profile_t4, {})
            fit_t4      = fit_by_profile.get(profile_t4, 0)
            dim_gaps_t4 = analyse_profile_gaps(profile_t4, requirements, cluster_gaps)
            if not dim_gaps_t4: continue

            tot_req_t4 = sum(d["requirement"] for d in dim_gaps_t4) or 1
            w_gap_t4   = round(sum(d["gap"] * d["requirement"] for d in dim_gaps_t4) / tot_req_t4)
            w_cov_t4   = 100 - w_gap_t4
            fit_icon   = "🟢" if fit_t4 >= 50 else ("🟡" if fit_t4 >= 25 else "🔴")
            sal_str_t4 = (f" · Entry {sal_t4['entry']//1000}k / Mid {sal_t4['mid']//1000}k / "
                          f"Senior {sal_t4['senior']//1000}k CHF") if sal_t4 else ""

            with st.expander(
                f"{fit_icon} {profile_t4}  ·  Fit {fit_t4}%  ·  Lücke {w_gap_t4}%{sal_str_t4}",
                expanded=False
            ):
                # Coverage progress bar
                fit_color_t4 = T["green"] if w_cov_t4 >= 50 else (T["yellow"] if w_cov_t4 >= 25 else T["red"])
                st.markdown(
                    f'<div style="background:{T["progress"]};border-radius:4px;margin-bottom:10px">'
                    f'<div style="background:{fit_color_t4};width:{w_cov_t4}%;height:7px;border-radius:4px"></div>'
                    f'</div>', unsafe_allow_html=True)

                for d in dim_gaps_t4[:8]:
                    if d["requirement"] < 5: continue
                    gap_col_t4 = T["red"] if d["gap"] > 40 else (T["yellow"] if d["gap"] > 15 else T["green"])
                    miss_str_t4 = " · ".join(s.title() for s in d["missing"][:3])
                    has_str_t4  = " · ".join(s.title() for s in d["has"][:3])
                    st.markdown(
                        f'<div style="margin-bottom:9px;padding:8px 10px;'
                        f'background:{T["bg2"]};border-radius:6px;border-left:3px solid {gap_col_t4}">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                        f'<span style="color:{T["text"]};font-size:0.84em;font-weight:600">{d["dim"]}</span>'
                        f'<span style="color:{gap_col_t4};font-size:0.78em;font-weight:600">Lücke {d["gap"]}%</span>'
                        f'</div>'
                        f'<div style="position:relative;height:7px;background:{T["progress"]};border-radius:3px;margin:3px 0">'
                        f'<div style="position:absolute;background:{T["accent"]};opacity:0.25;'
                        f'width:{d["requirement"]}%;height:7px;border-radius:3px"></div>'
                        f'<div style="position:absolute;background:{T["green"]};'
                        f'width:{d["coverage"]}%;height:7px;border-radius:3px"></div></div>'
                        f'<div style="color:{T["dim"]};font-size:0.7em">'
                        f'Anforderung: {d["requirement"]}% · Coverage: {d["coverage"]}%</div>'
                        + (f'<div style="color:{T["green"]};font-size:0.72em;margin-top:2px">✅ {has_str_t4}</div>'
                           if has_str_t4 else '')
                        + (f'<div style="color:{T["red"]};font-size:0.72em;margin-top:1px">🔴 Fehlend: {miss_str_t4}</div>'
                           if miss_str_t4 else '')
                        + '</div>',
                        unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "📋 **Datenquellen:** LinkedIn-Stellenanzeigen Finance & Banking (49k Jobs) für Schritte 1–5 · "
        "Robert Half Salary Guide Schweiz 2024 für Schritt 6 (25. Pz. = Entry · 50. Pz. = Mid · 75. Pz. = Senior). "
        "Zürich-Aufschlag +7% nicht eingerechnet. Keine Gewähr auf Vollständigkeit oder Aktualität."
    )
