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

st.set_page_config(page_title="SkillBridge", page_icon="🎓", layout="wide")

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

DATASETS = {
    "Consulting & Strategy": "ConsultingData_clustered_slim.csv",
    "Finance & Banking":     "Banking_clustered_slim.csv",
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

# Robert Half Daten
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


# ── Hilfsfunktionen ────────────────────────────────────────────

def classify_skill(skill):
    s = skill.lower()
    if any(h in s for h in HARD_KEYWORDS): return "hard"
    if any(soft in s for soft in SOFT_KEYWORDS): return "soft"
    return "hard"

def is_valid_skill(skill):
    return not any(kw in skill.lower() for kw in EXCLUDE_SKILL_KEYWORDS)

def demand_label(pct):
    """Kurze Beschreibung der Nachfragestärke eines Skills."""
    if pct >= 60:   return "Sehr stark nachgefragt"
    elif pct >= 40: return "Stark nachgefragt"
    elif pct >= 20: return "Häufig nachgefragt"
    else:           return "Selten nachgefragt"

def get_dim_badges(gap, top_skills_dict, skill_score_df, global_sal_median):
    """Berechnet Demand-, Karriere- und Gehaltsbadge für eine Dimension."""
    all_skills = gap["has"] + gap["missing"]
    demands = [top_skills_dict.get(s, 0) for s in all_skills]
    avg_demand = sum(demands) / len(demands) if demands else 0
    if avg_demand >= 45:   demand_badge = "🔥 Sehr gefragt"
    elif avg_demand >= 22: demand_badge = "📊 Gefragt"
    else:                  demand_badge = "🔍 Nischenbereich"

    dim_df = skill_score_df[skill_score_df["skill"].isin(all_skills)]
    avg_career = dim_df["career_ratio"].mean() if not dim_df.empty else 0
    if avg_career >= 0.5:   career_badge = "🚀 Karrierehebel"
    elif avg_career >= 0.3: career_badge = "📈 Wachstumspotenzial"
    else:                   career_badge = "➡️ Einstiegsbereich"

    sal_ratio = gap["sal_median"] / global_sal_median if global_sal_median > 0 else 1.0
    if sal_ratio >= 1.1:   sal_badge = "💰 Überdurchschnittlich"
    elif sal_ratio >= 0.9: sal_badge = "💰 Marktüblich"
    else:                  sal_badge = "💰 Unterdurchschnittlich"

    return demand_badge, career_badge, sal_badge


def make_dimension_skill_radar(gap, top_skills_dict, selected_set, T):
    """Radar: Top-Skills einer Dimension vs. Nutzer-Coverage."""
    all_skills = gap["has"] + gap["missing"]
    display_skills = all_skills[:8]
    if len(display_skills) < 3:
        return None
    labels = [s.title() for s in display_skills]
    raw_vals = [max(top_skills_dict.get(s, 5), 5) for s in display_skills]
    max_val = max(raw_vals)
    market_vals = [round(v / max_val * 100) for v in raw_vals]
    user_vals   = [market_vals[i] if s in selected_set else 0
                   for i, s in enumerate(display_skills)]
    lc = labels + [labels[0]]
    mc = market_vals + [market_vals[0]]
    uc = user_vals   + [user_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=mc, theta=lc, fill="toself", name="Marktanforderung",
        fillcolor="rgba(107,159,212,0.12)",
        line=dict(color="#6B9FD4", width=1.5, dash="dot"),
        hovertemplate="%{theta}<br>Marktgewicht: %{r}<extra></extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=uc, theta=lc, fill="toself", name="Deine Skills",
        fillcolor="rgba(74,230,138,0.22)",
        line=dict(color="#4AE68A", width=2),
        hovertemplate="%{theta}<br>Vorhanden<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#444",
                            tickfont=dict(size=8), gridcolor="#2A3040", showticklabels=False),
            angularaxis=dict(color="#888", gridcolor="#2A3040"),
            bgcolor="rgba(0,0,0,0)"
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=10),
                    orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T.get("font_clr", "#aaa"), size=10),
        height=430,
        margin=dict(l=60, r=60, t=30, b=65)
    )
    return fig


def compute_dim_overlap_requirements(cluster_profiles_dict):
    """Für non-Banking: dim-to-dim Skill-Overlap als Requirements für den Radar."""
    dims = {
        name: set(p["top_skills"])
        for name, p in cluster_profiles_dict.items()
        if name not in EXCLUDE_CLUSTERS and p["top_skills"]
    }
    reqs = {}
    for profile_dim, profile_skills in dims.items():
        dim_reqs = {}
        for other_dim, other_skills in dims.items():
            union   = profile_skills | other_skills
            overlap = len(profile_skills & other_skills) / len(union) if union else 0
            dim_reqs[other_dim] = round(overlap * 100)
        max_r = max(dim_reqs.values()) if dim_reqs else 1
        reqs[profile_dim] = {d: round(v / max_r * 100) for d, v in dim_reqs.items()}
    return reqs


def make_factor_importance_chart(skills_with_scores, factor, T):
    """Horizontales Balkendiagramm: Top-Skills nach gewähltem Faktor."""
    if not skills_with_scores:
        return None
    names = [s["skill"].title() for s in skills_with_scores]
    vals  = [round(s["val"], 1)  for s in skills_with_scores]
    bar_color = (T["accent"]  if "Nachfrage"  in factor
                 else (T["green"] if "Karriere" in factor else T["yellow"]))
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=bar_color,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
    x_labels = {
        "📊 Nachfrage":        "% der Jobinserate",
        "🚀 Karrierewachstum": "RF Feature Importance Senior (x1000)",
        "💰 Gehaltspremium":   "Gehaltseffekt in CHF (t-Test vs. Median)",
    }
    fig.update_layout(
        xaxis_title=x_labels.get(factor, ""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T.get("font_clr", "#aaa"), size=11),
        height=max(220, len(names) * 48 + 80),
        margin=dict(l=10, r=90, t=10, b=40),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed",
                   tickfont=dict(color=T.get("font_clr", "#aaa"))),
        xaxis=dict(gridcolor=T.get("grid", "#2A3040"),
                   tickfont=dict(color=T.get("font_clr", "#aaa"))),
    )
    return fig




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
SOURCE_MAP  = {"Consulting & Strategy": "Consulting", "Finance & Banking": "Banking"}

@st.cache_data
def train_rf_senior(filepath, top_n=80):
    """Random Forest: abhängige Variable = Senior-Position (binär)."""
    df = load_dataset(filepath)
    df = df[~df["job_cluster"].isin(EXCLUDE_CLUSTERS)].copy()
    df["is_senior"] = df["job_level"].apply(
        lambda x: 1 if any(kw in str(x).lower()
                           for kw in ["senior", "director", "executive", "lead", "manager", "head", "vp"])
        else 0
    )
    if df["is_senior"].sum() < 20:
        return pd.DataFrame(columns=["skill", "importance"])
    counter = Counter()
    for skills in df["skills_parsed"]: counter.update(set(skills))
    feature_skills = [s for s, _ in counter.most_common(top_n) if is_valid_skill(s)]
    X = pd.DataFrame([{s: int(s in set(sk)) for s in feature_skills}
                      for sk in df["skills_parsed"]])
    y = df["is_senior"].values
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X, y)
    return pd.DataFrame({"skill": feature_skills,
                         "importance": rf.feature_importances_}
                       ).sort_values("importance", ascending=False).reset_index(drop=True)

@st.cache_data
def train_rf_salary(primary_cat, top_n=80):
    """Random Forest Regression: abhängige Variable = Jahresgehalt (CHF)."""
    from sklearn.ensemble import RandomForestRegressor
    try:
        df = pd.read_csv(SALARY_FILE)
        source = SOURCE_MAP.get(primary_cat, "Consulting")
        df = df[df["_source"] == source].dropna(subset=["salary_mid"]).copy()
        if len(df) < 50:
            return pd.DataFrame(columns=["skill", "importance"])
        def parse(entry):
            try: return ast.literal_eval(str(entry))
            except: return []
        df["skills_parsed"] = df["skills_filtered"].apply(parse)
        counter = Counter()
        for skills in df["skills_parsed"]: counter.update(set(skills))
        feature_skills = [s for s, _ in counter.most_common(top_n) if is_valid_skill(s)]
        X = pd.DataFrame([{s: int(s in set(sk)) for s in feature_skills}
                          for sk in df["skills_parsed"]])
        y = df["salary_mid"].values
        rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        return pd.DataFrame({"skill": feature_skills,
                             "importance": rf.feature_importances_}
                            ).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["skill", "importance"])

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

@st.cache_data
def compute_rh_similarity(filepath):
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
    df = load_dataset(filepath)
    dimensions = [d for d in df["job_cluster"].unique() if d not in EXCLUDE_CLUSTERS]
    dim_texts = {dim: " ".join(df[df["job_cluster"] == dim]["job_title"].dropna().tolist())
                 for dim in dimensions}
    if not dim_texts: return {}
    all_texts = list(dim_texts.values()) + RH_PROFILES
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(all_texts)
    n = len(dim_texts)
    sim_matrix = cosine_similarity(tfidf[:n], tfidf[n:])
    result = {}
    for j, profile in enumerate(RH_PROFILES):
        result[profile] = {
            dim: round(float(sim_matrix[i, j]), 4)
            for i, dim in enumerate(dim_texts.keys())
        }
    return result

def compute_best_fit_roles(rh_sim_map, cluster_coverage_lookup, cluster_gaps):
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
    if mws_df.empty: return []
    result = mws_df[mws_df["skill"].isin(missing_set)].copy()
    return result.sort_values("lernwert", ascending=False).to_dict("records")

def estimate_level(coverage_pct, salary):
    if coverage_pct >= 70:   return "Senior Level", salary["senior"]
    elif coverage_pct >= 40: return "Mid Level",    salary["mid"]
    else:                    return "Entry Level",  salary["entry"]

def make_interactive_radar(cluster_gaps, requirements, selected_profiles, T=None):
    if T is None:
        T = {"font_clr": "#aaaaaa", "muted": "#888888", "grid": "#2A3040"}
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
            name=f"Ziel: {profile}",
            fillcolor=f"rgba({rgb},0.10)",
            line=dict(color=hex_c, width=2, dash=dash),
            hovertemplate="%{theta}<br>Anforderung: %{r}%<extra>" + profile + "</extra>"))
    fig.add_trace(go.Scatterpolar(
        r=cc, theta=lc, fill="toself", name="Deine Skills",
        fillcolor="rgba(74,230,138,0.22)",
        line=dict(color="#4AE68A", width=2.5),
        hovertemplate="%{theta}<br>Deine Coverage: %{r}%<extra></extra>"))
    axis_color   = T.get("muted", "#888")
    grid_color   = T.get("grid",  "#2A3040")
    font_color   = T.get("font_clr", "#aaa")
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=axis_color,
                            tickcolor=axis_color, gridcolor=grid_color,
                            tickfont=dict(size=9, color=font_color)),
            angularaxis=dict(color=axis_color, gridcolor=grid_color,
                             tickfont=dict(color=font_color)),
            bgcolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(0,0,0,0.35)", font=dict(size=11, color=font_color),
                    orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, size=11), height=430,
        margin=dict(l=60, r=60, t=30, b=50))
    return fig

def compute_dim_requirements(all_sims):
    requirements = {}
    for profile, dim_sims in all_sims.items():
        if not dim_sims:
            requirements[profile] = {}
            continue
        max_s = max(dim_sims.values()) or 1.0
        requirements[profile] = {dim: round(s / max_s * 100) for dim, s in dim_sims.items()}
    return requirements

def analyse_profile_gaps(profile, requirements, cluster_gaps):
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
    import datetime
    today = datetime.date.today().strftime("%d.%m.%Y")
    lines = [
        "=" * 62,
        "SKILLBRIDGE: PERSOENLICHER KARRIERE-RAPPORT",
        f"Bereich: {primary_cat}",
        f"Datum:   {today}",
        "=" * 62, "",
        "DEINE AKTUELLEN SKILLS",
        "-" * 40,
    ]
    for s in sorted(selected_skills):
        lines.append(f"  {s.title()}")
    lines += ["", "FEHLENDE SKILLS (Top 10 nach Lernwert)", "-" * 40]
    for s in sorted(list(missing_set))[:10]:
        lines.append(f"  {s.title()}")
    lines += ["", "TOP PASSENDE JOBPROFILE", "-" * 40]
    for role in best_fit_roles[:5]:
        sal = role["salary"]
        lines.append(f"  {role['fit_pct']:3d}%  {role['profile']}")
        lines.append(f"        Entry CHF {sal['entry']:,} / Mid CHF {sal['mid']:,} / Senior CHF {sal['senior']:,}")
    lines.append("")
    for profile in selected_profiles:
        sal      = RH_SALARY_DATA.get(profile, {})
        dim_gaps = analyse_profile_gaps(profile, requirements, cluster_gaps)
        tot_req  = sum(d["requirement"] for d in dim_gaps) or 1
        w_gap    = round(sum(d["gap"] * d["requirement"] for d in dim_gaps) / tot_req)
        lines += [f"PROFIL-ANALYSE: {profile}", "-" * 40]
        if sal:
            lines.append(f"  Marktgehalt CH 2024 (Robert Half):")
            lines.append(f"    Entry CHF {sal['entry']:,} / Mid CHF {sal['mid']:,} / Senior CHF {sal['senior']:,}")
        lines.append(f"  Gesamtluecke zu diesem Profil: {w_gap}%")
        lines += ["", "  PRIORITAERE LERNBEREICHE:"]
        for d in dim_gaps[:6]:
            if d["gap"] < 3: continue
            miss_str = ", ".join(s.title() for s in d["missing"][:3])
            lines.append(f"  [{d['gap']:3d}% Luecke]  {d['dim']}")
            lines.append(f"              Anforderung: {d['requirement']}% / Coverage: {d['coverage']}%")
            if miss_str:
                lines.append(f"              Fehlende Skills: {miss_str}")
        lines.append("")
    lines += [
        "=" * 62,
        "DATENQUELLEN",
        "  LinkedIn Finance & Banking: 49.000 Stellenanzeigen",
        "  Robert Half Salary Guide Schweiz 2024",
        "  Keine Gewaehr auf Vollstaendigkeit oder Aktualitaet.",
        "=" * 62,
    ]
    return "\n".join(lines)

def make_svs_mws_comparison_chart(mws_df, selected_set, top_n=15):
    df = mws_df.copy()
    df["Status"] = df["skill"].apply(lambda s: "Vorhanden" if s in selected_set else "Fehlend")
    top = df.nlargest(top_n, "mws").sort_values("mws", ascending=True)
    fig = go.Figure()
    colors_svs = ["#4A7CC0" if s == "Fehlend" else "#2A5A8A" for s in top["Status"]]
    colors_mws = ["#4AE68A" if s == "Fehlend" else "#27AE60" for s in top["Status"]]
    fig.add_trace(go.Bar(
        name="Skill-Value-Score (LinkedIn)", y=top["skill"].str.title(),
        x=top["svs"], orientation="h", marker_color=colors_svs, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>SVS: %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Bar(
        name="Markt-Wert-Score (RH CH 2024)", y=top["skill"].str.title(),
        x=top["mws"], orientation="h", marker_color=colors_mws, opacity=0.85,
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

def render_dimension_card(gap, skill_score_df, selected_set, global_sal_median):
    """Karte fuer Schritt 5 (Lernpfad)."""
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
            if modifier == "soft_penalty":    mod_str = "Soft-Skill Penalty x 0.4"
            elif modifier == "basic_penalty": mod_str = "Basic-Skill Penalty x 0.2"
            elif modifier == "sig_boost":     mod_str = "Signifikanz-Boost x 1.25"
            else:                             mod_str = "Kein Modifier"
            tooltip = (f"Skill-Value-Score: {score}/100&#10;"
                       f"Nachfrage (30%): {demand}% der Jobs&#10;"
                       f"Karriere  (40%): {career}x Senior-Anteil&#10;"
                       f"Gehalt    (30%): {salary:.2f}x Median&#10;{mod_str}")
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
            f"Prioritaet = Gehalts-Potenzial x Verbleibende Luecke&#10;"
            f"Coverage = {gap['has_score']:.0f} / {gap['total_score']:.0f} = {coverage}%"
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
            f'Prioritaet: {priority_pct}</span></div>', unsafe_allow_html=True)
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


# ══════════════════════════════════════════════════════════════
# UI START
# ══════════════════════════════════════════════════════════════

if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

_hdr_r_col, = st.columns([1])
with _hdr_r_col:
    pass

_, _toggle_col = st.columns([10, 1])
with _toggle_col:
    _lbl = "☀️" if not st.session_state.light_mode else "🌙"
    if st.button(_lbl, key="theme_toggle", use_container_width=True):
        st.session_state.light_mode = not st.session_state.light_mode

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
    .stApp, .stAppViewContainer, section.main,
    .block-container, [data-testid="stHeader"] { background: #F8FAFD !important; }
    .stApp p, .stApp span, .stApp div, .stApp li,
    .stApp label, .stApp small, .stApp strong, .stApp em,
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span,
    [data-testid="stCheckbox"] p, [data-testid="stToggle"] p,
    [data-baseweb="checkbox"] span,
    [class*="st-emotion"] p, [class*="st-emotion"] span { color: #31333F !important; }
    h1, h2, h3, h4, h5, h6 { color: #1A2A40 !important; }
    .stTabs [data-baseweb="tab"] { color: #555 !important; }
    .stTabs [aria-selected="true"] { color: #1A2A40 !important; font-weight:700 !important; }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span { color: #1A2A40 !important; }
    [data-baseweb="select"], [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div, [data-baseweb="input"],
    [data-baseweb="input"] > div { background: #FFFFFF !important; color: #31333F !important; }
    [data-baseweb="select"] *, [data-baseweb="input"] * { color: #31333F !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] * { background: #FFFFFF !important; color: #31333F !important; }
    [data-testid="stButton"] > button { background-color: #FFFFFF !important; color: #1A2A40 !important; border: 1px solid #C5D5E8 !important; }
    [data-testid="stButton"] > button:hover { background-color: #E8EEF8 !important; border-color: #6B9FD4 !important; }
    [data-testid="stAlert"] p { color: #1A2A40 !important; }
    [data-testid="stSidebar"] { background: #EEF2F8 !important; }
    [data-testid="stDownloadButton"] button { background: #E8EEF8 !important; color: #1A2A40 !important; border-color: #9AAFC4 !important; }
    html body [data-testid="stVerticalBlockBorderWrapper"] { border: 1.5px solid #6090B8 !important; border-radius: 10px !important; background-color: #FFFFFF !important; }
    [data-testid="stCheckbox"] label { color: #1A2A40 !important; font-weight: 500 !important; }
    [data-testid="stCheckbox"] span { color: #1A2A40 !important; }
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


# ══════════════════════════════════════════════════════════════
# SCHRITT 1: Bereich wählen
# ══════════════════════════════════════════════════════════════

with st.container(border=True):
    st.subheader("Schritt 1: In welchem Bereich möchtest du arbeiten?")
    selected_categories = []
    cols = st.columns(len(DATASETS))
    for i, cat in enumerate(DATASETS.keys()):
        if cols[i].checkbox(f"**{cat}**", key=f"cat_{cat}"): selected_categories.append(cat)
    if not selected_categories:
        st.info("Wähle oben mindestens einen Jobbereich aus.")
if not selected_categories:
    st.stop()

primary_cat = selected_categories[0]
filepath    = os.path.join(BASE_PATH, DATASETS[primary_cat])

with st.spinner("Datensatz und Modell werden geladen..."):
    top_skills_dict   = get_top_skills(filepath)
    cluster_profiles  = get_cluster_profiles(filepath)
    importance_df     = train_random_forest(filepath)
    importance_senior = train_rf_senior(filepath)
    importance_salary = train_rf_salary(primary_cat)
    salary_impact_df  = compute_salary_impact(primary_cat)
    cluster_salary_df = compute_cluster_salary(primary_cat)
    skill_score_df    = compute_skill_value_score(filepath, primary_cat)


# ══════════════════════════════════════════════════════════════
# SCHRITT 2: Skills auswählen (Checkboxen)
# ══════════════════════════════════════════════════════════════

st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 2: Welche Skills bringst du bereits mit?")
    st.caption("Wähle alle Skills aus, die auf dich zutreffen.")

    all_skills_set = set(
        s for profile in cluster_profiles.values()
        for s in profile["top_skills"] if is_valid_skill(s)
    )
    # Merge audit/auditing → "auditing"
    if "audit" in all_skills_set:
        all_skills_set.discard("audit")
        all_skills_set.add("auditing")
    # Merge business → "business administration" (spezifischerer Begriff)
    if "business" in all_skills_set:
        all_skills_set.discard("business")
        all_skills_set.add("business administration")
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

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
    n_sel = len(selected_skills)
    if n_sel == 0:
        st.info("Du kannst auch ohne vorhandene Skills direkt zur Analyse gehen.")
    elif n_sel < 10:
        st.warning(
            f"**{n_sel} Skills ausgewählt.** Für aussagekräftige Ergebnisse empfehlen wir "
            f"mindestens 10 Skills. Du kannst aber auch jetzt schon weitermachen."
        )
    else:
        st.success(f"**{n_sel} Skills ausgewählt.** Bereit zur Analyse.")

    if st.button("Weiter zur Analyse", type="primary"):
        st.session_state["analyse_ready"] = True

if not st.session_state.get("analyse_ready", False):
    st.stop()


# ── Abgeleitete Daten ─────────────────────────────────────────
cluster_skills_all = set(
    s for profile in cluster_profiles.values()
    for s in profile["top_skills"] if is_valid_skill(s)
)
top_skills_dict = {k: v for k, v in top_skills_dict.items() if k in cluster_skills_all}
selected_set    = set(selected_skills)
all_top_set     = set(top_skills_dict.keys())
missing_set     = all_top_set - selected_set

global_sal_median = 100000.0
cluster_sal_lookup = {}
if not cluster_salary_df.empty:
    global_sal_median  = cluster_salary_df["median_salary"].median()
    cluster_sal_lookup = dict(zip(cluster_salary_df["cluster"], cluster_salary_df["median_salary"]))

def get_skill_score_val(s):
    row = skill_score_df[skill_score_df["skill"] == s]
    return float(row["score"].values[0]) if not row.empty else 10.0

cluster_gaps = []
for cluster_name, profile in cluster_profiles.items():
    top = profile["top_skills"]
    if not top: continue
    has     = [s for s in top if s in selected_set]
    missing = [s for s in top if s not in selected_set]
    total_score = sum(get_skill_score_val(s) for s in top)
    has_score   = sum(get_skill_score_val(s) for s in has)
    coverage    = round(has_score / total_score * 100) if total_score > 0 else 0
    sal_median  = cluster_sal_lookup.get(cluster_name, global_sal_median)
    sal_norm    = sal_median / global_sal_median if global_sal_median > 0 else 1.0
    priority_score = sal_norm * (coverage / 100)
    cluster_gaps.append({
        "cluster": cluster_name, "coverage": coverage,
        "has": has, "missing": missing, "pct": profile["pct"],
        "sal_median": sal_median, "sal_norm": round(sal_norm, 2),
        "priority_score": round(priority_score, 3),
        "total_score": round(total_score, 1), "has_score": round(has_score, 1),
    })
cluster_gaps.sort(key=lambda x: x["priority_score"], reverse=True)

# Vergleichs-Banner Daten
def compute_banner_data():
    own_df  = skill_score_df[skill_score_df["skill"].isin(selected_set)]
    miss_df = skill_score_df[skill_score_df["skill"].isin(missing_set)].nlargest(20, "score")
    def smean(df, col): return round(df[col].mean(), 1) if not df.empty else 0
    return {
        "own_demand":  smean(own_df,  "demand_pct"),
        "miss_demand": smean(miss_df, "demand_pct"),
        "own_career":  smean(own_df,  "career_ratio"),
        "miss_career": smean(miss_df, "career_ratio"),
        "own_salary":  smean(own_df,  "salary_ratio"),
        "miss_salary": smean(miss_df, "salary_ratio"),
    }

banner = compute_banner_data()

# Tab 2 Dimensionsauswahl init
if "tab2_dim" not in st.session_state and cluster_gaps:
    st.session_state.tab2_dim = cluster_gaps[0]["cluster"]


# ══════════════════════════════════════════════════════════════
# SCHRITT 3: Deine Top-Skills + Dimensionserklärung
# ══════════════════════════════════════════════════════════════

st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Schritt 3: Deine Top-Skills im Überblick")

    if not selected_skills:
        st.info("Keine Skills ausgewählt. Die Analyse zeigt dir, was du lernen kannst.")
    else:
        my_sorted = sorted(selected_skills,
                           key=lambda s: top_skills_dict.get(s, 0), reverse=True)[:5]
        n = len(my_sorted); chunk = max(1, -(-n // 2))
        col1, col2 = st.columns(2)
        for i, skill in enumerate(my_sorted):
            pct = top_skills_dict.get(skill, 0)
            if pct >= 60:   bar_color = T["green"]
            elif pct >= 40: bar_color = T["accent"]
            elif pct >= 20: bar_color = T["yellow"]
            else:           bar_color = T["muted"]
            col = col1 if i < chunk else col2
            with col:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;'
                    f'padding:5px 0;border-bottom:1px solid {T["border"]}">'
                    f'<span style="color:{T["text"]};font-size:0.88em;min-width:155px">'
                    f'{skill.title()}</span>'
                    f'<div style="flex:1;background:{T["progress"]};border-radius:3px;height:6px">'
                    f'<div style="background:{bar_color};width:{min(pct*1.2,100):.0f}%;'
                    f'height:6px;border-radius:3px"></div></div>'
                    f'<span style="color:{bar_color};font-size:0.8em;font-weight:600;'
                    f'min-width:38px;text-align:right">{pct}%</span>'
                    f'<span style="color:{T["muted"]};font-size:0.72em;min-width:210px">'
                    f'in {pct}% der Jobinserate nachgefragt</span>'
                    f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ANALYSE-TABS
# ══════════════════════════════════════════════════════════════

st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

# ── Profil-Matching vor den Tabs berechnen (wird in Tab 1 und Tab 2 gebraucht) ──
salary_source_label = (
    "Robert Half Salary Guide Schweiz 2024"
    if primary_cat == "Finance & Banking"
    else "Aus LinkedIn-Inseraten extrahiert"
)

with st.spinner("Jobprofil-Matching wird berechnet..."):
    if primary_cat == "Finance & Banking":
        rh_sim_map    = compute_rh_similarity(filepath)
        full_sims     = compute_full_profile_similarities(filepath)
        requirements  = compute_dim_requirements(full_sims)
        skill_dim_map = get_skill_primary_dimension(filepath)
        cluster_coverage_lookup = {g["cluster"]: g["coverage"] for g in cluster_gaps}
        best_fit_roles = compute_best_fit_roles(rh_sim_map, cluster_coverage_lookup, cluster_gaps)
        mws_df = compute_global_markt_wert_score(
            skill_score_df, rh_sim_map, skill_dim_map, importance_df)
    else:
        requirements  = compute_dim_overlap_requirements(cluster_profiles)
        best_fit_roles = [
            {
                "profile":     g["cluster"],
                "fit_pct":     g["coverage"],
                "salary":      {
                    "entry":  round(g["sal_median"] * 0.85),
                    "mid":    round(g["sal_median"]),
                    "senior": round(g["sal_median"] * 1.2),
                },
                "dims_detail": [{"dim": g["cluster"], "cov": g["coverage"],
                                  "has": g["has"][:4], "missing": g["missing"][:2]}],
            }
            for g in sorted(cluster_gaps, key=lambda x: x["coverage"], reverse=True)
        ]
        rh_sim_map = {}
        mws_df     = pd.DataFrame()

all_profile_names = [r["profile"] for r in best_fit_roles]
no_comp = "Kein zweites Profil"

# Session-State Defaults für Tab-Verknüpfung
if "radar_p1" not in st.session_state and best_fit_roles:
    st.session_state["radar_p1"] = best_fit_roles[0]["profile"]
if "radar_p1_sel" not in st.session_state and best_fit_roles:
    st.session_state["radar_p1_sel"] = best_fit_roles[0]["profile"]
if "radar_p2_sel" not in st.session_state:
    st.session_state["radar_p2_sel"] = no_comp

st.markdown(f"""<style>
[data-testid="stTabs"] {{
    border: 1.5px solid {T["border2"]};
    border-radius: 12px;
    padding: 0 16px 20px 16px;
    background-color: {T["bg2"]};
}}
[data-testid="stTabs"] [data-testid="stTabBar"] {{
    border-radius: 10px 10px 0 0;
    padding-top: 6px;
}}
</style>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Jetzt bewerben", "📈 Skills entwickeln", "🔬 Skill-Hebel"])


# ══════════════════════════════════════════════════════════════
# TAB 1: JETZT BEWERBEN
# ══════════════════════════════════════════════════════════════

with tab1:

    # --- TOP 3 KARTEN ---
    st.markdown(
        f'<div style="color:{T["text"]};font-size:1em;font-weight:700;margin-bottom:4px">'
        f'Deine Top-Matches</div>'
        f'<div style="color:{T["muted"]};font-size:0.78em;margin-bottom:14px">'
        f'Klicke eine Karte an, um das Profil im Radar zu analysieren. '
        f'Gehaltsquelle: {salary_source_label}.</div>',
        unsafe_allow_html=True)

    top3_cols = st.columns(3)
    for idx, role in enumerate(best_fit_roles[:3]):
        coverage = role["fit_pct"]
        sal      = role["salary"]
        name     = role["profile"]
        is_sel   = st.session_state.get("radar_p1") == name

        if coverage >= 60:   cov_color = "#27AE60"
        elif coverage >= 35: cov_color = "#F1C40F"
        else:                cov_color = "#E67E22"

        border_c = T["accent"] if is_sel else T["border"]
        border_w = "2px" if is_sel else "1px"
        bg_c     = T["bg3"] if is_sel else T["bg2"]

        dim_skills = []
        for dd in role["dims_detail"]:
            dim_skills.extend(dd["has"] + dd["missing"])
        avg_d = sum(top_skills_dict.get(s, 0) for s in dim_skills) / len(dim_skills) if dim_skills else 0
        demand_txt = "🔥 Stark nachgefragt" if avg_d >= 45 else ("📊 Gefragt" if avg_d >= 22 else "🔍 Nische")
        sal_txt = f"💰 CHF {sal['mid']:,}"

        with top3_cols[idx]:
            rank_icon = ["🥇", "🥈", "🥉"][idx]
            st.markdown(
                f'<div style="background:{bg_c};border:{border_w} solid {border_c};'
                f'border-radius:10px;padding:14px 16px;margin-bottom:8px">'
                f'<div style="display:flex;align-items:flex-start;gap:6px;margin-bottom:10px">'
                f'<span style="font-size:1.2em">{rank_icon}</span>'
                f'<span style="color:{T["text"]};font-size:0.88em;font-weight:700;line-height:1.35">'
                f'{name}</span></div>'
                f'<div style="color:{T["muted"]};font-size:0.72em;margin-bottom:3px">DEINE SKILL-COVERAGE</div>'
                f'<div style="color:{cov_color};font-size:2em;font-weight:800;line-height:1.1">{coverage}%</div>'
                f'<div style="background:{T["progress"]};border-radius:3px;margin:6px 0 10px">'
                f'<div style="background:{cov_color};width:{coverage}%;height:6px;border-radius:3px"></div></div>'
                f'<div style="display:flex;flex-direction:column;gap:4px">'
                f'<span style="color:{T["accent"]};font-size:0.78em">{demand_txt}</span>'
                f'<span style="color:{T["yellow"]};font-size:0.78em">{sal_txt} Marktgehalt</span>'
                f'</div></div>',
                unsafe_allow_html=True)

            btn_lbl = "✓ Im Radar" if is_sel else "Im Radar anzeigen"
            if st.button(btn_lbl, key=f"tab1_card_{name}",
                         use_container_width=True,
                         type="primary" if is_sel else "secondary"):
                st.session_state["radar_p1"]     = name
                st.session_state["radar_p1_sel"] = name
                st.rerun()

    # --- INTERAKTIVER RADAR ---
    st.markdown("<div style='margin:18px 0 6px 0'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{T["text"]};font-size:1em;font-weight:700;margin-bottom:4px">'
        f'Interaktiver Profil-Vergleich</div>'
        f'<div style="color:{T["muted"]};font-size:0.78em;margin-bottom:10px">'
        f'Grüne Fläche: dein Skill-Profil. Gepunktete Ringe: Anforderungen der gewählten Profile. '
        f'Du kannst zwei Profile gleichzeitig vergleichen.</div>',
        unsafe_allow_html=True)

    d1c, d2c = st.columns(2)
    with d1c:
        p1_idx = all_profile_names.index(st.session_state.get("radar_p1_sel", all_profile_names[0])) \
                 if st.session_state.get("radar_p1_sel") in all_profile_names else 0
        sel_p1 = st.selectbox("🎯 Zielprofil 1", all_profile_names,
                               index=p1_idx, key="radar_p1_sel")
    with d2c:
        sel_p2_raw = st.selectbox("🔵 Vergleichsprofil (optional)",
                                   [no_comp] + all_profile_names, key="radar_p2_sel")
    sel_p2 = sel_p2_raw if sel_p2_raw != no_comp else None
    active_profiles = [sel_p1] + ([sel_p2] if sel_p2 else [])

    if sel_p1 != st.session_state.get("radar_p1"):
        st.session_state["radar_p1"] = sel_p1

    # Radar
    st.plotly_chart(
        make_interactive_radar(cluster_gaps, requirements, active_profiles, T=T),
        use_container_width=True)

    # ── 3-Schritt-Roadmap ──────────────────────────────────────
    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{T["text"]};font-size:0.95em;font-weight:700;margin-bottom:4px">'
        f'Dein 3-Schritte-Plan für: {sel_p1}</div>',
        unsafe_allow_html=True)

    if primary_cat == "Finance & Banking":
        dim_gaps_road = analyse_profile_gaps(sel_p1, requirements, cluster_gaps)
    else:
        sel_gap_road  = next((g for g in cluster_gaps if g["cluster"] == sel_p1), None)
        dim_gaps_road = [{"dim": sel_p1, "gap": 100 - sel_gap_road["coverage"],
                          "missing": sel_gap_road["missing"], "has": sel_gap_road["has"],
                          "requirement": 100, "coverage": sel_gap_road["coverage"]}
                         ] if sel_gap_road else []

    top_missing_road = []
    for d in sorted(dim_gaps_road, key=lambda x: x["gap"], reverse=True):
        for s in d["missing"]:
            if s not in top_missing_road and classify_skill(s) == "hard":
                top_missing_road.append(s)
            if len(top_missing_road) >= 9:
                break
        if len(top_missing_road) >= 9:
            break

    road_steps = [
        ("🚀 Schritt 1: Sofort starten",
         "Höchste Priorität. Diese Skills haben den grössten Markteffekt.",
         T["rank1"], top_missing_road[:3]),
        ("📚 Schritt 2: Nächste 3 Monate",
         "Vertieft und ergänzt Schritt 1.",
         T["rank2"], top_missing_road[3:6]),
        ("🎯 Schritt 3: Mittelfristig",
         "Spezialisierung und Differenzierung.",
         T["accent"], top_missing_road[6:9]),
    ]

    road_cols = st.columns(3)
    for ci, (label, desc, color, skills) in enumerate(road_steps):
        with road_cols[ci]:
            items = "".join(
                f'<div style="padding:5px 0;border-bottom:1px solid {T["border"]}">'
                f'<span style="color:{T["text"]};font-size:0.85em;font-weight:600">'
                f'{["🥇","🥈","🥉","4.","5.","6."][i]} {s.title()}</span>'
                f'<div style="color:{T["muted"]};font-size:0.7em">'
                f'in {top_skills_dict.get(s,0)}% der Inserate</div>'
                f'</div>'
                for i, s in enumerate(skills)
            ) if skills else f'<div style="color:{T["muted"]};font-size:0.78em">Alle Skills vorhanden.</div>'
            st.markdown(
                f'<div style="background:{T["bg2"]};border:2px solid {color}55;'
                f'border-radius:8px;padding:12px 14px">'
                f'<div style="color:{color};font-size:0.85em;font-weight:700;margin-bottom:4px">'
                f'{label}</div>'
                f'<div style="color:{T["muted"]};font-size:0.72em;margin-bottom:8px">{desc}</div>'
                + items + '</div>', unsafe_allow_html=True)

    if primary_cat == "Finance & Banking":
        st.markdown("---")
        rapport_txt = generate_rapport(
            selected_skills, missing_set, best_fit_roles,
            active_profiles, cluster_gaps, requirements, primary_cat)
        dl_col, info_col = st.columns([2, 5])
        with dl_col:
            st.download_button("📥 Rapport herunterladen (.txt)",
                                data=rapport_txt, file_name="skillbridge_rapport.txt",
                                mime="text/plain", use_container_width=True)
        with info_col:
            st.caption("Persönlicher Karriere-Rapport mit Skills, Top-Profilen und Lückenanalyse.")


# ══════════════════════════════════════════════════════════════
# TAB 2: SKILLS ENTWICKELN (mit SVS)
# ══════════════════════════════════════════════════════════════

with tab2:

    current_p1     = st.session_state.get("radar_p1_sel", all_profile_names[0] if all_profile_names else "")
    current_p2_raw = st.session_state.get("radar_p2_sel", no_comp)
    current_p2     = current_p2_raw if current_p2_raw != no_comp else None

    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<div style="color:{T["text"]};font-size:1.5em;font-weight:800;margin-bottom:6px">'
        f'Dein Lernpfad</div>'
        f'<div style="color:{T["muted"]};font-size:0.92em;line-height:1.6;max-width:680px">'
        f'Wo du mit wenigen zusätzlichen Skills die grössten Fortschritte '
        f'in Bezug auf Gehalt und Karriere erzielst.</div>'
        f'</div>',
        unsafe_allow_html=True)

    with st.expander("Wie wird die Coverage berechnet?", expanded=False):
        st.markdown(f"""
**Coverage** zeigt, wie viel Prozent einer Job-Dimension du bereits abdeckst.

Die Berechnung ist **SVS-gewichtet**: ein seltener High-Value-Skill zählt mehr als ein weit verbreiteter Basic-Skill.

**Formel:**

Coverage = Summe SVS (deine Skills in der Dimension) / Summe SVS (alle Skills der Dimension) x 100

**Beispiel:** Eine Dimension hat 6 Skills mit einem Gesamt-SVS von 320. Du hast 2 davon, mit einem Gesamt-SVS von 140. Coverage = 140 / 320 = 44%.

**Was bedeutet das in der Praxis?** Wer einen hochgewichteten Skill wie Python (SVS 90) besitzt, hat eine höhere Coverage als jemand der drei Basic-Skills (je SVS 15) hat, selbst wenn er mehr Skills abdeckt. Das ist bewusst so: der Score belohnt Relevanz, nicht Quantität.
        """)

    # Verknüpfte Dimensionen aus Tab 1
    if primary_cat == "Finance & Banking":
        linked_dims = set()
        for profile in [current_p1] + ([current_p2] if current_p2 else []):
            profile_reqs = requirements.get(profile, {})
            top3 = sorted(profile_reqs.items(), key=lambda x: x[1], reverse=True)[:3]
            linked_dims.update(d for d, r in top3 if r > 0)
        if not linked_dims:
            p1_role = next((r for r in best_fit_roles if r["profile"] == current_p1), None)
            if p1_role:
                linked_dims = {dd["dim"] for dd in p1_role["dims_detail"]}
    else:
        linked_dims = {current_p1}
        if current_p2: linked_dims.add(current_p2)

    profile_label = f"**{current_p1}**"
    if current_p2: profile_label += f" und **{current_p2}**"
    st.markdown(
        f'<div style="background:{T["bg3"]};border:1px solid {T["accent"]};'
        f'border-radius:8px;padding:8px 14px;margin-bottom:14px">'
        f'<span style="color:{T["accent"]};font-size:0.8em">🔗 Verknüpft mit Tab 1: '
        f'{profile_label}</span>'
        f'<span style="color:{T["muted"]};font-size:0.75em;margin-left:10px">'
        f'Relevante Dimensionen: {", ".join(linked_dims) if linked_dims else "keine"}</span>'
        f'</div>', unsafe_allow_html=True)

    linked_gaps = [g for g in cluster_gaps if g["cluster"] in linked_dims]
    if not linked_gaps:
        linked_gaps = cluster_gaps[:2]

    def get_svs_row(s):
        r = skill_score_df[skill_score_df["skill"] == s]
        return r.iloc[0] if not r.empty else None

    # ── SVS Erklärbox ──────────────────────────────────────────
    with st.expander("ℹ️ Was ist der Skill-Value-Score (SVS)?", expanded=False):
        st.markdown(f"""
Der **Skill-Value-Score (0 bis 100)** kombiniert drei gleichzeitig gewichtete Dimensionen:

| Dimension | Gewicht | Messung |
|---|---|---|
| 📊 Nachfrage | 30% | % der Jobinserate die diesen Skill verlangen |
| 🚀 Karrierewachstum | 40% | Anteil Senior-Positionen mit diesem Skill |
| 💰 Gehaltspremium | 30% | Gehaltsindex relativ zum Branchenmedian |

**Modifikatoren:**
- Hard Skill mit statistisch signifikantem Gehaltseffekt: x1.25
- Soft Skill: x0.4 (schwerer formell nachweisbar)
- Basic Skill (Excel, Word etc.): x0.2 (erwartet, kein Differenzierungsmerkmal)

Der dreiteilige Balken unter jedem Skill zeigt die drei Komponenten direkt: **blau** = Nachfrage, **grün** = Karriere, **gelb** = Gehalt.
        """)

    # ── Skills pro Dimension ───────────────────────────────────
    for gap in linked_gaps:
        coverage = gap["coverage"]
        if coverage == 0:   cov_c = "#C0392B"
        elif coverage < 40: cov_c = "#E67E22"
        elif coverage < 70: cov_c = "#F1C40F"
        else:               cov_c = "#27AE60"

        # Hervorgehobener Dimensions-Titel
        st.markdown(
            f'<div style="background:{T["bg3"]};border:2px solid {T["accent"]};'
            f'border-radius:10px;padding:10px 16px;margin:16px 0 8px 0">'
            f'<div style="display:flex;align-items:center;gap:14px">'
            f'<span style="color:{T["accent"]};font-size:1.15em;font-weight:800">'
            f'{gap["cluster"]}</span>'
            f'<span style="color:{cov_c};font-size:0.88em;font-weight:600">'
            f'{coverage}% Coverage</span>'
            f'<span style="color:{T["muted"]};font-size:0.78em">'
            f'{len(gap["missing"])} fehlend / {len(gap["has"])} vorhanden</span>'
            f'</div>'
            f'<div style="background:{T["progress"]};border-radius:3px;margin-top:6px">'
            f'<div style="background:{cov_c};width:{coverage}%;height:5px;border-radius:3px">'
            f'</div></div></div>',
            unsafe_allow_html=True)

        missing_hard = sorted([s for s in gap["missing"] if classify_skill(s) == "hard"],
                               key=lambda s: get_skill_score_val(s), reverse=True)
        missing_soft = sorted([s for s in gap["missing"] if classify_skill(s) == "soft"],
                               key=lambda s: get_skill_score_val(s), reverse=True)
        has_skills   = sorted(gap["has"], key=lambda s: get_skill_score_val(s), reverse=True)

        col_miss, col_has = st.columns([3, 2])

        with col_miss:
            if missing_hard:
                st.markdown(
                    f'<div style="color:{T["accent"]};font-size:0.82em;font-weight:700;'
                    f'margin-bottom:8px">🔧 Fehlende Hard Skills, nach SVS priorisiert</div>',
                    unsafe_allow_html=True)
                for i, s in enumerate(missing_hard):
                    row = get_svs_row(s)
                    if row is None: continue
                    svs    = int(row["score"])
                    demand = round(row["demand_pct"])
                    career = row["career_ratio"]
                    sal_r  = row["salary_ratio"]
                    modifier = row.get("modifier", "none")

                    if modifier == "sig_boost":    mod_txt = "✅ Stat. Gehaltsboost ×1.25"
                    elif modifier == "soft_penalty": mod_txt = "⚠️ Soft-Skill ×0.4"
                    elif modifier == "basic_penalty": mod_txt = "📉 Basic-Skill ×0.2"
                    else: mod_txt = ""

                    if career >= 0.5:   career_b = f'<span style="color:{T["green"]};font-size:0.7em">🚀 Senior-Hebel</span>'
                    elif career >= 0.3: career_b = f'<span style="color:{T["yellow"]};font-size:0.7em">📈 Karrierepotenzial</span>'
                    else:               career_b = f'<span style="color:{T["muted"]};font-size:0.7em">➡️ Einstieg</span>'

                    if sal_r >= 1.1:   sal_b = f'<span style="color:{T["green"]};font-size:0.7em">💰 Über Median</span>'
                    elif sal_r >= 0.9: sal_b = f'<span style="color:{T["muted"]};font-size:0.7em">💰 Marktüblich</span>'
                    else:              sal_b = f'<span style="color:{T["dim"]};font-size:0.7em">💰 Unter Median</span>'

                    rank_c = T["rank1"] if i == 0 else (T["rank2"] if i == 1 else T["rank3"])
                    rank_i = "🥇" if i == 0 else ("🥈" if i == 1 else f"{i+1}.")

                    # Segmentierte SVS-Balken: drei farbige Teile
                    d_contrib = round(min(demand * 0.30, 100) * svs / max(svs, 1), 1)
                    k_contrib = round(min(career * 100 * 0.40, 100) * svs / max(svs, 1), 1)
                    g_contrib = round(svs - d_contrib - k_contrib, 1)
                    g_contrib = max(0, g_contrib)
                    # Breiten proportional zu SVS (Gesamtbalken = svs% von 100px)
                    total_w = svs
                    d_w = round(d_contrib / 100 * 100, 1) if svs > 0 else 0
                    k_w = round(k_contrib / 100 * 100, 1) if svs > 0 else 0
                    g_w = max(0, round(total_w - d_w - k_w, 1))

                    st.markdown(
                        f'<div style="margin-bottom:9px;padding:9px 12px;background:{T["bg2"]};'
                        f'border-radius:8px;border-left:3px solid {rank_c}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'margin-bottom:6px">'
                        f'<span style="color:{rank_c};font-size:0.9em;font-weight:700">'
                        f'{rank_i} {s.title()}</span>'
                        f'<span style="color:{rank_c};font-size:0.82em;font-weight:800">'
                        f'SVS {svs}/100</span></div>'
                        # Segmentierter Balken
                        f'<div style="display:flex;background:{T["progress"]};'
                        f'border-radius:3px;height:8px;margin-bottom:5px;overflow:hidden">'
                        f'<div style="background:{T["accent"]};width:{d_w}%;height:8px" '
                        f'title="Nachfrage: {demand}%"></div>'
                        f'<div style="background:{T["green"]};width:{k_w}%;height:8px" '
                        f'title="Karriere: {round(career*100)}%"></div>'
                        f'<div style="background:{T["yellow"]};width:{g_w}%;height:8px" '
                        f'title="Gehalt: {round(sal_r,2)}x Median"></div>'
                        f'</div>'
                        # Legende unter dem Balken
                        f'<div style="display:flex;gap:10px;margin-bottom:5px">'
                        f'<span style="color:{T["accent"]};font-size:0.68em">📊 {demand}%</span>'
                        f'<span style="color:{T["green"]};font-size:0.68em">🚀 {round(career*100)}%</span>'
                        f'<span style="color:{T["yellow"]};font-size:0.68em">💰 {round(sal_r,2)}x</span>'
                        + (f'<span style="color:{T["yellow"]};font-size:0.68em">{mod_txt}</span>' if mod_txt else '')
                        + f'</div>'
                        f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
                        f'{career_b}{sal_b}'
                        + f'</div></div>', unsafe_allow_html=True)

            if missing_soft:
                st.markdown(
                    f'<div style="color:{T["muted"]};font-size:0.82em;font-weight:700;'
                    f'margin:10px 0 6px 0">💬 Fehlende Soft Skills</div>', unsafe_allow_html=True)
                for s in missing_soft[:4]:
                    row    = get_svs_row(s)
                    svs    = int(row["score"]) if row is not None else 0
                    demand = round(row["demand_pct"]) if row is not None else 0
                    tooltip = f"SVS: {svs}/100 | in {demand}% der Inserate"
                    st.markdown(
                        f'<div style="margin-bottom:5px;padding:6px 10px;background:{T["bg2"]};'
                        f'border-radius:6px;border-left:2px solid {T["muted"]}">'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="color:{T["text"]};font-size:0.85em">{s.title()}</span>'
                        f'<span style="color:{T["muted"]};font-size:0.72em">'
                        f'in {demand}% der Inserate</span></div>'
                        f'<div title="{tooltip}" style="cursor:help">'
                        f'<div style="background:{T["progress"]};border-radius:3px;margin-top:4px">'
                        f'<div style="background:{T["muted"]};width:{svs}%;height:4px;border-radius:3px">'
                        f'</div></div></div></div>', unsafe_allow_html=True)

        with col_has:
            st.markdown(
                f'<div style="color:{T["green"]};font-size:0.82em;font-weight:700;'
                f'margin-bottom:8px">✅ Du bringst bereits mit</div>', unsafe_allow_html=True)
            if has_skills:
                for s in has_skills:
                    row = get_svs_row(s)
                    svs = int(row["score"]) if row is not None else 0
                    pct = top_skills_dict.get(s, 0)
                    demand = round(row["demand_pct"]) if row is not None else 0
                    career = row["career_ratio"] if row is not None else 0
                    sal_r  = row["salary_ratio"] if row is not None else 1
                    tooltip = f"SVS: {svs}/100 | Nachfrage: {demand}% | Karriere: {career:.2f}x | Gehalt: {sal_r:.2f}x"
                    st.markdown(
                        f'<div style="padding:6px 10px;margin-bottom:4px;background:{T["bg"]};'
                        f'border-radius:6px;border-left:3px solid {T["green"]}">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                        f'<span style="color:{T["text"]};font-size:0.83em">{s.title()}</span>'
                        f'<span style="color:{T["green"]};font-size:0.75em;font-weight:700">'
                        f'{svs}/100</span></div>'
                        f'<div title="{tooltip}" style="cursor:help">'
                        f'<div style="background:{T["progress"]};border-radius:3px">'
                        f'<div style="background:{T["green"]};width:{svs}%;height:4px;border-radius:3px">'
                        f'</div></div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span style="color:{T["muted"]};font-size:0.78em">'
                    f'Noch keine Skills dieser Dimension vorhanden.</span>',
                    unsafe_allow_html=True)

        st.markdown(f'<hr style="margin:10px 0;border-color:{T["border"]}">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: SKILL-HEBEL
# ══════════════════════════════════════════════════════════════

with tab3:

    current_p1_t3     = st.session_state.get("radar_p1_sel", all_profile_names[0] if all_profile_names else "")
    current_p2_t3_raw = st.session_state.get("radar_p2_sel", no_comp)
    current_p2_t3     = current_p2_t3_raw if current_p2_t3_raw != no_comp else None

    # Verknüpfte Dimensionen
    if primary_cat == "Finance & Banking":
        linked_dims_t3 = set()
        for profile in [current_p1_t3] + ([current_p2_t3] if current_p2_t3 else []):
            profile_reqs = requirements.get(profile, {})
            top3 = sorted(profile_reqs.items(), key=lambda x: x[1], reverse=True)[:3]
            linked_dims_t3.update(d for d, r in top3 if r > 0)
        if not linked_dims_t3:
            p1_role = next((r for r in best_fit_roles if r["profile"] == current_p1_t3), None)
            if p1_role:
                linked_dims_t3 = {dd["dim"] for dd in p1_role["dims_detail"]}
    else:
        linked_dims_t3 = {current_p1_t3}
        if current_p2_t3: linked_dims_t3.add(current_p2_t3)

    linked_gaps_t3 = [g for g in cluster_gaps if g["cluster"] in linked_dims_t3]
    if not linked_gaps_t3:
        linked_gaps_t3 = cluster_gaps[:2]

    profile_label_t3 = f"**{current_p1_t3}**"
    if current_p2_t3: profile_label_t3 += f" und **{current_p2_t3}**"

    st.markdown(
        f'<div style="color:{T["text"]};font-size:1em;font-weight:700;margin-bottom:2px">'
        f'Skill-Hebel maximieren</div>'
        f'<div style="color:{T["muted"]};font-size:0.78em;margin-bottom:10px">'
        f'Welcher fehlende Skill hat den grössten messbaren Effekt auf Nachfrage, '
        f'Karrierewachstum oder Gehalt? Verknüpft mit: {profile_label_t3}</div>',
        unsafe_allow_html=True)

    factor_choice = st.radio(
        "Zu maximierende Dimension:",
        ["📊 Nachfrage", "🚀 Karrierewachstum", "💰 Gehaltspremium"],
        horizontal=True, key="tab3_factor"
    )

    factor_context = {
        "📊 Nachfrage": (
            "**Methode: Direktmessung** über alle Jobinserate im gewählten Bereich. "
            "Der Balken zeigt in wie vielen Prozent der Stellen dieser Skill explizit verlangt wird. "
            "Wert 65 = 65 von 100 Stellen fordern diesen Skill."
        ),
        "🚀 Karrierewachstum": (
            "**Methode: Random Forest Klassifikation** (150 Bäume, balanced class weights). "
            "Abhängige Variable: Senior-Position (1) vs. Nicht-Senior (0). "
            "Der Balken zeigt die Feature Importance dieses Skills (x1000 skaliert). "
            "Hohe Werte bedeuten: der Skill unterscheidet Senior- von Junior-Profilen besonders stark."
        ),
        "💰 Gehaltspremium": (
            "**Hauptwert: Statistisch signifikanter Gehaltseffekt in CHF** (t-Test, p < 0.05). "
            "Der Wert zeigt den Unterschied zwischen dem Median-Gehalt von Stellen MIT und OHNE diesen Skill. "
            "Beispiel: CHF +18'500 bedeutet, Stellen die diesen Skill verlangen zahlen im Median CHF 18'500 mehr. "
            "Zusatzinfo: RF Feature Importance aus dem Gehalts-Random-Forest als Bestätigung."
        ),
    }

    st.markdown(
        f'<div style="background:{T["bg2"]};border:1px solid {T["border"]};'
        f'border-radius:8px;padding:10px 14px;margin-bottom:14px">'
        f'<div style="color:{T["muted"]};font-size:0.78em;line-height:1.6">'
        f'{factor_context.get(factor_choice, "")}</div></div>',
        unsafe_allow_html=True)

    all_missing_t3 = set(s for g in linked_gaps_t3 for s in g["missing"])

    factor_scores_t3 = []

    if "Nachfrage" in factor_choice:
        # Direktzählung: fehlende Skills aus linked_dims, sortiert nach demand
        for s in all_missing_t3:
            row = skill_score_df[skill_score_df["skill"] == s]
            if row.empty: continue
            val = float(row.iloc[0]["demand_pct"])
            if val > 0:
                factor_scores_t3.append({"skill": s, "val": val})

    elif "Karriere" in factor_choice:
        # RF Senior: ranke ALLE skills im RF-Modell, dann filter auf fehlende
        rf_src = importance_senior if not importance_senior.empty else importance_df
        for _, rf_row in rf_src.iterrows():
            s = rf_row["skill"]
            if s not in all_missing_t3: continue
            val = float(rf_row["importance"]) * 1000  # skalieren für lesbare Zahlen
            if val > 0:
                factor_scores_t3.append({"skill": s, "val": val})

    else:  # Gehalt: t-Test CHF als Hauptwert, RF als Bestätigung
        # Baue ein Lookup: skill -> (chf_diff, rf_importance)
        sal_lookup = {}
        if not salary_impact_df.empty:
            for _, sal_row in salary_impact_df.iterrows():
                sk = sal_row["skill"]
                diff = float(sal_row["salary_diff"])
                if sk not in sal_lookup or diff > sal_lookup[sk][0]:
                    sal_lookup[sk] = (diff, 0.0)

        # RF importance als Bestätigung hinzufügen
        rf_src = importance_salary if not importance_salary.empty else pd.DataFrame()
        if not rf_src.empty:
            for _, rf_row in rf_src.iterrows():
                sk = rf_row["skill"]
                imp = float(rf_row["importance"]) * 1000
                if sk in sal_lookup:
                    sal_lookup[sk] = (sal_lookup[sk][0], imp)

        for s in all_missing_t3:
            if s in sal_lookup:
                chf_diff, rf_imp = sal_lookup[s]
                if chf_diff > 0:
                    factor_scores_t3.append({"skill": s, "val": chf_diff, "rf_imp": rf_imp})
            else:
                # Fallback: Gehaltsindex aus skill_score_df
                row = skill_score_df[skill_score_df["skill"] == s]
                if row.empty: continue
                sal_r = float(row.iloc[0]["salary_ratio"])
                chf_est = max(0, (sal_r - 1.0) * global_sal_median)
                if chf_est > 500:
                    imp_row = rf_src[rf_src["skill"] == s] if not rf_src.empty else pd.DataFrame()
                    rf_imp = float(imp_row["importance"].values[0]) * 1000 if not imp_row.empty else 0.0
                    factor_scores_t3.append({"skill": s, "val": chf_est, "rf_imp": rf_imp})

    top_factor_t3 = sorted(factor_scores_t3, key=lambda x: x["val"], reverse=True)[:8]

    if top_factor_t3:
        chart_t3 = make_factor_importance_chart(top_factor_t3, factor_choice, T)
        if chart_t3:
            st.plotly_chart(chart_t3, use_container_width=True)

        st.markdown(
            f'<div style="color:{T["muted"]};font-size:0.75em;font-weight:600;'
            f'margin-bottom:10px">TOP 3 SKILLS MIT GRÖSSTEM HEBEL</div>',
            unsafe_allow_html=True)

        top3_t3_cols = st.columns(3)
        for fi, item in enumerate(top_factor_t3[:3]):
            s   = item["skill"]
            val = item["val"]
            row = skill_score_df[skill_score_df["skill"] == s]
            svs = int(row["score"].values[0]) if not row.empty else 0
            demand = round(row.iloc[0]["demand_pct"]) if not row.empty else 0
            # Find which dimension this skill belongs to
            dim_name = next(
                (g["cluster"] for g in linked_gaps_t3 if s in g["missing"]), ""
            )
            rank_c = T["rank1"] if fi == 0 else (T["rank2"] if fi == 1 else T["rank3"])
            rank_i = ["🥇", "🥈", "🥉"][fi]

            if "Nachfrage" in factor_choice:
                val_display = f"{val:.0f}%"
                val_label   = "der Stellen verlangen diesen Skill"
                val_interp  = f"Grundlage für {val:.0f}% aller Stellen in diesem Bereich"
                bar_w = min(val * 1.2, 100)
                rf_note = ""
            elif "Karriere" in factor_choice:
                val_display = f"{val:.2f}"
                val_label   = "RF Feature Importance (Senior, x1000)"
                val_interp  = "Hoher Wert: dieser Skill unterscheidet Senior- von Junior-Profilen"
                bar_w = min(val / max(top_factor_t3[0]["val"], 1) * 100, 100)
                rf_note = ""
            else:
                # Gehalt: val = CHF-Differenz
                val_display = f"CHF +{val:,.0f}".replace(",", "'")
                val_label   = "statistischer Gehaltseffekt vs. Dimensions-Median"
                val_interp  = (f"Stellen die diesen Skill verlangen zahlen im Median "
                               f"CHF {val:,.0f} mehr".replace(",", "'"))
                bar_w = min(val / max(top_factor_t3[0]["val"], 1) * 100, 100)
                rf_imp = item.get("rf_imp", 0)
                rf_note = (f'<div style="color:{T["dim"]};font-size:0.68em;margin-top:4px">'
                           f'RF Importance: {rf_imp:.2f} (Gehalts-Modell)</div>') if rf_imp > 0 else ""

            with top3_t3_cols[fi]:
                st.markdown(
                    f'<div style="background:{T["bg2"]};border:2px solid {rank_c};'
                    f'border-radius:10px;padding:14px">'
                    f'<div style="color:{rank_c};font-size:0.9em;font-weight:700;margin-bottom:4px">'
                    f'{rank_i} {s.title()}</div>'
                    f'<div style="color:{T["muted"]};font-size:0.72em;margin-bottom:8px">'
                    f'Dimension: {dim_name}</div>'
                    f'<div style="color:{rank_c};font-size:1.5em;font-weight:800;line-height:1.1;'
                    f'margin-bottom:4px">{val_display}</div>'
                    f'<div style="color:{T["muted"]};font-size:0.72em;margin-bottom:6px">'
                    f'{val_label}</div>'
                    f'<div style="background:{T["progress"]};border-radius:3px;margin-bottom:8px">'
                    f'<div style="background:{rank_c};width:{bar_w:.0f}%;height:5px;border-radius:3px">'
                    f'</div></div>'
                    f'<div style="color:{T["dim"]};font-size:0.72em;line-height:1.4">'
                    f'{val_interp}</div>'
                    f'<div style="color:{T["dim"]};font-size:0.7em;margin-top:6px">'
                    f'in {demand}% der Inserate · SVS {svs}</div>'
                    + rf_note
                    + f'</div>', unsafe_allow_html=True)
    else:
        st.info("Keine fehlenden Skills in den verknüpften Dimensionen gefunden.")



# Disclaimer
st.markdown("---")
st.caption(
    "📋 **Datenquellen:** LinkedIn-Stellenanzeigen (Finance & Banking 49k / Consulting & Strategy 46k Jobs). "
    "Gehalts-Benchmark Finance & Banking: Robert Half Salary Guide Schweiz 2024 "
    "(25. Pz. = Entry / 50. Pz. = Mid / 75. Pz. = Senior). "
    "Consulting & Strategy: Gehaltsdaten aus LinkedIn-Inseraten extrahiert. "
    "Keine Gewähr auf Vollständigkeit oder Aktualität."
)

