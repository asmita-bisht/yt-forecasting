# streamlit_app.py

# imports
import os, re, json, math, glob, warnings
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap
from models.soft_voting_ensemble import SoftVotingEnsemble  

#  CSS 
st.markdown("""
<style>
.result-hero { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; margin:10px 0 6px; }
.result-head { font-size:26px; font-weight:700; letter-spacing:.2px; }
.result-sub { color:#9ca3af; font-size:.95rem; }
.band-badge { padding:.26rem .7rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); font-weight:700; font-size:.95rem; letter-spacing:.2px; }
</style>
""", unsafe_allow_html=True)

#  discovery 
def latest(path_glob: str):
    hits = glob.glob(path_glob)
    return max(hits, key=os.path.getmtime) if hits else None

def find_bundle_and_meta():
    bundle = latest("models/phase3_bundle_*.joblib") or latest("models/phase3_deployed_*.joblib") or latest("*.joblib")
    meta   = latest("reports/bundle_meta_*.json")   or latest("reports/metrics_*.json")        or latest("*.json")
    return bundle, meta

@st.cache_resource(show_spinner=False)
def load_bundle(p: str):
    obj = joblib.load(p)
    if not hasattr(obj, "predict_proba"):
        raise TypeError("Loaded object has no predict_proba; expected a scikit-learn Pipeline.")
    return obj

@st.cache_resource(show_spinner=False)
def load_meta(p: str):
    with open(p, "r") as f: return json.load(f)

#  feature helpers 
YT_CATS = {
    1:"Film & Animation", 2:"Autos & Vehicles", 10:"Music", 15:"Pets & Animals",
    17:"Sports", 19:"Travel & Events", 20:"Gaming", 22:"People & Blogs",
    23:"Comedy", 24:"Entertainment", 25:"News & Politics", 26:"Howto & Style",
    27:"Education", 28:"Science & Tech", 29:"Nonprofits", 30:"Movies", 43:"Shows",
}

FRIENDLY = {
    "channel_subscriberCount_log1p": "Channel subscribers (log-scaled)",
    "channel_viewCount_log1p":       "Channel total views (log-scaled)",
    "channel_videoCount_log1p":      "Channel total uploads (log-scaled)",
    "duration_log1p":                "Video length (log-scaled)",
    "tags_count":                    "Number of tags",
    "has_tags":                      "Has tags",
    "title_has_question":            "Title contains question mark",
    "title_has_exclaim":             "Title contains exclamation mark",
    "desc_len_words":                "Description length (words)",
    "title_caps_ratio":              "Title ALL CAPS ratio",
    "pub_hour":                      "Publish hour",
    "pub_dow":                       "Publish weekday",
    "pub_hour_sin":                  "Publish hour (cyclical – sin)",
    "pub_hour_cos":                  "Publish hour (cyclical – cos)",
}

FEATURE_GROUP = {
    "channel_subscriberCount_log1p":"Channel strength",
    "channel_viewCount_log1p":"Channel strength",
    "channel_videoCount_log1p":"Channel strength",
    "duration_log1p":"Metadata quality",
    "tags_count":"Metadata quality",
    "has_tags":"Metadata quality",
    "title_has_question":"Metadata quality",
    "title_has_exclaim":"Metadata quality",
    "desc_len_words":"Metadata quality",
    "title_caps_ratio":"Metadata quality",
    "pub_hour":"Timing", "pub_dow":"Timing", "pub_hour_sin":"Timing", "pub_hour_cos":"Timing",
}

def band_from_score(score_0_100: float):
    if score_0_100 >= 85:  return ("Very high", "#16a34a")
    if score_0_100 >= 70:  return ("High",      "#22c55e")
    if score_0_100 >= 40:  return ("Moderate",  "#f59e0b")
    return ("Low", "#9ca3af")

def safe_len_words(x): 
    if x is None or (isinstance(x,float) and math.isnan(x)): return 0
    return len(str(x).split())

def safe_len_chars(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return 0
    return len(str(x))

def caps_ratio(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return 0.0
    s = re.sub(r'[^A-Za-z]', '', str(x)); 
    if not s: return 0.0
    return sum(c.isupper() for c in s)/len(s)

def count_char(x, ch): 
    if x is None or (isinstance(x,float) and math.isnan(x)): return 0
    return str(x).count(ch)

def text_is_missing(x) -> bool:
    if x is None: return True
    s = str(x).strip().lower()
    return s in {"", "nan", "none"}

def tags_count(x):
    if x is None: return 0
    if isinstance(x, (list,tuple)): return len(x)
    s = str(x)
    if s.startswith('[') and s.endswith(']'):
        try:
            import ast; lst = ast.literal_eval(s)
            return len(lst) if isinstance(lst, list) else 0
        except Exception: pass
    if '|' in s: return len([t for t in s.split('|') if t.strip()])
    if ',' in s: return len([t for t in s.split(',') if t.strip()])
    return 1 if s.strip() else 0

def tags_to_text(x) -> str:
    if x is None or (isinstance(x,float) and math.isnan(x)): return "no_tags"
    if isinstance(x,(list,tuple)):
        toks = [str(t).strip().lower().replace(' ','_') for t in x if str(t).strip()]
    else:
        s = str(x)
        if s.startswith('[') and s.endswith(']'):
            try:
                import ast; lst = ast.literal_eval(s)
                toks = [str(t).strip().lower().replace(' ','_') for t in lst if str(t).strip()]
            except Exception:
                toks = [t.strip().lower().replace(' ','_') for t in re.split(r'[|,]', s) if t.strip()]
        else:
            toks = [t.strip().lower().replace(' ','_') for t in re.split(r'[|,]', s) if t.strip()]
    return " ".join(toks) if toks else "no_tags"

def log1p_num(x):
    try: return float(np.log1p(float(x)))
    except Exception: return float(np.log1p(0.0))

def pretty_value(base, val):
    if base == "channel_subscriberCount_log1p": return f"{int(np.expm1(val)):,} subs"
    if base == "channel_viewCount_log1p":       return f"{int(np.expm1(val)):,} views"
    if base == "channel_videoCount_log1p":      return f"{int(np.expm1(val)):,} uploads"
    if base == "duration_log1p":                return f"{int(np.expm1(val)/60):,} mins"
    if base == "tags_count":                    return f"{int(val)} tags"
    if base in {"pub_hour","pub_hour_sin","pub_hour_cos"}: return f"{int(val)}h"
    return str(round(float(val),2))

def build_feature_row(
    title: str, description: str, tags_raw: str, category_name: str,
    duration_sec: float, publish_hour: int, publish_weekday: int,
    ch_subs: Optional[float], ch_views: Optional[float], ch_uploads: Optional[float]
) -> pd.DataFrame:

    title = "" if title is None else str(title)
    description = "" if description is None else str(description)
    tags_text = tags_to_text(tags_raw)
    t_missing = int(text_is_missing(title))
    d_missing = int(text_is_missing(description))

    title_words = safe_len_words(title); title_chars = safe_len_chars(title)
    title_caps  = caps_ratio(title)
    t_exc = count_char(title,'!'); t_q = count_char(title,'?')
    desc_words = safe_len_words(description)
    title_has_question = int('?' in title); title_has_exclaim = int('!' in title)
    title_emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', title))
    desc_emoji_count  = len(re.findall(r'[\U00010000-\U0010ffff]', description))

    duration_log1p = log1p_num(duration_sec)
    tags_cnt       = tags_count(tags_raw)
    has_tags       = int(tags_cnt > 0)

    h = int(publish_hour) % 24
    r = 2*np.pi*(h/24.0)
    pub_hour_sin = float(np.sin(r)); pub_hour_cos = float(np.cos(r))

    subs_log1p  = log1p_num(0 if ch_subs    is None else ch_subs)
    views_log1p = log1p_num(0 if ch_views   is None else ch_views)
    vids_log1p  = log1p_num(0 if ch_uploads is None else ch_uploads)


    cat_id = next((cid for cid,nm in YT_CATS.items() if nm == category_name), None)
    if cat_id is None:
        try: cat_id = int(category_name)
        except Exception: cat_id = 24

    row = {
        "title": title, "description": description, "tags_text": tags_text,
        "duration_log1p": duration_log1p, "tags_count": tags_cnt,
        "title_len_words": title_words, "title_len_chars": title_chars,
        "title_caps_ratio": title_caps, "title_num_exclaim": t_exc, "title_num_question": t_q,
        "desc_len_words": desc_words,
        "pub_hour": float(h), "pub_dow": float(publish_weekday),
        "pub_hour_sin": pub_hour_sin, "pub_hour_cos": pub_hour_cos,
        "channel_subscriberCount_log1p": subs_log1p,
        "channel_viewCount_log1p":       views_log1p,
        "channel_videoCount_log1p":      vids_log1p,
        "title_missing": t_missing, "description_missing": d_missing, "has_tags": has_tags,
        "title_has_question": title_has_question, "title_has_exclaim": title_has_exclaim,
        "title_emoji_count": title_emoji_count, "desc_emoji_count": desc_emoji_count,
        "categoryId": cat_id,
    }
    return pd.DataFrame([row])

def to_score(p): return float(round(100.0 * float(p), 2))

#  tips (suggestions given to user based on SHAP results)
def tip_for(base: str, value, shap_val: float) -> str:
    pos = shap_val > 0
    if base == "has_tags":
        return "Add 6–12 specific tags around 2–3 core topics. If you have over this amount, be careful of overuse" if not pos else "This means the choice of having tags has increased your score. Generally having 6–12 specific tags around 2–3 core topics is the sweet spot. "
    if base == "tags_count":
        return "Your tag count is decreasing your score/ try to stick to ~8–15 specific tags and avoid broad, noisy tags." if value == 0 or (isinstance(value,(int,float)) and value < 6) else "This means the number of tags you have is improving your score. "
    if base == "duration_log1p":
        return "Make sure to match length to content depth; avoid ultra-short for long-form."
    if base == "title_has_question":
        return "If you do not have a question mark, consider trying a question-style hook if natural for the content. If you do have a question mark present, consider removing it if possible given context." if not pos else "This means the choice of either having a question mark or not having one increased your score."
    if base == "title_has_exclaim":
        return "If these are present in your title, avoid multiple exclamation marks; 0–1 max." if not pos else "This means the choice of either using or not using a question mark improved your score."
    if base == "desc_len_words":
        return "Your description length decreased your score. Having a short, helpful description with 1–2 key phrases is recommended" if not pos else "This means the length of your description increased your score."
    if base == "title_caps_ratio":
        return "Reduce ALL-CAPS; keep to 1–2 words max." if not pos else "Title casing is fine."
    if base in {"pub_hour","pub_hour_sin","pub_hour_cos"}:
        if pos:
            return (
                "Your publish hour lines up with global YouTube engagement trends the model has seen. "
                "Still, confirm in your own YouTube Analytics (‘When your viewers are on YouTube’) to make sure it matches your audience."
            )
        else:
            return (
                "This publish hour doesn’t align strongly with global YouTube engagement patterns. "
                "Check your YouTube Analytics (‘When your viewers are on YouTube’) and consider shifting to peak times for your audience."
            )
    if base == "pub_dow":
        if pos:
            return (
                "This publish day matches broader platform-level patterns of higher engagement. "
                "Verify in your YouTube Analytics if your viewers also respond well on this day."
            )
        else:
            return (
                "This publish day appears weaker based on global YouTube engagement trends the model learned. "
                "Test Thu–Sun or whichever days your Analytics shows as strongest, and compare results."
            )
    if base == "channel_videoCount_log1p":
        if pos:
            return "A high upload count is indicative of a possibly consistent upload schedule. Keep it up."
        else:
            return "Few total uploads could imply that uploads are not consistent. If this is the case, try to establish a steady cadence, such as uploading 1–2 times a week!"

    if base == "channel_viewCount_log1p":
        if pos:
            return "Your lifetime view count is likely to be high and improving your score. Recreate strong hooks/thumbnails from your past hits."
        else:
            return "Your lifetime view count is likely to be on the lower side and decreasing your chances. Sometimes, a video gone viral can bring viewers to other videos on your channel—keep iterating to capture that effect!"

    if base == "channel_subscriberCount_log1p":
        if pos:
            return "Your subscriber count is boosting your score. Your channel strength helps—continue to engage with your audience and publish consistently."
        else:
            return "Your subscriber count is contributing to decreasing your score. Try focusing on growing your audience."
    return "Tweak this setting and re-score to see the impact."

#  SHAP 
def render_shap(bundle, X):
    # get steps
    if hasattr(bundle, "named_steps") and "pre" in bundle.named_steps:
        pre = bundle.named_steps["pre"]
        clf = bundle.named_steps.get("model", bundle.named_steps.get("clf", None))
    else:
        pre, clf = None, None
    if pre is None or clf is None:
        raise RuntimeError("Pipeline must have steps 'pre' and 'model'/'clf'.")

    # build numeric background by perturbing engineered features
    base = X.iloc[0].copy(); rows = []

    for dur in [30, 120, 600, 1800, 3600]:
        r = base.copy(); r["duration_log1p"] = float(np.log1p(dur)); rows.append(r)
    for h in [0, 6, 12, 18, 21]:
        r = base.copy()
        hh = int(h)%24; ang = 2*np.pi*(hh/24.0)
        r["pub_hour"] = float(hh); r["pub_hour_sin"] = float(np.sin(ang)); r["pub_hour_cos"] = float(np.cos(ang))
        rows.append(r)
    for subs in [0, 1_000, 100_000, 1_000_000]:
        r = base.copy(); r["channel_subscriberCount_log1p"] = float(np.log1p(subs)); rows.append(r)
    for views in [0, 100_000, 5_000_000, 50_000_000]:
        r = base.copy(); r["channel_viewCount_log1p"] = float(np.log1p(views)); rows.append(r)
    for vids in [0, 50, 500, 5_000]:
        r = base.copy(); r["channel_videoCount_log1p"] = float(np.log1p(vids)); rows.append(r)
    for t in [0,3,8]:
        r = base.copy(); r["tags_count"]=float(t); r["has_tags"]=int(t>0); rows.append(r)
    for q in [0,1]:
        r = base.copy(); r["title_has_question"]=int(q); rows.append(r)
    for e in [0,1]:
        r = base.copy(); r["title_has_exclaim"]=int(e); rows.append(r)

    bg_raw = pd.DataFrame(rows)
    bg_num = pre.transform(bg_raw)
    x_num  = pre.transform(X)

    def _dense(a):
        try: return a.toarray()
        except Exception: return np.asarray(a)

    bg_dense = _dense(bg_num)
    x_dense  = _dense(x_num)

    # feature names from preprocessor
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        feat_names = [f"f{i}" for i in range(x_dense.shape[1])]

    # single explainer (fix: preserve names via wrapper + independent masker)
    def f_num(A):
        A = np.asarray(A)
        A_df = pd.DataFrame(A, columns=feat_names)
        return clf.predict_proba(A_df)[:, 1]

    masker = shap.maskers.Independent(bg_dense)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        explainer = shap.Explainer(f_num, masker)
        sh = explainer(x_dense)

    # ensure names
    sh.feature_names = feat_names

    #  assemble contributions (use names we control) 
    contribs_all = pd.DataFrame({
        "feature": feat_names,
        "shap_value": sh.values[0],
        "feature_value": np.asarray(x_dense[0]).ravel(),
    })

    # numeric/engineered only
    raw_num_cols = {
        "duration_log1p","tags_count","title_len_words","title_len_chars","title_caps_ratio",
        "title_num_exclaim","title_num_question","desc_len_words","pub_hour","pub_dow",
        "pub_hour_sin","pub_hour_cos","channel_subscriberCount_log1p",
        "channel_viewCount_log1p","channel_videoCount_log1p","title_missing",
        "description_missing","has_tags","title_has_question","title_has_exclaim",
        "title_emoji_count","desc_emoji_count",
    }

    def base_name(n): return str(n).split("__",1)[-1]
    def is_numeric_feature(n)->bool:
        s = str(n); b = base_name(s)
        return s.startswith("num__") or (s in raw_num_cols) or (b in raw_num_cols)

    contribs_num = contribs_all[contribs_all["feature"].map(is_numeric_feature)].copy()
    if contribs_num.empty: contribs_num = contribs_all.copy()

    contribs_num["base"] = contribs_num["feature"].map(base_name)
    contribs_num["friendly_feature"] = contribs_num["base"].map(lambda b: FRIENDLY.get(b, b))
    if "display_value" not in contribs_num.columns:
        contribs_num["display_value"] = [
            pretty_value(b, v) for b, v in zip(contribs_num["base"], contribs_num["feature_value"])
        ]

    contribs_num = contribs_num.sort_values("shap_value", key=np.abs, ascending=False)
    topN = contribs_num.head(15).copy()

    #  title + popover help 
    HELP = ("**What is SHAP?**  \n"
            "SHAP (Shapley Additive Explanations) is a method from machine learning that attributes a model’s prediction to each input. The numbers you see are the size of each input’s contribution to your viral score. Positive numbers mean boosts to your score, and negative numbers mean penalties to your score.")

    c1, c2 = st.columns([1, 0.18])
    with c1:
        st.markdown("#### How Your Video Features Affected the Prediction (using SHAP)")
    with c2:
        with st.popover("What is SHAP?"): st.markdown(HELP)

    st.caption("This bar chart shows how each input affected *this* prediction relative to an average reference point. "
               "**Green bars** pushed probability up; **red bars** pushed it down. Longer bars = bigger impact. "
               "Labels like “log-scaled” or “cyclical” reflect engineered features used by the model.")

    #  SHAP bar chart 
    top_plot = topN.copy()
    bar_colors = ["#10b981" if v > 0 else "#ef4444" for v in top_plot["shap_value"]]
    fig = plt.figure(figsize=(7.4, 4.6), dpi=150)
    plt.axvline(0, linewidth=1, alpha=0.6)
    plt.barh(top_plot["friendly_feature"][::-1], top_plot["shap_value"][::-1], color=bar_colors[::-1])
    plt.xlabel("SHAP value (impact on probability)"); plt.ylabel("Feature")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    #  driver cards (top 3 + / top 3 −) 
    top_pos = topN[topN["shap_value"] > 0].head(3).copy()
    top_neg = topN[topN["shap_value"] < 0].head(3).copy()
    impact_pts = lambda x: round(float(x) * 100.0, 2)

    def _driver_card(title, impact_text, suggestion, color):
        bg = "#052e1b" if color == "green" else "#3a0a0a"
        border = "#16a34a" if color == "green" else "#ef4444"
        emoji = "📈" if color == "green" else "📉"
        st.markdown(
            f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:12px; padding:12px 14px; margin:6px 0;">
            <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                <div style="font-weight:700">{emoji} {title}</div>
                <div style="font-size:.9rem;opacity:.9">{impact_text}</div>
            </div>
            <div style="font-size:.9rem;margin-top:6px">{suggestion}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Your Top Numeric Drivers")
    lcol, rcol = st.columns(2)
    with lcol:
        st.caption("Improved Your Score (top 3)")
        for _, row in top_pos.iterrows():
            _driver_card(
                row["friendly_feature"],
                f"+{impact_pts(row['shap_value'])} pts",
                tip_for(row["base"], row["display_value"], row["shap_value"]),
                "green",
            )
    with rcol:
        st.caption("Hurt Your Score (top 3)")
        for _, row in top_neg.iterrows():
            _driver_card(
                row["friendly_feature"],
                f"-{abs(impact_pts(row['shap_value']))} pts",
                tip_for(row["base"], row["display_value"], row["shap_value"]),
                "red",
            )

#  UI 
st.title("YouTube Engagement Predictor")
st.caption(
    "This tool predicts how engaging your YouTube video could be before you upload. Just enter your title, description, tags, category, and planned publish time. You’ll get back a score for your video, a breakdown of what boosted or decreased the score, and quick suggestions to improve your video's chances of performing well."
)

# initialize config once
if "bundle_path" not in st.session_state or "meta_path" not in st.session_state:
    auto_bundle, auto_meta = find_bundle_and_meta()
    st.session_state.bundle_path = auto_bundle or ""
    st.session_state.meta_path   = auto_meta or ""

bundle_path = st.session_state.bundle_path
meta_path   = st.session_state.meta_path

if not bundle_path or not os.path.exists(bundle_path):
    st.error("Bundle not found. Put your Phase-3 pipeline under models/ (e.g., phase3_bundle_*.joblib).")
    st.stop()

if not meta_path or not os.path.exists(meta_path):
    st.warning("Meta JSON not found. Using defaults (threshold=0.5, top_frac=0.10).")

bundle = load_bundle(bundle_path)
meta   = load_meta(meta_path) if os.path.exists(meta_path) else {}
thr       = float(meta.get("threshold_val_top_frac", 0.5))
top_frac  = float(meta.get("top_frac", 0.10))

st.markdown("#### Enter video details")
with st.form("single_video_form", clear_on_submit=False):
    WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    hour_labels = [f"{(h%12) or 12}:00 {'AM' if h<12 else 'PM'}" for h in range(24)]

    title = st.text_area("Title", height=80, placeholder='e.g., "We tried the FASTEST street food in Tokyo 🍜"')
    description = st.text_area("Description", height=110, placeholder="A short description…")
    tags_raw = st.text_area("Tags (comma or | separated)", height=60, placeholder="food, tokyo, ramen | travel vlog")

    c1, c2, c3, c4 = st.columns([3,1,1,1])
    with c1:
        cat_vals_sorted = sorted(YT_CATS.values())
        default_idx = cat_vals_sorted.index("Entertainment") if "Entertainment" in cat_vals_sorted else 0
        cat_name = st.selectbox("Category", cat_vals_sorted, index=default_idx, key="cat")
    with c2:
        duration_min = st.number_input("Duration (mins)", min_value=0.5, max_value=24*60.0, value=10.0, step=0.5, key="dur_min")
        duration_sec = int(round(float(duration_min) * 60))
    with c3:
        sel_label = st.selectbox("Publish time", hour_labels, index=18, key="pub_time")
        pub_hour = hour_labels.index(sel_label)
    with c4:
        pub_dow_name = st.selectbox("Publish day", WEEKDAYS, index=4, key="pub_day")
        pub_dow = WEEKDAYS.index(pub_dow_name)

    with st.expander("Channel size (optional)", expanded=False):
        ch_subs    = st.number_input("Channel subscribers", min_value=0, value=0, step=1000)
        ch_views   = st.number_input("Channel total views", min_value=0, value=0, step=10000)
        ch_uploads = st.number_input("Channel total uploads", min_value=0, value=0, step=10)

    go = st.form_submit_button("Score my video")

#  required-field validation (no blank forms) 
def _is_blank(x):
    return x is None or (isinstance(x, str) and x.strip() == "")

if go:
    problems = []
    if _is_blank(title):       problems.append("Title")
    if _is_blank(description): problems.append("Description")
    if _is_blank(tags_raw):    problems.append("Tags")

    missing_chan = []
    for label, val in [("subscribers", ch_subs), ("total views", ch_views), ("total uploads", ch_uploads)]:
        try:
            if val is None or float(val) <= 0:
                missing_chan.append(label)
        except Exception:
            missing_chan.append(label)
    if missing_chan:
        problems.append("Channel metrics: " + ", ".join(missing_chan))

    if problems:
        msg = "Please fill in: " + " • ".join(problems)
        try:
            st.toast(msg, icon="⚠️")
        except Exception:
            st.error(msg)
        st.stop()

#  score + explain 
if go:
    X = build_feature_row(
        title=title, description=description, tags_raw=tags_raw,
        category_name=cat_name, duration_sec=duration_sec,
        publish_hour=pub_hour, publish_weekday=pub_dow,
        ch_subs=ch_subs, ch_views=ch_views, ch_uploads=ch_uploads
    )

    try:
        prob = float(bundle.predict_proba(X)[:, 1][0])
    except Exception as e:
        st.exception(e); st.stop()

    score = to_score(prob)
    band_label, band_color = band_from_score(score)
    top_pct = int(round(top_frac * 100))

    st.divider()
    st.markdown(
        f"""
        <div class="result-hero">
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="band-badge" style="background:{band_color}22; color:{band_color}; border-color:{band_color}55;">{band_label}</span>
            <div class="result-head">Your video scored {score:.1f}/100</div>
          </div>
        </div>
        <div class="result-sub">
          This means your video has an estimated <b>{prob:.0%}</b> chance of landing in the top <b>{top_pct}%</b> of similar videos
          (our “viral-potential” range), based on the metadata you entered and patterns learned by the model.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")  # spacer
    try:
        render_shap(bundle, X)
    except Exception as e:
        st.warning(f"SHAP explanation skipped ({type(e).__name__}): {e}")

#  footer / controls 
st.markdown("---")
with st.expander("FAQ"):
    st.markdown("**How does this work?**")
    st.write(
        "This tool uses a machine learning model built as a *soft voting ensemble*. "
        "That means it blends several different algorithms and averages their predicted probabilities "
        "to make a single estimate. The model was trained on YouTube metadata such as titles, "
        "descriptions, tags, categories, and publish timing. By learning from thousands of past videos, "
        "it finds patterns that separate top performers from the rest. When you enter your details, "
        "the model compares them to those learned patterns and estimates the chance your video will land in "
        "the top group of similar uploads."
    )

    st.markdown("**Is this updated live?**")
    st.write(
        "No. This is a static model trained on a fixed dataset of videos collected during August and September 2025. "
        "Predictions reflect trends from that period and will not change until the model is retrained. "
        "In the future, the goal is to move to a continuously updated model that refreshes with new video data "
        "so results stay aligned with current trends."
    )

    st.markdown("**Is this a guarantee?**")
    st.write(
        "No. This score is an estimate based on metadata and historical patterns. Actual performance also depends on "
        "factors not included here, like audience retention, watch time, click-through rates on thumbnails, social sharing, "
        "and external events. As more videos are collected and the model is retrained, estimates should improve, "
        "but they will always be an estimate and not a guarantee."
    )

st.markdown("---")
with st.expander("Artifact configuration (auto-discovered; change if needed)", expanded=False):
    st.text_input("Bundle .joblib path", key="bundle_path")
    st.text_input("Meta .json path",   key="meta_path")
    colA, colB = st.columns([1,3])
    with colA:
        if st.button("Apply & reload"): st.experimental_rerun()
    with colB:
        st.caption("Tip: Place files under models/ and reports/ to be picked up automatically.")

st.markdown("---")
st.caption("Model bundle is a scikit-learn Pipeline with steps ('pre', <preprocessor>), ('clf' or 'model', <estimator>). Meta JSON provides a validation-chosen threshold and top-fraction.")
