# app.py
# CPEMPH Funding Finder (no DB, no keys) — Streamlit
# Pulls from open public feeds only and shows deadlines nearest first.

import re
import html
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from datetime import datetime, date, timezone  # <-- import date
import pytz
import streamlit as st

# Put page config first (Streamlit best practice)
st.set_page_config(page_title="CPEMPH Funding Finder (no DB)", layout="wide")

# =========================
# 1) Open, no-key sources
# =========================
GRANTS_GOV_HEALTH_RSS = "https://www.grants.gov/rss/GG_HealthCategory.xml"  # Health-category grants
NIH_ALL_ACTIVE_FOAS_XML = "https://grants.nih.gov/web_services/XML/NIH_Sponsored_FOAs.xml"

SA_TZ = pytz.timezone("Africa/Johannesburg")

# Simple HTTP session with a UA for friendlier servers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (FundingFinder/1.0; +https://streamlit.io)"
})

# =========================
# Helpers
# =========================
def _coerce_date(s):
    """Parse a date string to datetime.date, else return None."""
    if not s:
        return None
    try:
        dt = dateparser.parse(str(s), fuzzy=True)
        if dt is None:
            return None
        return dt.date()
    except Exception:
        return None

DATE_REGEX = re.compile(
    r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?,?\s*)?"
    r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
)

def _find_date_in_text(text):
    """Heuristic: find first date-like string in text and parse it."""
    if not text:
        return None
    m = DATE_REGEX.search(text)
    if m:
        return _coerce_date(m.group(0))
    return None

def _first_nonempty(*vals):
    for v in vals:
        if v and str(v).strip():
            return v
    return None

def _looks_sa_friendly(text):
    """Heuristic: mark as SA/international-friendly if text hints at foreign/international eligibility."""
    if not text:
        return False
    text_l = text.lower()
    keywords = [
        "international", "global", "low- and middle-income", "lmic",
        "foreign institutions", "foreign components", "africa", "south africa",
        "partnership", "collaborat", "global health"
    ]
    return any(k in text_l for k in keywords)

# =========================
# 2) Fetchers (cached)
# =========================
@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def fetch_grants_gov_health():
    """
    Parse Grants.gov Health RSS (open, no key).
    Returns list of dicts with title, link, agency, posted, deadline, summary, source.
    """
    rows = []
    try:
        feed = feedparser.parse(GRANTS_GOV_HEALTH_RSS)
    except Exception as e:
        st.warning(f"Could not parse Grants.gov RSS ({e}).")
        return rows

    for it in feed.entries:
        title = it.get("title")
        link = it.get("link")
        summary = html.unescape(_first_nonempty(it.get("summary"), it.get("description"), ""))

        closing_raw = _first_nonempty(
            it.get("closingdate"), it.get("closingDate"), it.get("closes"),
            it.get("closedate"), it.get("close_date")
        )
        posted_raw = _first_nonempty(
            it.get("published"), it.get("pubDate"), it.get("updated"), it.get("postdate"),
        )
        deadline = _coerce_date(closing_raw) or _find_date_in_text(summary)
        posted = _coerce_date(posted_raw)

        agency = _first_nonempty(it.get("agency"), it.get("author"), "")
        if not (title and link):
            continue

        rows.append({
            "source": "Grants.gov (Health RSS)",
            "title": title,
            "agency": agency,
            "link": link,
            "posted_date": posted,
            "deadline": deadline,
            "summary": summary,
            "sa_international_ok": _looks_sa_friendly(summary),
        })
    return rows

@st.cache_data(ttl=60 * 60)
def fetch_nih_active_foas():
    """
    Parse NIH Guide 'All Active FOAs' XML (open, no key).
    """
    rows = []
    try:
        resp = SESSION.get(NIH_ALL_ACTIVE_FOAS_XML, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"Could not reach NIH FOA XML ({e}). Showing other sources only.")
        return rows

    try:
        soup = BeautifulSoup(resp.content, "xml")  # requires lxml
    except Exception as e:
        st.warning(f"XML parsing failed ({e}).")
        return rows

    record_candidates = soup.find_all(["FOA", "Row", "row", "Record", "record", "notice", "Opportunity"])
    if not record_candidates:
        record_candidates = soup.find_all(recursive=False)

    for rec in record_candidates:
        title = _first_nonempty(
            getattr(rec.find("Title"), "text", None),
            getattr(rec.find("title"), "text", None),
            getattr(rec.find("Summary"), "text", None),
            getattr(rec.find("summary"), "text", None),
        )
        link = _first_nonempty(
            getattr(rec.find("Url"), "text", None),
            getattr(rec.find("URL"), "text", None),
            getattr(rec.find("Link"), "text", None),
            getattr(rec.find("link"), "text", None),
        )
        summary = _first_nonempty(
            getattr(rec.find("Summary"), "text", None),
            getattr(rec.find("Description"), "text", None),
            getattr(rec.find("summary"), "text", None),
            getattr(rec.find("description"), "text", None),
        )
        deadline = _coerce_date(_first_nonempty(
            getattr(rec.find("ExpirationDate"), "text", None),
            getattr(rec.find("ExpireDate"), "text", None),
            getattr(rec.find("CloseDate"), "text", None),
            getattr(rec.find("ClosingDate"), "text", None),
            getattr(rec.find("ApplicationDueDate"), "text", None),
        )) or _find_date_in_text(summary)

        posted = _coerce_date(_first_nonempty(
            getattr(rec.find("ReleaseDate"), "text", None),
            getattr(rec.find("PostedDate"), "text", None),
            getattr(rec.find("OpenDate"), "text", None),
        ))
        agency = _first_nonempty(
            getattr(rec.find("ICName"), "text", None),
            getattr(rec.find("Agency"), "text", None),
            getattr(rec.find("InstituteCenter"), "text", None),
            "NIH"
        )
        if not (title and link):
            continue

        rows.append({
            "source": "NIH Guide (Active FOAs)",
            "title": title.strip(),
            "agency": agency.strip() if isinstance(agency, str) else agency,
            "link": link.strip(),
            "posted_date": posted,
            "deadline": deadline,
            "summary": (summary or "").strip(),
            "sa_international_ok": _looks_sa_friendly(summary),
        })
    return rows

# =========================
# 3) Streamlit UI
# =========================
st.title("CPEMPH Funding Finder — no DB, no keys (pull-only)")

with st.sidebar:
    st.markdown("### Filters")
    include_grants_gov = st.checkbox("Include Grants.gov (Health)", value=True)
    include_nih = st.checkbox("Include NIH Guide (Active FOAs)", value=True)

    default_terms = "public health, epidemiology, global health, bioethics, medical humanities, philosophy of medicine"
    terms = st.text_input("Search terms (comma-separated)", value=default_terms)

    only_future = st.checkbox("Only show open calls (deadline in future)", value=True)
    sa_pref = st.checkbox(
        "Boost internationally/LMIC-friendly items",
        value=True,
        help="Heuristic: looks for words like 'international', 'LMIC', 'foreign institutions', 'South Africa', etc."
    )

    st.markdown("---")
    run_btn = st.button("🔄 Refresh results")

if run_btn or "autostarted" not in st.session_state:
    st.session_state["autostarted"] = True

    all_rows = []
    if include_grants_gov:
        all_rows.extend(fetch_grants_gov_health())
    if include_nih:
        all_rows.extend(fetch_nih_active_foas())

    df = pd.DataFrame(all_rows)

    if df.empty:
        st.info("No results fetched (feed may be temporarily unavailable, or filters are too strict). Try again or relax filters.")
    else:
        # --- text filter ---
        term_list = [t.strip().lower() for t in (terms or "").split(",") if t.strip()]
        if term_list:
            def row_matches_terms(row, term_list):
                blob = " ".join([str(row.get("title","")), str(row.get("summary","")), str(row.get("agency",""))]).lower()
                return all(t in blob for t in term_list)
            df = df[df.apply(lambda r: row_matches_terms(r, term_list), axis=1)]

        # --- date filter ---
        today = datetime.now(SA_TZ).date()
        if "deadline" not in df.columns:
            df["deadline"] = None

        if only_future:
            # Keep rows where deadline exists and is >= today
            df = df[df["deadline"].apply(lambda d: isinstance(d, date) and d >= today)]

        # If everything got filtered out, don't crash subsequent ops
        if df.empty:
            st.info("No items match your filters/terms. Try clearing the search or turning off 'Only open calls'.")
        else:
            # --- sort ---
            if sa_pref and "sa_international_ok" in df.columns:
                df = df.sort_values(by=["sa_international_ok", "deadline"], ascending=[False, True], kind="stable")
            elif "deadline" in df.columns:
                df = df.sort_values(by=["deadline"], ascending=[True], kind="stable")

            # --- display ---
            show = df[[
                "title", "agency", "deadline", "posted_date", "source", "link", "summary", "sa_international_ok"
            ]].rename(columns={
                "title": "Title",
                "agency": "Funder / Agency",
                "deadline": "Deadline",
                "posted_date": "Posted",
                "source": "Source",
                "link": "Link",
                "summary": "Summary",
                "sa_international_ok": "Intl/LMIC-friendly (heuristic)"
            })

            st.success(f"Found {len(show)} opportunities.")
            st.caption("Tip: click a column header to sort; shift-click to sort by multiple columns.")
            st.dataframe(show, use_container_width=True, hide_index=True)

            csv_bytes = show.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="cpemph_funding_results.csv",
                mime="text/csv"
            )

with st.expander("About the data / disclaimers"):
    st.markdown("""
- **Sources used (no keys required):**
  - Grants.gov **Health category RSS** (public).  
  - NIH Guide **All Active FOAs** XML (public).  
- We heuristic-parse deadlines and eligibility text; always click through to the official page before applying.
- “International/LMIC-friendly” is a *best-effort text check* only.
    """)
