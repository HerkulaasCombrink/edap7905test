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
from datetime import datetime, date
import pytz
import streamlit as st

# ---------- Streamlit setup ----------
st.set_page_config(page_title="CPEMPH Funding Finder (no DB)", layout="wide")
st.title("CPEMPH Funding Finder — no DB, no keys (pull-only)")

# ---------- Sources (no API keys) ----------
GRANTS_GOV_HEALTH_RSS = "https://www.grants.gov/rss/GG_HealthCategory.xml"
NIH_ALL_ACTIVE_FOAS_XML = "https://grants.nih.gov/web_services/XML/NIH_Sponsored_FOAs.xml"
SA_TZ = pytz.timezone("Africa/Johannesburg")

# HTTP session with a basic UA for friendlier servers
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (FundingFinder/1.0; +https://streamlit.io)"})


# ---------- Helpers ----------
def _coerce_date(s):
    if not s:
        return None
    try:
        dt = dateparser.parse(str(s), fuzzy=True)
        return dt.date() if dt else None
    except Exception:
        return None

DATE_REGEX = re.compile(
    r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?,?\s*)?"
    r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
)

def _find_date_in_text(text):
    if not text:
        return None
    m = DATE_REGEX.search(text)
    return _coerce_date(m.group(0)) if m else None

def _first_nonempty(*vals):
    for v in vals:
        if v and str(v).strip():
            return v
    return None

def _looks_sa_friendly(text):
    if not text:
        return False
    txt = text.lower()
    keys = [
        "international", "global", "low- and middle-income", "lmic",
        "foreign institutions", "foreign components", "africa", "south africa",
        "partnership", "collaborat", "global health"
    ]
    return any(k in txt for k in keys)


# ---------- Fetchers (cached) ----------
@st.cache_data(ttl=60*60)
def fetch_grants_gov_health():
    rows = []
    try:
        feed = feedparser.parse(GRANTS_GOV_HEALTH_RSS)
    except Exception as e:
        st.warning(f"Grants.gov RSS parse error: {e}")
        return rows

    for it in feed.entries:
        title = it.get("title")
        link = it.get("link")
        if not (title and link):
            continue

        summary = html.unescape(_first_nonempty(it.get("summary"), it.get("description"), ""))

        closing_raw = _first_nonempty(
            it.get("closingdate"), it.get("closingDate"), it.get("closes"),
            it.get("closedate"), it.get("close_date")
        )
        posted_raw = _first_nonempty(it.get("published"), it.get("pubDate"), it.get("updated"), it.get("postdate"))
        deadline = _coerce_date(closing_raw) or _find_date_in_text(summary)
        posted = _coerce_date(posted_raw)
        agency = _first_nonempty(it.get("agency"), it.get("author"), "")

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


@st.cache_data(ttl=60*60)
def fetch_nih_active_foas():
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

    record_candidates = soup.find_all(["FOA", "Row", "row", "Record", "record", "notice", "Opportunity"]) or soup.find_all(recursive=False)

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
        if not (title and link):
            continue

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


# ---------- Sidebar / Controls ----------
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
        help="Simple heuristic (e.g., 'international', 'LMIC', 'South Africa')."
    )
    st.markdown("---")
    run_btn = st.button("🔄 Refresh results")


# ---------- Run ----------
if run_btn or "autostarted" not in st.session_state:
    st.session_state["autostarted"] = True

    all_rows = []
    if include_grants_gov:
        all_rows.extend(fetch_grants_gov_health())
    if include_nih:
        all_rows.extend(fetch_nih_active_foas())

    df = pd.DataFrame(all_rows)
    st.caption(f"Fetched {len(df)} raw items")

    if df.empty:
        st.info("No results fetched (feeds might be unavailable). Try again or relax filters.")
    else:
        # Ensure deadline column exists
        if "deadline" not in df.columns:
            df["deadline"] = None

        today = datetime.now(SA_TZ).date()

        # ---------- ONE-LINE FILTER ----------
        df = df[df.apply(
            lambda r: (
                (lambda ts, blob: (not ts) or any(t in blob for t in ts))
                ([t.strip().lower() for t in (terms or "").split(",") if t.strip()],
                 " ".join([str(r.get("title","")), str(r.get("summary","")), str(r.get("agency",""))]).lower())
            ) and (
                (not only_future) or
                (isinstance(r.get("deadline"), date) and r.get("deadline") >= today) or
                (r.get("deadline") is None)
            ),
            axis=1
        )]

        st.caption(f"After filters: {len(df)} items")

        if df.empty:
            st.info("No items match your filters. Try clearing keywords or toggling 'Only open calls'.")
        else:
            # Sort: SA/LMIC-friendly first (if chosen), then by deadline (None last)
            def none_last_sortkey(d):
                return (d is None, d)

            if sa_pref and "sa_international_ok" in df.columns:
                df = df.sort_values(
                    by=["sa_international_ok", "deadline"],
                    ascending=[False, True],
                    kind="stable"
                )
                # Apply None-last ordering for deadline
                df["__ord__"] = df["deadline"].map(none_last_sortkey)
                df = df.sort_values(by=["sa_international_ok", "__ord__"], ascending=[False, True], kind="stable").drop(columns=["__ord__"])
            else:
                df["__ord__"] = df["deadline"].map(none_last_sortkey)
                df = df.sort_values(by="__ord__", ascending=True, kind="stable").drop(columns=["__ord__"])

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
            st.caption("Tip: click a column header to sort; shift-click for multi-sort.")
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
- **Sources used (no keys):** Grants.gov **Health RSS** and NIH Guide **All Active FOAs** XML.
- Deadlines are parsed heuristically; always verify on the official page before applying.
- The “Intl/LMIC-friendly” flag is a best-effort text check only.
""")
