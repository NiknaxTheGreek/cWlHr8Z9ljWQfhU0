from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .validation import require, require_columns, require_unique

RAW_REQUIRED_COLUMNS = ("id", "job_title", "location", "connection", "fit")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
EDA_TOKEN_RE = re.compile(r"[A-Za-z]+")
EDA_STOPWORDS = frozenset({"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"})


@dataclass(frozen=True)
class InvalidContentRule:
    rule_id: str
    reason: str


def load_raw_candidates(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, RAW_REQUIRED_COLUMNS, "RAW_SCHEMA_COLUMNS")
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    require_unique(df, "id", "RAW_ID_UNIQUE")
    return df


def load_raw(path: Path | str) -> pd.DataFrame:
    """Reviewer-facing alias for the immutable raw-source loader."""
    return load_raw_candidates(path)


def exact_title_universe(raw: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw, ("id", "job_title"), "EXACT_TITLE_INPUT")
    work = raw.copy()
    work["job_title"] = work["job_title"].astype(str)
    rows = []
    for title, grp in work.groupby("job_title", sort=False, dropna=False):
        ids = sorted(int(x) for x in grp["id"].tolist())
        rows.append({
            "representative_id": ids[0],
            "job_title": title,
            "source_count": len(ids),
            "source_ids": json.dumps(ids),
        })
    out = pd.DataFrame(rows).sort_values("representative_id", kind="stable").reset_index(drop=True)
    require_unique(out, "representative_id", "EXACT_REP_ID_UNIQUE")
    return out


def invalid_content_rule(title: str) -> InvalidContentRule | None:
    raw = str(title).strip()
    low = raw.casefold()
    if "staffing" in low and "seeking" in low and PHONE_RE.search(raw):
        return InvalidContentRule(
            "INVALID_EMPLOYER_SOLICITATION",
            "Employer/staffing solicitation with phone contact; not a candidate profile",
        )
    normalized = re.sub(r"[^a-z0-9]+", " ", low).strip()
    if normalized == "always set them up for success":
        return InvalidContentRule(
            "INVALID_NON_ROLE_SLOGAN",
            "Non-occupational slogan with no candidate role information",
        )
    return None


def remove_invalid_records(
    title_universe: pd.DataFrame,
    expected_removals: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(title_universe, ("representative_id", "job_title"), "INVALID_RULE_INPUT")
    keep_rows, audit_rows = [], []
    for row in title_universe.to_dict("records"):
        rule = invalid_content_rule(row["job_title"])
        if rule is None:
            keep_rows.append(row)
        else:
            audit_rows.append({
                "representative_id": int(row["representative_id"]),
                "job_title": row["job_title"],
                "exclusion_rule_id": rule.rule_id,
                "reason": rule.reason,
            })
    require(
        len(audit_rows) == expected_removals,
        "INVALID_REMOVAL_COUNT",
        f"Expected {expected_removals} invalid records, found {len(audit_rows)}",
    )
    return pd.DataFrame(keep_rows).reset_index(drop=True), pd.DataFrame(audit_rows)


def remove_invalid(
    title_universe: pd.DataFrame,
    expected_removals: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reviewer-facing alias preserving the explicit exclusion audit."""
    return remove_invalid_records(title_universe, expected_removals=expected_removals)


def normalize_title_key(title: str) -> str:
    text = unicodedata.normalize("NFKC", str(title)).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_normalized_titles(valid_titles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(valid_titles, ("representative_id", "job_title"), "DEDUP_INPUT")
    work = valid_titles.copy()
    work["normalized_title"] = work["job_title"].map(normalize_title_key)
    require(work["normalized_title"].ne(""), "NORMALIZED_TITLE_NONEMPTY", "Normalized title cannot be empty")
    work = work.sort_values("representative_id", kind="stable")
    chosen = work.drop_duplicates("normalized_title", keep="first").copy()
    audit = work[["representative_id", "job_title", "normalized_title"]].copy()
    audit["kept"] = audit["representative_id"].isin(chosen["representative_id"])
    chosen = chosen.sort_values("representative_id", kind="stable").reset_index(drop=True)
    require_unique(chosen, "normalized_title", "NORMALIZED_TITLE_UNIQUE")
    return chosen, audit.reset_index(drop=True)


def normalized_dedup(
    valid_titles: pd.DataFrame,
    return_audit: bool = False,
):
    """Notebook-facing normalized-title deduplication helper."""
    clean, audit = deduplicate_normalized_titles(valid_titles)
    return (clean, audit) if return_audit else clean


def clean_candidates(
    raw: pd.DataFrame,
    expected_raw_rows: int = 104,
    expected_exact_titles: int = 52,
    expected_removals: int = 2,
    expected_final: int = 50,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    require(len(raw) == expected_raw_rows, "RAW_ROW_COUNT", f"Expected {expected_raw_rows} raw rows, got {len(raw)}")
    title_universe = exact_title_universe(raw)
    require(
        len(title_universe) == expected_exact_titles,
        "EXACT_TITLE_COUNT",
        f"Expected {expected_exact_titles} exact title representatives, got {len(title_universe)}",
    )
    valid_titles, exclusion_audit = remove_invalid_records(title_universe, expected_removals)
    clean, dedup_audit = deduplicate_normalized_titles(valid_titles)
    require(len(clean) == expected_final, "CLEAN_FINAL_COUNT", f"Expected {expected_final} clean representatives, got {len(clean)}")
    audits = {"title_universe": title_universe, "exclusions": exclusion_audit, "dedup": dedup_audit}
    return clean, audits


def _eda_tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.casefold() for match in EDA_TOKEN_RE.findall(str(text)))
        if token not in EDA_STOPWORDS
    ]


def text_frequency_summary(titles) -> dict[str, pd.DataFrame]:
    """Focused title-language EDA used only to motivate the two locked queries.

    This tokenizer is intentionally simple and separate from Word2Vec preprocessing.
    It lowercases alphabetic tokens, removes a small fixed stopword set, and reports
    transparent corpus counts without feeding directly into the model target.
    """
    title_list = [str(x) for x in titles]
    token_lists = [_eda_tokens(title) for title in title_list]
    tokens = [token for row in token_lists for token in row]
    bigrams = [
        f"{row[i]} {row[i + 1]}"
        for row in token_lists
        for i in range(len(row) - 1)
    ]

    unigram_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)

    unigrams = pd.DataFrame(
        sorted(unigram_counts.items(), key=lambda item: (-item[1], item[0])),
        columns=["token", "count"],
    )
    bigram_frame = pd.DataFrame(
        sorted(bigram_counts.items(), key=lambda item: (-item[1], item[0])),
        columns=["bigram", "count"],
    )
    summary = pd.DataFrame(
        {
            "measure": [
                "titles",
                "retained_tokens",
                "unique_unigrams",
                "unique_bigrams",
            ],
            "value": [
                len(title_list),
                len(tokens),
                len(unigram_counts),
                len(bigram_counts),
            ],
        }
    )
    return {"summary": summary, "unigrams": unigrams, "bigrams": bigram_frame}
