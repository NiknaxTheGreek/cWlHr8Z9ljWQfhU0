from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import pandas as pd

from .config import HR_SCORE_ADJACENT, HR_SCORE_DIRECT, HR_SCORE_IRRELEVANT
from .validation import require, require_columns


@dataclass(frozen=True)
class HRRule:
    rule_id: str
    hr_class: str
    pattern: str
    reason: str
    score: float


DIRECT_RULES: tuple[HRRule, ...] = (
    HRRule("HR_DIRECT_HUMAN_RESOURCES", "direct", r"\bhuman\s+resources?\b", "Explicit Human Resources wording", HR_SCORE_DIRECT),
    HRRule("HR_DIRECT_HR_ACRONYM", "direct", r"\bhr\b", "Explicit HR acronym", HR_SCORE_DIRECT),
    HRRule("HR_DIRECT_CHRO", "direct", r"\bchro\b", "Chief Human Resources Officer acronym", HR_SCORE_DIRECT),
    HRRule("HR_DIRECT_GPHR_SPHR", "direct", r"\b(?:gphr|sphr)\b", "Professional HR credential", HR_SCORE_DIRECT),
)
ADJACENT_RULES: tuple[HRRule, ...] = (
    HRRule("HR_ADJ_PEOPLE_DEVELOPMENT", "adjacent", r"\bpeople\s+development\b", "People-development role adjacent to HR", HR_SCORE_ADJACENT),
    HRRule("HR_ADJ_PEOPLE_OPERATIONS", "adjacent", r"\bpeople\s+operations\b", "People-operations role adjacent to HR", HR_SCORE_ADJACENT),
    HRRule("HR_ADJ_TALENT", "adjacent", r"\btalent\s+(?:management|acquisition|development)\b", "Talent function adjacent to HR", HR_SCORE_ADJACENT),
)


def ruleset_payload() -> list[dict]:
    return [r.__dict__ for r in (*DIRECT_RULES, *ADJACENT_RULES)] + [{"rule_id":"HR_IRRELEVANT_FALLBACK","hr_class":"irrelevant","pattern":"<fallback>","reason":"No direct or adjacent HR rule matched","score":HR_SCORE_IRRELEVANT}]


def ruleset_hash() -> str:
    payload = json.dumps(ruleset_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def classify_title(title: str) -> dict:
    text = str(title).casefold()
    for rule in DIRECT_RULES:
        if re.search(rule.pattern, text, flags=re.IGNORECASE):
            return {"hr_class":rule.hr_class,"hr_rule_id":rule.rule_id,"hr_reason":rule.reason,"H":rule.score}
    for rule in ADJACENT_RULES:
        if re.search(rule.pattern, text, flags=re.IGNORECASE):
            return {"hr_class":rule.hr_class,"hr_rule_id":rule.rule_id,"hr_reason":rule.reason,"H":rule.score}
    return {"hr_class":"irrelevant","hr_rule_id":"HR_IRRELEVANT_FALLBACK","hr_reason":"No direct or adjacent HR rule matched","H":HR_SCORE_IRRELEVANT}


def apply_hr_rules(clean_50: pd.DataFrame, expected_hr34: int = 34) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(clean_50, ("representative_id", "job_title"), "HR_RULE_INPUT")
    classified = clean_50.copy()
    details = pd.DataFrame(classified["job_title"].map(classify_title).tolist())
    classified = pd.concat([classified.reset_index(drop=True), details], axis=1)
    require(classified["H"].isin([0.0, 0.5, 1.0]).all(), "HR_SCORE_SET", "Unexpected H score")
    hr34 = classified[classified["H"] > 0].copy().reset_index(drop=True)
    require(len(hr34) == expected_hr34, "HR34_COUNT", f"Expected {expected_hr34} HR-relevant candidates, found {len(hr34)}")
    return classified, hr34
