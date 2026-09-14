import pandas as pd
from src.data_pipeline import deduplicate_normalized_titles, exact_title_universe, invalid_content_rule, normalize_title_key, remove_invalid_records


def test_content_rules_are_content_based():
    assert invalid_content_rule("Nortia Staffing is seeking Human Resources, Payroll & Administrative Professionals!! (408) 709-2621").rule_id=="INVALID_EMPLOYER_SOLICITATION"
    assert invalid_content_rule("Always set them up for Success").rule_id=="INVALID_NON_ROLE_SLOGAN"
    assert invalid_content_rule("Aspiring Human Resources Professional") is None


def test_invalid_removal_precedes_normalized_dedup():
    raw=pd.DataFrame({"id":[1,2,3,4],"job_title":["HR Manager","HR Manager","Always set them up for Success","People Development Coordinator"]})
    universe=exact_title_universe(raw); valid,exclusions=remove_invalid_records(universe,expected_removals=1); clean,_=deduplicate_normalized_titles(valid)
    assert len(exclusions)==1 and set(clean["representative_id"])=={1,4}


def test_normalization_is_deterministic():
    assert normalize_title_key(" Human  Resources | Manager ")=="human resources manager"
