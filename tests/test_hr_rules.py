from src.hr_rules import classify_title, ruleset_hash


def test_hr_rule_precedence_and_scores():
    assert classify_title("Human Resources Generalist")["H"]==1.0
    assert classify_title("HR Manager")["H"]==1.0
    assert classify_title("People Development Coordinator")["H"]==0.5
    assert classify_title("Biology Student")["H"]==0.0


def test_ruleset_hash_stable():
    assert ruleset_hash()==ruleset_hash()
