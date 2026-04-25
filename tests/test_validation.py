from osint_env.validation import run_validation_suite


def test_validation_suite_passes_repo_gate():
    result = run_validation_suite()
    assert result["passed"] is True
    assert len(result["checks"]) >= 4
