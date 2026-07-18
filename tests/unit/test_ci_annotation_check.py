"""TEMPORARY: deliberately failing test to verify CI failure annotations.

This file exists only to confirm that
`pytest-github-actions-annotate-failures` surfaces failures as inline
annotations on the PR/commit in GitHub Actions.

>>> DELETE THIS FILE once the annotation behaviour has been confirmed. <<<
"""


def test_ci_annotation_smoke_check():
    expected = 2
    actual = 1 + 2
    assert actual == expected, "intentional failure to test CI annotations"
