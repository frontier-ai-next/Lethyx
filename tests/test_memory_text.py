"""Universal text behaviour: no alphabet, regex class, or stopword assumptions."""

from __future__ import annotations

from lethyx.memory import text


def test_tokens_split_every_unicode_script() -> None:
    value = "OpenCode → Lethyx/память: メモリ・バックエンド, ذاكرة, mémoire"
    assert text.tokens(value) == [
        "opencode",
        "lethyx",
        "память",
        "メモリ",
        "バックエンド",
        "ذاكرة",
        "mémoire",
    ]


def test_normalize_is_nfkc_casefolded_and_whitespace_collapsed() -> None:
    assert text.normalize("  Ｌｅｔｈｙｘ\tПАМЯТЬ  \n") == "lethyx память"


def test_rank_finds_a_spaceless_japanese_document() -> None:
    documents = [
        "Проект использует локальную память.",
        "メモリのバックエンドはSQLiteです。",
        "The deployment runs on Windows.",
    ]
    ranked = text.rank("バックエンドは何ですか", documents)
    assert ranked
    assert ranked[0][0] == 1
    assert ranked[0][1] == 1.0


def test_rank_tolerates_cyrillic_inflection_via_ngrams() -> None:
    documents = [
        "Мы делали установку памяти вчера.",
        "The cat sat on the mat.",
    ]
    ranked = text.rank("что мы делаем с памятью", documents)
    assert ranked
    assert ranked[0][0] == 0


def test_rank_weights_common_terms_by_corpus_frequency() -> None:
    documents = [
        "the backend stores records",
        "the user prefers short answers",
        "the deployment is local",
        "the project uses Windows",
    ]
    ranked = text.rank("the backend", documents)
    assert ranked
    # "the" occurs everywhere; "backend" is the discriminating term.
    assert ranked[0][0] == 0


def test_rank_prefers_a_title_match_over_incidental_body_terms() -> None:
    documents = [
        (
            "Lethyx local deployment and patches\n"
            "Lethyx is installed from a local clone. Local patches add compact "
            "responses, instructions, recall and a projection wait. Restart the "
            "server after editing the source. This is the deployment record."
        ),
        (
            "OpenCode config locations\n"
            "The global config lives at C:/Users/example/.config/opencode/opencode.json "
            "and the skills live beside it."
        ),
    ]
    ranked = text.rank(
        "where is the opencode config",
        documents,
        titles=[
            "Lethyx local deployment and patches",
            "OpenCode config locations",
        ],
    )
    assert ranked
    assert ranked[0][0] == 1


def test_rank_does_not_match_on_corpus_glue_alone() -> None:
    documents = [
        "Lethyx is the local memory system.",
        "The user is on Windows.",
        "The project is local.",
        "The backend is SQLite.",
        "The skills are installed.",
    ]
    assert text.rank("what is the capital of France", documents) == []


def test_rank_prefers_a_discriminating_title_over_incidental_long_body_terms() -> None:
    documents = [
        (
            "Lethyx local deployment and patches\n"
            "Lethyx is active and there are limitations. The deployment is local."
        ),
        (
            "Memory backend switched to mnemosyne\n"
            "Lethyx runs on the mnemosyne backend."
        ),
        (
            "OpenCode config locations\n"
            "The config is local and the skills are installed."
        ),
        (
            "Python and Node toolchain\n"
            "Python is installed and the tools are local."
        ),
        (
            "Installed OpenCode skills\n"
            "Skills are installed and the config is local."
        ),
    ]
    ranked = text.rank(
        "what memory backend is active and what are its limitations",
        documents,
        titles=[
            "Lethyx local deployment and patches",
            "Memory backend switched to mnemosyne",
            "OpenCode config locations",
            "Python and Node toolchain",
            "Installed OpenCode skills",
        ],
    )
    assert ranked
    assert ranked[0][0] == 1
    assert ranked[0][2] == ["memory", "backend"]


def test_rank_keeps_a_title_anchor_when_other_script_terms_do_not_match() -> None:
    documents = [
        "Lethyx is local.",
        "The config is here.",
        "Installed OpenCode skills",
        "Python is installed.",
        "OpenCode config locations",
    ]
    ranked = text.rank(
        "где находится конфигурация opencode",
        documents,
        titles=[
            "Lethyx local deployment",
            "Config",
            "Installed OpenCode skills",
            "Python toolchain",
            "OpenCode config locations",
        ],
    )
    assert ranked
    assert "opencode" in ranked[0][2]


def test_rank_returns_nothing_for_unrelated_text() -> None:
    documents = ["PostgreSQL tuning notes", "Rust async runtime"]
    assert text.rank("акварельные краски", documents) == []


def test_similarity_is_symmetric_and_identical_text_is_one() -> None:
    left = "OpenCode workspace setup"
    right = "workspace OpenCode setup notes"
    assert text.similarity(left, left) == 1.0
    assert text.similarity(left, right) == text.similarity(right, left)
    assert 0.0 < text.similarity(left, right) < 1.0
