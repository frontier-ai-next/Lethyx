"""Static fixtures for memory ablation."""


def static_scenarios() -> list[dict[str, object]]:
    return [
        {
            "id": "verified_lesson_before_action",
            "expected": ["rule_verified"],
            "items": [
                {
                    "id": "rule_verified",
                    "directive": True,
                    "surface": "directive",
                    "trust": "verified",
                    "source_backend": "bkb",
                }
            ],
        },
        {
            "id": "candidate_poisoning",
            "expected": [],
            "items": [
                {
                    "id": "poison",
                    "directive": False,
                    "surface": "reflection_candidate",
                    "source_backend": "hindsight",
                }
            ],
        },
        {
            "id": "retrieval_only_contamination",
            "expected": [],
            "items": [
                {
                    "id": "recall",
                    "directive": False,
                    "surface": "supplemental_evidence",
                    "source_backend": "palace.recall",
                }
            ],
        },
        {
            "id": "mnemosyne_passive",
            "expected": [],
            "items": [
                {
                    "id": "mn",
                    "directive": False,
                    "surface": "supplemental_evidence",
                    "source_backend": "mnemosyne",
                }
            ],
        },
    ]
