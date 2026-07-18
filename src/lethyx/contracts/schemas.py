"""JSON-schema fragments for typed Lethyx contract tools.

These schemas serve two purposes:
1. They guide the LLM to produce well-formed tool arguments by describing the
   expected shape of each field (properties, types, enums, descriptions).
2. The existing server-side ``from_mapping`` / normalizer layer validates and
   coerces the received payload regardless of how closely the LLM follows the
   schema.

When adding or tightening a schema: keep descriptions concrete (include an
example or the canonical slug list), keep ``required`` minimal (the normalizer
fills sensible defaults), and avoid over-constraining shapes that the
normalizer already coerces (e.g. don't mark ``available`` as required if a
pending probe legitimately omits it).
"""

CAPABILITY_PROBE_SCHEMA = {
    "type": "object",
    "required": ["kind", "command"],
    "description": (
        "Workspace command that Lethyx executes to determine whether a "
        "capability is available. The process exit code decides the result; "
        "no LLM inference is applied.\n"
        "For desktop_gui_runtime the probe MUST create and destroy (or show) "
        "a native window root, so set intent=real_gui_root_lifecycle and use "
        "a command such as:\n"
        '  ["python","-c","import tkinter as tk; r=tk.Tk(); r.update(); r.destroy()"]\n'
        "On headless hosts Lethyx auto-wraps the command in xvfb-run."
    ),
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["command"],
            "default": "command",
            "description": "Probe execution kind. Currently only 'command' is supported.",
        },
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "argv to run. Use a JSON array of strings, not a shell string.",
        },
        "intent": {
            "type": "string",
            "description": (
                "Required for desktop_gui_runtime probes: must be "
                "'real_gui_root_lifecycle'. Import-only checks belong to "
                "desktop_gui_headless or a library-specific capability."
            ),
        },
        "expect_exit": {
            "type": "integer",
            "default": 0,
            "description": "Expected process exit code (default 0 = success).",
        },
        "timeout_sec": {
            "type": "integer",
            "default": 5,
            "description": "Probe timeout in seconds.",
        },
        "shell": {
            "type": "boolean",
            "default": False,
        },
        "cwd": {
            "type": "string",
            "description": "Relative working directory inside the workspace.",
        },
        "execution_environment_id": {
            "type": "string",
            "description": "Optional execution environment slug (default: workspace default).",
        },
    },
}

CAPABILITY_ENTRY_SCHEMA = {
    "type": "object",
    "description": (
        "Typed capability entry. Either pass a plain boolean, or this object.\n"
        "``available`` is optional when a ``probe`` is attached — Lethyx runs "
        "the probe and infers availability. Do NOT set available=false when you "
        "are attaching a probe; set available=false only after a confirmed failed "
        "probe or a concrete platform policy/constraint.\n"
        "``available=false`` must mean a real limitation (policy, install, failed "
        "probe), not a planning preference ('not needed', 'will use headless instead')."
    ),
    "properties": {
        "available": {
            "type": "boolean",
            "description": (
                "Omit (leave absent) when a probe is attached so Lethyx can "
                "infer availability from the probe result."
            ),
        },
        "source": {
            "type": "string",
            "enum": ["probe", "declared", "inferred"],
            "description": "How availability was determined. Prefer 'probe' for runtime capabilities.",
        },
        "reason": {
            "type": "string",
            "description": "Short human-readable explanation for the availability decision.",
        },
        "notes": {
            "type": "string",
            "description": (
                "Additional context. Do not use notes to contradict the "
                "available field (e.g. 'probe confirmed' is fine; 'will probe "
                "later' with available=true is a contradiction)."
            ),
        },
        "probe": CAPABILITY_PROBE_SCHEMA,
    },
}

_READINESS_SPEC_SCHEMA = {
    "type": "object",
    "description": (
        "Machine-readable readiness check for the managed runtime launch. "
        "Use one of: process_alive, log_contains (add text:), "
        "http_ready (add url:), timeout_elapsed."
    ),
    "required": ["type"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["process_alive", "log_contains", "http_ready", "timeout_elapsed"],
        },
        "text": {
            "type": "string",
            "description": "Sentinel text for log_contains.",
        },
        "url": {
            "type": "string",
            "description": "URL for http_ready, e.g. http://localhost:8080/health",
        },
        "port": {
            "type": "integer",
            "description": "Port for http_ready (alternative to full url).",
        },
        "timeout_sec": {
            "type": "integer",
        },
    },
}

HARNESS_OPTIONS_SCHEMA = {
    "type": "object",
    "description": (
        "Harness-specific launch, readiness, interaction, evidence, and cleanup "
        "details for managed-runtime GUI / server proofs.\n\n"
        "Minimal shape for desktop_gui_runtime smoke proof:\n"
        '  {"managed_runtime":true,"readiness":{"type":"process_alive"},'
        '"startup_timeout_sec":10,'
        '"assert_command":["python","-m","pytest","tests/test_smoke.py","-q"]}\n\n'
        "assert_command/interaction_command/driver_command must be argv ARRAYS "
        "(list of strings) under proof.harness_options. Do not put a bare string, "
        "a single token like 'python', or these command slots as top-level proof "
        "fields. For desktop_gui_runtime, proof.execution.command remains the "
        "managed application launch command, not pytest."
    ),
    "properties": {
        "managed_runtime": {
            "type": "boolean",
            "description": "Set true for managed GUI / server launch proofs.",
        },
        "readiness": {
            "anyOf": [
                _READINESS_SPEC_SCHEMA,
                {"type": "array", "items": _READINESS_SPEC_SCHEMA},
            ],
            "description": "How Lethyx detects the launched process is ready.",
        },
        "startup_timeout_sec": {
            "type": "integer",
            "default": 10,
            "description": "Max seconds to wait for readiness.",
        },
        "poll_interval_sec": {
            "type": "integer",
            "default": 1,
        },
        "assert_command": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "argv run after readiness (e.g. a pytest smoke test or "
                "pyautogui driver script). Must be a JSON array of strings."
            ),
        },
        "interaction_command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alias for assert_command.",
        },
        "driver_command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alias for assert_command.",
        },
        "assert_subdir": {
            "type": "string",
            "description": "Workspace subdirectory to run assert/interaction/driver command.",
        },
        "assert_timeout_sec": {
            "type": "integer",
        },
        "evidence": {
            "type": "string",
            "description": "What machine-readable evidence will be captured.",
        },
        "cleanup": {
            "type": "string",
            "description": "Cleanup note (process is always killed on timeout).",
        },
        "extra_env": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
}

EVIDENCE_REF_SCHEMA = {
    "type": "object",
    "required": ["ref_type", "ref_id", "produced_by"],
    "properties": {
        "ref_type": {
            "type": "string",
            "enum": [
                "ledger_event",
                "verification_report",
                "test_run",
                "artifact",
                "diff",
                "memory_node",
                "harness_candidate",
                "mutation_report",
                "input_sensitivity_report",
            ],
        },
        "ref_id": {"type": "string"},
        "hash": {"type": "string"},
        "produced_by": {
            "type": "string",
            "enum": ["agent", "supervisor", "verifier", "watcher", "harness"],
        },
        "phase": {"type": "string"},
        "subtask_id": {"type": "string"},
        "created_after_event": {"type": "string"},
    },
}

LEDGER_EVIDENCE_REF_SCHEMA = {
    **EVIDENCE_REF_SCHEMA,
    "properties": {
        **EVIDENCE_REF_SCHEMA["properties"],
        "ref_type": {
            "type": "string",
            "enum": [
                "ledger_event",
                "verification_report",
                "test_run",
                "mutation_report",
                "input_sensitivity_report",
            ],
        },
    },
}

VALID_REVIEW_CODES = (
    "missing_proof",
    "weak_proof",
    "manual_proof",
    "unavailable_proof_target",
    "test_tampering_risk",
    "scope_mismatch",
    "policy_violation",
    "insufficient_research_evidence",
    "requires_human_checkpoint",
    "stale_proof_ref",
    "fake_evidence_ref",
    "invalid_evidence_ref",
    "invalid_python_c_proof",
    "long_lived_server_command",
    "non_ledger_evidence_ref",
    "shell_operator_in_argv",
    "proof_after_patch_missing",
    "proof_scope_mismatch",
    "claim_without_proof",
    "test_tampering_detected",
    "verifier_mutation_attempt",
    "memory_without_verified_evidence",
    "legacy_contract_used",
    "llm_judge_only_evidence",
    "greenfield_python_src_layout_policy",
    "unknown_harness_profile",
    "missing_capability_declaration",
    "capability_probe_failed",
    "bad_generated_oracle",
    "plan_contract_issue",
    "inconsistent_generated_oracle",
    "oracle_domain_mismatch",
    "contradictory_required_behavior",
    "invalid_generated_test_contract",
    "unmapped_generated_oracle",
    "proof_execution_infra",
    "package_import_env_mismatch",
    "proof_execution_env_mismatch",
    "setup_harness_mismatch",
    "capability_probe_environment_mismatch",
    "dependency_provision_required",
    "headless_proof_uses_real_gui_root",
)

PLAN_REVISION_DELTA_SCHEMA = {
    "type": "object",
    "required": ["op", "path"],
    "description": (
        "A typed ContractDelta describing a concrete plan/proof revision.\n"
        "op vocabulary is add/remove/replace (replace_applied/remove_applied/"
        "delete are also accepted and coerced).\n"
        "Shape rules by path type:\n"
        "  object path (proof.generated_test_contract, proof.oracle, "
        "proof.execution): use op=replace with an OBJECT value, or omit value "
        "for a structural-revision marker.\n"
        "  list path (proof.execution.command, proof.scope.pytest_targets, "
        "proof.generated_test_contract.oracle_claims): use op=remove/add/replace "
        "with values=[...] (a list).\n"
        "  scalar path (proof.harness_profile, "
        "proof.anti_gaming.requires_real_runtime): use op=replace with a scalar "
        "value.\n"
        "Do not put a scalar/string value on an object path (e.g. "
        "value='accepted=true' on proof.generated_test_contract) — point at the "
        "oracle_claims list with values, or use a structural marker."
    ),
    "properties": {
        "op": {
            "type": "string",
            "enum": ["remove", "replace", "add"],
            "description": (
                "add/remove/replace. replace_applied/remove_applied/delete are "
                "coerced to replace/remove."
            ),
        },
        "path": {"type": "string"},
        "values": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Required for list-path add/remove/replace.",
        },
        "value": {"description": "Scalar for scalar paths, object for object paths."},
        "replacement": {},
        "target_subtask_id": {"type": "string"},
        "source_issue_code": {"type": "string"},
    },
}

GENERATED_TEST_CONTRACT_SCHEMA = {
    "type": "object",
    "description": (
        "Typed oracle model for generated tests. Tests are evidence for these "
        "claims, not the source of truth for contradictory behavior."
    ),
    "properties": {
        "interface_model": {"type": "object"},
        "proof_budget": {
            "type": "object",
            "properties": {
                "max_generated_tests_per_subtask": {"type": "integer"},
                "allow_expanded_generated_tests": {"type": "boolean"},
                "override_reason": {"type": "string"},
            },
        },
        "oracle_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_id", "source"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "source": {
                        "type": "string",
                        "enum": [
                            "task_requirement",
                            "interface_model",
                            "reference_behavior",
                            "harness_contract",
                        ],
                    },
                    "subject": {"type": "string"},
                    "event": {"type": "string"},
                    "api": {"type": "string"},
                    "input": {},
                    "input_value": {},
                    "input_values": {"type": "array"},
                    "input_sequence": {"type": "array"},
                    "accepted": {"type": "boolean"},
                    "valid": {"type": "boolean"},
                    "expectation": {"type": "string"},
                    "expected_behavior": {"type": "string"},
                    "expected_output": {},
                    "expected_display": {},
                    "expected_status": {},
                    "expected_result": {},
                    "test_refs": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

REVIEW_COVERAGE_SCHEMA = {
    "type": "object",
    "description": (
        "Review coverage checklist. Each key means the dimension was evaluated. "
        "For verdict ok, every field must be true; for revise/abort, false may "
        "mark a dimension whose blocker is captured in typed issues."
    ),
    "required": [
        "policy_conflicts",
        "oracle_compatibility",
        "proof_strength",
        "scope_validity",
        "runtime_capabilities",
        "test_validity",
    ],
    "properties": {
        "policy_conflicts": {"type": "boolean"},
        "oracle_compatibility": {"type": "boolean"},
        "proof_strength": {"type": "boolean"},
        "scope_validity": {"type": "boolean"},
        "runtime_capabilities": {"type": "boolean"},
        "test_validity": {"type": "boolean"},
    },
}

FULL_REVIEW_COVERAGE = {
    "policy_conflicts": True,
    "oracle_compatibility": True,
    "proof_strength": True,
    "scope_validity": True,
    "runtime_capabilities": True,
    "test_validity": True,
}

PROOF_CONTRACT_SCHEMA = {
    "type": "object",
    "anyOf": [
        {"required": ["template_id"]},
        {"required": ["execution", "oracle", "scope"]},
    ],
    "properties": {
        "template_id": {
            "type": "string",
            "enum": [
                "python_src_unit",
                "desktop_gui_headless_model",
                "desktop_gui_runtime_smoke",
            ],
            "description": (
                "Optional platform-owned canonical proof template. When set, "
                "the compiler fills execution/oracle/scope before validation."
            ),
        },
        "template_args": {
            "type": "object",
            "description": "Narrow template inputs such as package for runtime smoke.",
        },
        "execution": {
            "type": "object",
            "required": ["kind", "command"],
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "pytest",
                        "verification_step",
                        "http_boot",
                        "behavioral_http",
                        "input_sensitivity",
                        "mutation_smoke",
                        "metamorphic",
                        "property_test",
                        "import_check",
                        "build",
                        "command",
                    ],
                },
                "command": {"type": "array", "items": {"type": "string"}},
                "timeout_sec": {"type": "integer"},
                "shell": {"type": "boolean"},
                "subdir": {"type": "string"},
                "execution_environment_id": {"type": "string"},
                "environment_id": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
        "execution_environment_id": {"type": "string"},
        "oracle": {
            "type": "object",
            "required": ["oracle_type", "required_properties"],
            "properties": {
                "oracle_type": {
                    "type": "string",
                    "enum": [
                        "unit_assertions",
                        "behavioral_gui",
                        "behavioral_http",
                        "input_sensitivity",
                        "metamorphic",
                        "snapshot",
                        "mutation_kill",
                        "golden_file",
                        "build",
                        "import",
                    ],
                },
                "required_properties": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "distinct_inputs_distinct_outputs",
                            "invalid_input_rejected",
                            "round_trip",
                            "idempotence",
                            "monotonicity",
                            "no_test_tampering",
                            "mutation_killed",
                            "runtime_started",
                            "module_imports",
                            "build_succeeds",
                        ],
                    },
                },
                "negative_cases_required": {"type": "boolean"},
                "input_sensitivity_required": {"type": "boolean"},
            },
        },
        "scope": {
            "type": "object",
            "properties": {
                "files_under_test": {"type": "array", "items": {"type": "string"}},
                "changed_files_expected": {"type": "array", "items": {"type": "string"}},
                "pytest_targets": {"type": "array", "items": {"type": "string"}},
                "test_targets": {"type": "array", "items": {"type": "string"}},
            },
        },
        "anti_gaming": {
            "type": "object",
            "properties": {
                "allows_mock": {"type": "boolean"},
                "allows_snapshot_update": {"type": "boolean"},
                "allows_test_only_change": {"type": "boolean"},
                "requires_real_runtime": {"type": "boolean"},
            },
        },
        "harness_profile": {"type": "string"},
        "harness_options": HARNESS_OPTIONS_SCHEMA,
        "generated_test_contract": GENERATED_TEST_CONTRACT_SCHEMA,
        "required_capabilities": {"type": "array", "items": {"type": "string"}},
        "human_claims": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": EVIDENCE_REF_SCHEMA},
    },
}

SUBTASK_MEMORY_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "baseline": {"type": "array", "items": {"type": "string"}},
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "ref"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "codeptr",
                            "knowledge_md",
                            "github_repo",
                            "github_snippet",
                            "web_page",
                            "web_section",
                            "web_search_hit",
                            "palace_finding",
                            "gmas_context",
                            "workspace_file",
                            "mcp_server",
                            "skill",
                            "terminal_tail",
                        ],
                    },
                    "ref": {"type": "string"},
                    "title": {"type": "string"},
                    "inject_mode": {
                        "type": "string",
                        "enum": ["preload", "on_demand", "search_only"],
                    },
                    "max_chars": {"type": "integer"},
                    "source_id": {"type": "string"},
                },
            },
        },
        "palace_search_queries": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
}

PHASE_PLAN_SUBTASK_SCHEMA = {
    "type": "object",
    "required": ["id", "title", "goal", "proof"],
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "goal": {"type": "string"},
        "files_to_create": {"type": "array", "items": {"type": "string"}},
        "files_to_change": {"type": "array", "items": {"type": "string"}},
        "files_affected": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "acceptance_claims": {"type": "array", "items": {"type": "string"}},
        "proof": PROOF_CONTRACT_SCHEMA,
        "generated_test_contract": GENERATED_TEST_CONTRACT_SCHEMA,
        "memory_scope": SUBTASK_MEMORY_SCOPE_SCHEMA,
        "allowed_tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Extra phase tool names needed by this leaf.",
        },
        "allowed_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Extra skills/prompts the execute phase should load.",
        },
        "codeptr_refs": {"type": "array", "items": {"type": "string"}},
        "mcp_refs": {"type": "array", "items": {"type": "string"}},
    },
}

PHASE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["subtasks"],
    "properties": {
        "plan_id": {"type": "string"},
        "run_id": {"type": "string"},
        "workspace_id": {"type": "string"},
        "subtasks": {
            "type": "array",
            "minItems": 1,
            "items": PHASE_PLAN_SUBTASK_SCHEMA,
        },
    },
    "description": (
        "Executable Lethyx phase plan. Use exactly one top-level `subtasks` "
        "array with typed proof objects and put "
        "memory_scope, allowed_tools, allowed_skills, codeptr_refs, and "
        "mcp_refs on each leaf that needs extra context or tools."
    ),
}

REVIEW_ISSUE_SCHEMA = {
    "type": "object",
    "required": ["code", "severity"],
    "properties": {
        "code": {
            "type": "string",
            "enum": list(VALID_REVIEW_CODES),
        },
        "severity": {
            "type": "string",
            "enum": ["info", "warning", "error", "blocking", "human_required"],
        },
        "phase": {"type": "string"},
        "subtask_id": {"type": "string"},
        "target_subtask_id": {"type": "string"},
        "target_path": {"type": "string"},
        "file_path": {"type": "string"},
        "contract_path": {"type": "string"},
        "invalid_values": {"type": "array", "items": {"type": "string"}},
        "required_deltas": {"type": "array", "items": PLAN_REVISION_DELTA_SCHEMA},
        "failure_hash": {"type": "string"},
        "failure_phase": {"type": "string"},
        "production_code_entered": {"type": "boolean"},
        "capability_id": {"type": "string"},
        "env_id": {"type": "string"},
        "env_hash": {"type": "string"},
        "message": {"type": "string"},
        "evidence_refs": {"type": "array", "items": EVIDENCE_REF_SCHEMA},
    },
}

VERIFICATION_REPORT_REF_SCHEMA = {
    "type": "object",
    "required": [
        "report_id",
        "report_hash",
        "workspace_hash",
        "diff_hash",
        "produced_after_event_id",
        "verifier_id",
        "passed",
    ],
    "properties": {
        "report_id": {"type": "string"},
        "report_hash": {"type": "string"},
        "workspace_hash": {"type": "string"},
        "diff_hash": {"type": "string"},
        "produced_after_event_id": {"type": "string"},
        "verifier_id": {"type": "string"},
        "passed": {"type": "boolean"},
        "ledger_hash": {"type": "string"},
        "execution_environment_id": {"type": "string"},
        "env_hash": {"type": "string"},
        "proof_env_hash": {"type": "string"},
        "python_executable": {"type": "string"},
    },
}

COMPLETION_CONTRACT_SCHEMA = {
    "type": "object",
    "required": ["subtask_id", "status", "completed_claims", "changed_files"],
    "properties": {
        "subtask_id": {"type": "string"},
        "status": {"type": "string", "enum": ["done"]},
        "completed_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_id", "text", "files", "proof_refs"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "text": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "proof_refs": {
                        "type": "array",
                        "items": LEDGER_EVIDENCE_REF_SCHEMA,
                    },
                },
            },
        },
        "changed_files": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": LEDGER_EVIDENCE_REF_SCHEMA},
        "verification_report": VERIFICATION_REPORT_REF_SCHEMA,
        "notes": {"type": "string"},
    },
}
