"""Tool schemas exposed by the Hermes provider."""

from typing import Any

from um_mem.memory.topology_schema import BLOCK_EVOLUTION_TARGET_SCHEMA, BLOCK_MERGE_SOURCE_SCHEMA


def provider_tool_schemas(
    *,
    default_recall_limit: int,
    max_recall_limit: int,
    default_recall_max_tokens: int,
    min_recall_tokens: int,
    max_recall_tokens: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "um_mem_provider_status",
            "description": "Report the active UM-MEM provider's negotiated runtime profile and content-free service health.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "um_mem_provider_block_map",
            "description": (
                "Show the available root memory sections without exposing record contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
                    "include_meta": {"type": "boolean", "default": False},
                    "child_limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_expand_block",
            "description": (
                "Expand one content-free branch before recall or remember. Inspect each "
                "leaf's writable reason, allowed operations, retention, native isolation, "
                "record count, health, and latency. Follow next_cursor for more children; "
                "restart pagination if the catalog revision changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                    "cursor": {"type": "string", "default": ""},
                },
                "required": ["block_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_navigation_envelope",
            "description": (
                "Read one content-free tree/graph navigation envelope with health facts, "
                "advisory observations, and explicit proposal options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                    "cursor": {"type": "string", "default": ""},
                },
                "required": ["block_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_recall_blocks",
            "description": (
                "Recall only from explicitly selected leaf blocks. The selected backend "
                "queries only those native partitions and returns only actually recalled "
                "advisory records within the active-frame budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "default": default_recall_limit,
                        "minimum": 1,
                        "maximum": max_recall_limit,
                    },
                    "source_horizon": {
                        "type": "string",
                        "enum": ["working", "durable", "archive"],
                        "default": "durable",
                    },
                    "logical_block_ids": {"type": "array", "items": {"type": "string"}},
                    "record_ids": {"type": "array", "items": {"type": "string"}},
                    "source_speakers": {"type": "array", "items": {"type": "string"}},
                    "temporal_mode": {"type": "string", "enum": ["current", "historical"]},
                    "as_of": {
                        "type": "string",
                        "description": "Optional event-time cutoff in ISO format. Omit unless the question asks for a temporal cutoff; a source date is not automatically a cutoff.",
                    },
                    "known_as_of": {
                        "type": "string",
                        "description": "Optional knowledge-time cutoff: an ISO timestamp WITH timezone for when UM-MEM committed the record. This is not the source/event date. Omit for ordinary historical questions; old source dates can predate the entire store.",
                    },
                    "revision_id": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "max_tokens": {
                        "type": "integer",
                        "default": default_recall_max_tokens,
                        "minimum": min_recall_tokens,
                        "maximum": max_recall_tokens,
                    },
                },
                "required": ["query", "logical_block_ids"],
            },
        },
        {
            "name": "um_mem_provider_search_all",
            "description": (
                "Explicit audited broad search after navigation cannot identify a suitable leaf "
                "or scoped recall and query refinement leave evidence insufficient. You choose "
                "whether to broaden the search; every materialized native partition "
                "is searched and the event is counted separately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "default": default_recall_limit,
                        "minimum": 1,
                        "maximum": max_recall_limit,
                    },
                    "source_horizon": {
                        "type": "string",
                        "enum": ["working", "durable", "archive"],
                        "default": "durable",
                    },
                    "source_speakers": {"type": "array", "items": {"type": "string"}},
                    "temporal_mode": {"type": "string", "enum": ["current", "historical"]},
                    "as_of": {
                        "type": "string",
                        "description": "Optional event-time cutoff in ISO format. Omit unless the question asks for a temporal cutoff; a source date is not automatically a cutoff.",
                    },
                    "known_as_of": {
                        "type": "string",
                        "description": "Optional knowledge-time cutoff: an ISO timestamp WITH timezone for when UM-MEM committed the record. This is not the source/event date. Omit for ordinary historical questions; old source dates can predate the entire store.",
                    },
                    "revision_id": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "max_tokens": {
                        "type": "integer",
                        "default": default_recall_max_tokens,
                        "minimum": min_recall_tokens,
                        "maximum": max_recall_tokens,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "um_mem_provider_list_records",
            "description": "List bounded canonical record IDs in one selected scope with a frozen cursor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": [
                            "user_global",
                            "workspace",
                            "project",
                            "session",
                            "task",
                            "subtask",
                        ],
                    },
                    "scope_id": {"type": "string", "minLength": 1},
                    "logical_block_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 128},
                    "cursor": {"type": "string"},
                },
                "required": ["scope", "scope_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_read_exact_record",
            "description": "Read one canonical record by its ID or unambiguous exact evidence reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "minLength": 1},
                    "evidence_ref": {"type": "string", "minLength": 1},
                },
                "oneOf": [{"required": ["record_id"]}, {"required": ["evidence_ref"]}],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_read_exact_records",
            "description": "Read a bounded batch of canonical record IDs in requested order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 128,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["record_ids"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_operation_status",
            "description": "Read an original operation receipt and current native projection status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string", "minLength": 1, "maxLength": 256}
                },
                "required": ["operation_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_projection_barrier",
            "description": "Wait for one committed record revision to reach its native projection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "minLength": 1},
                    "commit_revision": {"type": "integer", "minimum": 0},
                    "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 120},
                },
                "required": ["record_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_remember",
            "description": (
                "Add a candidate or propose a correction by memory ID inside one "
                "agent-writable leaf selected from the expanded block map. Never write to "
                "directive or index leaves; keep one independently changeable fact per "
                "atomic child claim and preserve its source witness. When rejected, "
                "follow the returned suggested_route and use repair with the same "
                "witness instead of creating a second source claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "operation_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "write_mode": {"type": "string", "enum": ["standard", "lossless_atomic"]},
                    "source_segments": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Exact source units. In lossless_atomic, content must equal their text concatenation, without added separators. You choose the semantic leaf; the host checks integrity only.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source_id": {"type": "string", "minLength": 1},
                                "text": {"type": "string", "minLength": 1},
                                "sha256": {"type": "string"},
                                "source_sha256": {"type": "string"},
                                "byte_start": {"type": "integer", "minimum": 0},
                                "byte_end": {"type": "integer", "minimum": 1},
                                "speaker": {"type": "string"},
                                "occurred_at": {"type": "string"},
                            },
                            "required": ["source_id", "text"],
                        },
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "correct", "repair"],
                    },
                    "memory_id": {"type": "string"},
                    "requested_block_id": {"type": "string"},
                    "reason": {"type": "string", "default": "correction"},
                    "title": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "repair_of": {"type": "string", "minLength": 1},
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "ref_type": {"type": "string"},
                                "ref_id": {"type": "string", "minLength": 1},
                                "hash": {"type": ["string", "null"]},
                                "produced_by": {"type": "string"},
                                "producer_id": {"type": "string", "minLength": 1},
                                "stage_id": {"type": ["string", "null"]},
                                "subtask_id": {"type": ["string", "null"]},
                                "created_after_event": {"type": ["string", "null"]},
                            },
                            "required": ["ref_type", "ref_id", "produced_by", "producer_id"],
                            "additionalProperties": False,
                        },
                    },
                    "child_claims": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["content", "operation", "requested_block_id"],
                "oneOf": [
                    {
                        "properties": {"operation": {"const": "add"}},
                        "required": ["operation"],
                        "not": {
                            "anyOf": [
                                {"required": ["memory_id"]},
                                {"required": ["repair_of"]},
                            ]
                        },
                    },
                    {
                        "properties": {
                            "operation": {"const": "correct"},
                            "memory_id": {"type": "string", "minLength": 1},
                        },
                        "required": ["operation", "memory_id"],
                    },
                    {
                        "properties": {
                            "operation": {"const": "repair"},
                            "repair_of": {"type": "string", "minLength": 1},
                            "evidence_refs": {"type": "array", "minItems": 1},
                        },
                        "required": ["operation", "repair_of", "evidence_refs"],
                        "not": {"required": ["memory_id"]},
                    },
                ],
            },
        },
        {
            "name": "um_mem_provider_amend",
            "description": (
                "Append one immutable CAS-guarded advisory-memory revision. Supply the "
                "current revision and selected writable leaf; trusted or directive memory "
                "continues to require the correction proposal and tribunal path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "minLength": 1},
                    "title": {"type": "string", "minLength": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata_patch": {"type": "object"},
                    "retention": {
                        "type": "string",
                        "enum": ["ephemeral", "session", "durable", "archive"],
                    },
                    "expires_at": {"type": "string"},
                    "requested_block_id": {"type": "string", "minLength": 1},
                },
                "required": [
                    "memory_id",
                    "expected_revision",
                    "reason",
                    "requested_block_id",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_create_block",
            "description": ("Create one free-topology block. UM-MEM generates its opaque UUID."),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "synopsis": {"type": "string", "minLength": 1},
                    "parent_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "index_only": {"type": "boolean", "default": False},
                    "token_budget": {"type": "integer", "minimum": 0, "default": 1000},
                    "expires_at": {
                        "type": "string",
                        "description": "Optional ISO-8601 block TTL expiry.",
                    },
                },
                "required": ["name", "synopsis", "parent_id", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_amend_block",
            "description": "CAS-amend the presentation of one free-topology block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "synopsis": {"type": "string", "minLength": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata_patch": {"type": "object"},
                    "token_budget": {
                        "type": "integer",
                        "minimum": 0,
                        "default": default_recall_max_tokens,
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "Optional ISO-8601 block TTL expiry.",
                    },
                },
                "required": ["block_id", "expected_revision", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_archive_block",
            "description": (
                "CAS-archive one workspace-owned memory block; archived blocks can be "
                "revived through amend_block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["block_id", "expected_revision", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_move_block",
            "description": "CAS-move one block beneath another active index block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "parent_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["block_id", "parent_id", "expected_revision", "reason"],
                "additionalProperties": False,
            },
        },
        *[
            {
                "name": f"um_mem_provider_{operation}_block",
                "description": f"CAS-{operation} one unprotected free-topology block.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "block_id": {"type": "string", "minLength": 1},
                        "expected_revision": {"type": "integer", "minimum": 1},
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["block_id", "expected_revision", "reason"],
                    "additionalProperties": False,
                },
            }
            for operation in ("revive", "delete")
        ],
        {
            "name": "um_mem_provider_split_block",
            "description": "Split one leaf through prepared native projections and one atomic catalog publication; retain operation_id for recovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "targets": {
                        "type": "array",
                        "minItems": 1,
                        "items": BLOCK_EVOLUTION_TARGET_SCHEMA,
                    },
                    "reason": {"type": "string", "minLength": 1},
                    "operation_id": {"type": "string", "minLength": 1},
                },
                "required": ["block_id", "expected_revision", "targets", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_merge_blocks",
            "description": "Merge leaves through prepared native projections and one atomic catalog publication; retain operation_id for recovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "minItems": 2,
                        "items": BLOCK_MERGE_SOURCE_SCHEMA,
                    },
                    "target": BLOCK_EVOLUTION_TARGET_SCHEMA,
                    "parent_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "operation_id": {"type": "string", "minLength": 1},
                },
                "required": ["sources", "target", "parent_id", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_pending_topology_evolutions",
            "description": "List split/merge generations needing projection or publication recovery.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_resume_topology_evolution",
            "description": "Resume one previously authored split/merge mapping by operation_id.",
            "parameters": {
                "type": "object",
                "properties": {"operation_id": {"type": "string", "minLength": 1}},
                "required": ["operation_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_abort_topology_evolution",
            "description": "Explicitly abandon an unpublished split/merge mapping; retain its journal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["operation_id", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_add_block_edge",
            "description": "Add one of the eight typed graph edges, separate from the tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_id": {"type": "string", "minLength": 1},
                    "dst_id": {"type": "string", "minLength": 1},
                    "edge_type": {
                        "type": "string",
                        "enum": [
                            "related_to",
                            "supports",
                            "contradicts",
                            "derived_from",
                            "supersedes",
                            "applies_to",
                            "learned_from",
                            "validated_by",
                        ],
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["src_id", "dst_id", "edge_type", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_remove_block_edge",
            "description": "CAS-remove a typed graph edge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edge_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["edge_id", "expected_revision", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_get_active_mask",
            "description": "Read the exact-scope durable active-memory mask.",
            "parameters": {
                "type": "object",
                "properties": {"mask_id": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_set_active_mask",
            "description": (
                "Select the bounded context that UM-MEM must inject automatically before later "
                "rounds in this exact session/task scope. Use after recall or a write reveals "
                "context that should remain in working focus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_block_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "token_budget": {
                        "type": "integer",
                        "minimum": 0,
                        "default": default_recall_max_tokens,
                    },
                    "activation_reason": {"type": "string", "minLength": 1},
                    "context_refs": {"type": "array", "items": {"type": "string"}},
                    "pinned_record_ids": {"type": "array", "items": {"type": "string"}},
                    "selectors": {"type": "array", "items": {"type": "object"}},
                    "priorities": {"type": "array", "items": {"type": "object"}},
                    "block_token_budgets": {"type": "array", "items": {"type": "object"}},
                    "expires_at": {"type": "string"},
                },
                "required": ["selected_block_ids", "activation_reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_amend_active_mask",
            "description": (
                "CAS-change the exact-scope active mask when working focus changes; subsequent "
                "rounds automatically receive the amended bounded context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "activation_reason": {"type": "string", "minLength": 1},
                    "selected_block_ids": {"type": "array", "items": {"type": "string"}},
                    "context_refs": {"type": "array", "items": {"type": "string"}},
                    "pinned_record_ids": {"type": "array", "items": {"type": "string"}},
                    "selectors": {"type": "array", "items": {"type": "object"}},
                    "priorities": {"type": "array", "items": {"type": "object"}},
                    "token_budget": {"type": "integer", "minimum": 0},
                    "block_token_budgets": {"type": "array", "items": {"type": "object"}},
                    "expires_at": {"type": "string"},
                },
                "required": ["mask_id", "expected_revision", "activation_reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_transition_active_mask",
            "description": "CAS-suspend, resume, or deactivate an active-memory mask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "transition": {
                        "type": "string",
                        "enum": ["suspend", "resume", "deactivate"],
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["mask_id", "expected_revision", "transition", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_resolve_active_mask",
            "description": "Materialize the current exact-scope mask into an advisory lease.",
            "parameters": {
                "type": "object",
                "properties": {"mask_id": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_observe_maintenance",
            "description": (
                "Record typed advisory maintenance evidence; this never mutates topology."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "observation_type": {
                        "type": "string",
                        "enum": [
                            "capacity_pressure",
                            "low_usage",
                            "high_revision_count",
                            "many_conflicts",
                            "active_reference_present",
                            "consolidation_available",
                        ],
                    },
                    "values": {"type": "object"},
                    "threshold_revision": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "block_id",
                    "observation_type",
                    "values",
                    "threshold_revision",
                    "evidence_ids",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_consolidate_observations",
            "description": (
                "Consolidation requires a host-issued supervisor or operator authority. "
                "This parent-scoped provider returns host_authority_required; submit the "
                "source IDs and verification reports through the authorized host API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ids": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string"},
                    },
                    "content": {"type": "string", "minLength": 1, "maxLength": 1024},
                    "title": {"type": "string", "minLength": 1, "maxLength": 96},
                    "reason": {"type": "string", "minLength": 1},
                    "verification_reports": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "object"},
                    },
                    "semantic_verification_report": {"type": "object"},
                    "requested_block_id": {"type": "string", "minLength": 1},
                },
                "required": [
                    "source_ids",
                    "content",
                    "title",
                    "reason",
                    "verification_reports",
                    "semantic_verification_report",
                    "requested_block_id",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_propose_promotion",
            "description": (
                "Propose publication of one exact candidate revision into a model-chosen leaf. "
                "The host independently checks the full content and typed verification reports "
                "and runs the tribunal; this tool cannot set trust or authority."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "minLength": 1},
                    "expected_source_revision_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "target_block_id": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "verification_reports": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "object"},
                    },
                },
                "required": [
                    "memory_id",
                    "expected_source_revision_id",
                    "reason",
                    "target_block_id",
                    "verification_reports",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_recent_verification",
            "description": (
                "List compact verifier-owned reports created from live host-observed "
                "test results in this session. It returns hashes and typed report IDs, "
                "never raw test output."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "um_mem_provider_export_graph",
            "description": ("Export the active workspace's scoped record graph for inspection."),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "um_mem_provider_export_obsidian",
            "description": "Export the active workspace's UM-MEM memory to the configured Obsidian vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_title": {"type": "string"},
                },
            },
        },
    ]
