import inspect
import re
from pathlib import Path

import um_mem
from um_mem import MemoryClient, RememberResult

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs/um-mem/memory_architecture.md",
    REPO_ROOT / "docs/um-mem/operator_guide.md",
    REPO_ROOT / "docs/um-mem/quickstart.md",
    REPO_ROOT / "docs/um-mem/hermes_runtime_integration_user_guide.md",
)
EXPECTED_PUBLIC_API = {
    "ActiveMemoryMask",
    "ActiveMemorySelector",
    "AgentSession",
    "ActivationLease",
    "DeepReadResult",
    "DependencyIssue",
    "DependencyValidation",
    "MemoryClient",
    "MemoryBusyError",
    "BackendCapabilities",
    "PinnedRecordRef",
    "MemoryRetractionReceipt",
    "RecallExplanation",
    "RememberResult",
    "WorkingRecordRef",
    "WorkingRecordSnapshot",
    "WorkingSetRead",
    "WorkingSnapshotBudgetError",
    "recall",
    "remember",
    "__version__",
}


def test_canonical_public_api_and_documented_example_arguments_match_runtime() -> None:
    assert set(um_mem.__all__) == EXPECTED_PUBLIC_API
    assert "memory_id" in RememberResult.__dataclass_fields__
    assert "block_id" not in RememberResult.__dataclass_fields__
    assert "max_tokens" in inspect.signature(MemoryClient.build_context).parameters
    assert "budget" not in inspect.signature(MemoryClient.build_context).parameters

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs/um-mem/quickstart.md").read_text(encoding="utf-8")
    assert "saved.memory_id" in readme
    assert "saved.memory_id" in quickstart
    assert "max_tokens=1200" in readme
    assert "max_tokens=1200" in quickstart


def test_public_docs_have_no_dead_local_markdown_links() -> None:
    for document in PUBLIC_DOCS:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            local_target = target.split("#", 1)[0]
            assert (document.parent / local_target).resolve().exists(), (
                f"dead link in {document.relative_to(REPO_ROOT)}: {target}"
            )


def test_docs_state_clean_store_and_do_not_advertise_removed_runtime_commands() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
    assert "clean-store" in combined
    assert "version-exact" in combined
    assert "migrate-builtins" not in combined
    assert "migrate-all" not in combined
    hermes = (REPO_ROOT / "docs/um-mem/hermes_runtime_integration_user_guide.md").read_text(
        encoding="utf-8"
    )
    assert "um-memory hermes-provider-install" in hermes
    assert "hermes memory setup" in hermes
    assert "--replacement" in hermes
    assert "um-memory hermes-provider-status --json" in hermes
    assert "deep_harness" in hermes
    assert "native_lifecycle_unqualified" in hermes
    assert "um_mem_provider_status" in hermes
