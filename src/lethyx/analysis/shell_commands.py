"""Lightweight shell command validation for proof contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellCommandIssue:
    code: str
    message: str


def validate_argv(
    argv: tuple[str, ...] | list[str], *, shell: bool = False
) -> list[ShellCommandIssue]:
    """Return conservative issues for command argv arrays."""

    issues: list[ShellCommandIssue] = []
    if shell:
        issues.append(
            ShellCommandIssue(
                "shell_command_not_portable",
                "Proof commands should use argv arrays with shell=false.",
            )
        )
    if not argv:
        issues.append(
            ShellCommandIssue(
                "empty_proof_command",
                "Proof command must contain at least one executable.",
            )
        )
    return issues
