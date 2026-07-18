"""JSON-like policy input projection for future policy backends."""

from lethyx.contracts.models import ContractBundle, json_ready


def to_policy_input(bundle: ContractBundle) -> dict:
    return json_ready(bundle)
