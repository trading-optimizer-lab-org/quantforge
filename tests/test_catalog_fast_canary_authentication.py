"""The canary's public checksum cannot replace its real requester signature."""

from base64 import b64encode
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import pytest
import yaml

from aurora.infra.github_performance.contracts import canonical_sha256
from aurora.infra.sp500_megarun import catalog_fast_canary_acceptance as canary
from aurora.infra.sp500_megarun.catalog_request_contract import (
    CatalogLaunchTicketV1, CatalogRunIntentV1, CatalogRunRequestV1, _attestation_payload,
)


def sign_canary_request(private, *, definition_sha256="c" * 64, generation=5):
    """Ephemeral test signer; never reads or changes any installed credential."""
    ticket = CatalogLaunchTicketV1(
        schema_version="1", request_id="018f47a2-6e91-7c34-8000-000000000003",
        campaign_key="catalog-fast-canary-v1", launch_generation=generation,
        campaign_definition_sha256=definition_sha256, prompt_sha256="4" * 64,
        previous_terminal_request_sha256="5" * 64,
    )
    intent = CatalogRunIntentV1(
        **ticket.model_dump(mode="json"), launch_ticket_sha256=ticket.launch_ticket_sha256,
        authorization="USER_EXPLICITLY_REQUESTED_NEW_CATALOG_RUN", free_resources_only=True,
        automatic_recovery=True, max_same_failure_count=3,
    )
    title = f"[AURORA CATALOG RUN REQUEST] {intent.request_id}"
    signature = private.sign(
        _attestation_payload(title, intent),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
        hashes.SHA256(),
    )
    der = private.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return CatalogRunRequestV1(
        **intent.model_dump(mode="json"), requester_public_key_sha256=sha256(der).hexdigest(),
        requester_attestation_algorithm="rsa-pss-sha256-v1",
        requester_attestation_b64=b64encode(signature).decode("ascii"),
    )


def _public_scope(context_sha256="a" * 64, request_sha256="b" * 64):
    identity = dict(campaign_key="catalog-fast-canary-v1", generation=5,
                    context_sha256=context_sha256, request_sha256=request_sha256,
                    execution_plan_sha256="c" * 64)
    return dict(**identity, enabled="true", acceptance_token=canary.build_canary_acceptance_token(**identity),
                worker_id=3, total_workers=4, checkpoint_slot_index=1, checkpoint_slot_count=1,
                strategy_ids=("strategy-000", "strategy-001"),
                attempt_id="authority:worker:003:attempt:1", recovery_block_id="d" * 64)


def test_self_created_public_hash_does_not_authorize_a_failure():
    with pytest.raises(ValueError, match="AUTHENTICATED_INPUTS_REQUIRED"):
        canary.should_inject_canary_failure(**_public_scope())


@pytest.mark.parametrize("mutation", ("signature", "generation", "key", "commit", "attempt", "context_hash"))
def test_signed_request_cannot_be_replaced_by_rehashed_cli_fields(tmp_path, monkeypatch, mutation):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    trusted = rsa.generate_private_key(public_exponent=65537, key_size=2048) if mutation == "key" else private
    public_path = tmp_path / "ephemeral-public.pem"
    public_path.write_bytes(trusted.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    monkeypatch.setattr(canary, "_PUBLIC_KEY_PATH", public_path)
    monkeypatch.setenv("GITHUB_SHA", "e" * 40)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    request = sign_canary_request(private, generation=4 if mutation == "generation" else 5)
    payload = request.model_dump(mode="json")
    if mutation == "signature":
        payload["requester_attestation_b64"] = b64encode(bytes(256)).decode("ascii")
    if mutation == "generation":
        payload["launch_generation"] = 5
    request_hash = CatalogRunRequestV1.model_validate(payload).request_sha256
    context = dict(schema_version="1", document_type="catalog_fast_request_context_v1",
                   request_mode="admit_new", issue_number=300, request=payload,
                   protected_commit_sha="e" * 40, actor="aurora-catalog-request-f10c7b40e1[bot]")
    context["content_sha256"] = canonical_sha256(context)
    scope = _public_scope(context["content_sha256"], request_hash)
    if mutation == "commit":
        monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    if mutation == "attempt":
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    if mutation == "context_hash":
        context["issue_number"] = 301
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    # None of these cases may reach the plan or scientific input reader.
    missing = tmp_path / "must-not-be-read"
    scope.update(request_context_path=context_path, sealed_plan_root=missing,
                 resolved_contract_path=missing, run_plan_path=missing,
                 payload_descriptor_path=missing, assignment_path=missing,
                 checkpoint_policy_path=missing)
    with pytest.raises(ValueError, match="ATTESTATION_INVALID|CONTEXT_INVALID"):
        canary.should_inject_canary_failure(**scope)


def test_missing_gate_evidence_cannot_be_replaced_by_a_valid_checksum(tmp_path):
    scope = _public_scope()
    missing = tmp_path / "missing.json"
    scope.update(request_context_path=missing, sealed_plan_root=tmp_path,
                 resolved_contract_path=missing, run_plan_path=missing,
                 payload_descriptor_path=missing, assignment_path=missing,
                 checkpoint_policy_path=missing)
    with pytest.raises(ValueError, match="DOCUMENT_INVALID"):
        canary.should_inject_canary_failure(**scope)


def test_nonselected_workers_do_not_read_canary_files(tmp_path):
    scope = _public_scope()
    scope.update(enabled="false", campaign_key="", context_sha256="", request_sha256="",
                 execution_plan_sha256="", acceptance_token="", request_context_path=tmp_path / "absent")
    assert canary.should_inject_canary_failure(**scope) is False


@pytest.mark.parametrize("generation", (2, 3, 4, 5, 6))
def test_actual_workflow_bootstrap_needs_only_stdlib(tmp_path, generation):
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/catalog-optimized-run.yml").read_text("utf-8"))
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    script = next(step["run"] for step in steps if step.get("id") == "canary")
    source = script.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    request = sign_canary_request(private, generation=generation)
    context = dict(schema_version="1", document_type="catalog_fast_request_context_v1",
                   request_mode="admit_new", protected_commit_sha="e" * 40,
                   request=request.model_dump(mode="json"))
    context["content_sha256"] = canonical_sha256(context)
    gate = tmp_path / "canary-gate"
    gate.mkdir()
    (gate / "catalog-fast-request-context.json").write_text(json.dumps(context), encoding="utf-8")
    output = tmp_path / "output.txt"
    environment = dict(os.environ, GITHUB_OUTPUT=str(output), RUNNER_TEMP=str(tmp_path),
                       CANARY_GATE_ARTIFACT="catalog-fast-gate-300",
                       EXPECTED_PROTECTED_COMMIT_SHA="e" * 40,
                       EXPECTED_REQUEST_SHA256=request.request_sha256,
                       EXPECTED_PLAN_SHA256="c" * 64)
    result = subprocess.run([sys.executable, "-I", "-S", "-c", source], cwd=root,
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text("utf-8").splitlines())
    assert values["fault_enabled"] == ("true" if generation == 5 else "false")
    if generation == 5:
        assert values["acceptance_token"] == canary.build_canary_acceptance_token(
            campaign_key="catalog-fast-canary-v1", generation=5,
            context_sha256=context["content_sha256"], request_sha256=request.request_sha256,
            execution_plan_sha256="c" * 64,
        )
    else:
        assert values["acceptance_token"] == ""


def test_worker_workflow_keeps_authentication_arguments_in_one_shell_command():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/catalog-optimized-worker.yml").read_text("utf-8"))
    steps = workflow["jobs"]["evaluate"]["steps"]
    command = next(step["run"] for step in steps if step.get("id") == "compute_1")
    assert "\n" not in command
    arguments = shlex.split(command)
    for flag in ("--canary-acceptance-token", "--canary-acceptance-context",
                 "--canary-acceptance-sealed-plan"):
        assert arguments.count(flag) == 1
        assert arguments[arguments.index(flag) + 1].startswith("$")
