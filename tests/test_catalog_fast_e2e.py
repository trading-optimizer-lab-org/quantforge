"""Focused consumer tests for the authenticated canary recovery hook."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import shutil
import pytest
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from aurora.infra.sp500_megarun.catalog_request_contract import CatalogRunRequestV1
from aurora.infra.sp500_megarun.catalog_run_request import parse_catalog_run_request
from aurora.infra.sp500_megarun.catalog_campaign_definition_contract import (
    parse_catalog_campaign_definition_bytes,
)
from aurora.infra.sp500_megarun.catalog_fast_path import (
    CatalogFastLaunchDecisionV1,
    CatalogPreparationIdentityV1,
)
from aurora.infra.sp500_megarun.catalog_recovery_blocks import resolve_recovery_block
from aurora.infra.sp500_megarun.catalog_optimization_contract import (
    RunOptimizationContractV1,
)
from aurora.infra.sp500_megarun.catalog_worker_failure import (
    decide_catalog_worker_recovery,
)
from aurora.infra.github_performance.contracts import canonical_sha256
from aurora.infra.sp500_megarun import catalog_fast_canary_acceptance as canary

from aurora.infra.sp500_megarun.catalog_fast_canary_acceptance import (
    build_canary_acceptance_token,
    canary_policy_sha256,
    load_canary_recovery_policy,
    should_inject_canary_failure,
)
from aurora.infra.sp500_megarun.catalog_worker_failure import (
    classify_worker_exception,
)


_CONTEXT = "a" * 64
_REQUEST = "b" * 64
_PLAN = "c" * 64
_BLOCK = "d" * 64
_ATTEMPT = "authority:worker:003:attempt:1"
_IDS = ("SCV1-000", "SCV1-001")
_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_INPUTS_ROOT = os.environ.get("CATALOG_FAST_E2E_INPUTS")
_PREPARED_INPUTS = Path(_LOCAL_INPUTS_ROOT) if _LOCAL_INPUTS_ROOT else None
_COMPONENT_STORE_ROOT = os.environ.get("CATALOG_FAST_E2E_COMPONENT_STORE")
_RUNTIME_INPUT_PACK = os.environ.get("CATALOG_FAST_E2E_RUNTIME_INPUT_PACK")
_SEALED_PLAN = (
    _PREPARED_INPUTS
    / "local-worker-qualification/reduction-qualification/local-terminal-integration/plan"
    if _PREPARED_INPUTS is not None
    else None
)
_CANARY_ACTOR = "aurora-catalog-request-f10c7b40e1[bot]"
_EXPECTED_RECOVERY_BLOCK_ID = (
    "03d1b5a5a4de98af003be4e6f74ccb2bbc868c7520c1c04dd64f41467caee74a"
)
_PRESERVED_WORKER_HASHES = {
    "worker-0/checkpoint_chain_manifest.json": "c6917010515eb2252058ee8670fb7ba39263b8051ddea2e9c2f734752b76d8cb",
    "worker-0/receipt.json": "13064c78ed4a4259af8ced928c2f5e37492e15f2cf47ba7d97a1259de338e052",
    "worker-0/resource_summary.json": "059f747fc4f0c367b2de51456bc1a1924320d204da7fb0807de5171f8af41a7b",
    "worker-0/resource_telemetry.parquet": "f20c77c1853c1183ca1e636fdaee5e040e197b73b9efce6009c8dc5a61a872f6",
    "worker-0/results.parquet": "c0cbe7a949d550548e69b50e3cc6f96c451fd1182c44d79c45f5ef1ed50d6ea4",
    "worker-0/selected_results.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "worker-0/shard_attempt_manifest.json": "9637448f82846557ecee2b8fdfac2bb75d318f5ac2160b6b9b12856edab874be",
    "worker-0/unit_attempts.parquet": "c9e4baefe58f19bae41b2fc2a514e21cb8eb9f567c97a7afc8dd3a06bcbc995f",
    "worker-1/checkpoint_chain_manifest.json": "45cd77b29931767f0fab49f8d8f890bd9e43d00c04dedc2d1b93bb3d8c3db3d4",
    "worker-1/receipt.json": "3acdda6f0e6a8b4902760d71243b6b375d2f9cadf4afa1e2444bbdf90db31eba",
    "worker-1/resource_summary.json": "ac6526436664b51d772a21c11283cb364f665bcb5012713448b3b1b10025b9b4",
    "worker-1/resource_telemetry.parquet": "b30a4dfb495a7f0083384781339ddf35a1b7b7a9c9fd539618a019f8de57745c",
    "worker-1/results.parquet": "07b086ee99f50697f4e78916b47a3826eb7ee887c80c28d21753693dfb07b2e7",
    "worker-1/shard_attempt_manifest.json": "279f62e5d097e7b95e3b76c24c32d12b18b3309e0bd5f64d769f217fdd026eb0",
    "worker-1/unit_attempts.parquet": "687a5a61856a4a0d5020dbcc2136c3e60b66853e830306ad738455ffeb6478e8",
    "worker-2/checkpoint_chain_manifest.json": "49fdac79ce28a3ee8ef502149fce0adafedcd5a2c8beb32892b0c9e21a515f85",
    "worker-2/receipt.json": "3caf4bf35583d1b7e55e97c7d7fe782db217c6ffcde8038d7f765534eb8966a8",
    "worker-2/resource_summary.json": "0cb94073121258388c15a288a43029b304a2a0b150f9198055735142cd2b3251",
    "worker-2/resource_telemetry.parquet": "6df0fc67241515d4bd7c01e34b666f793c22e33e5447c60ab9c54617cbcc9567",
    "worker-2/results.parquet": "3c19e30adb45a878bea977470b5cd6cd6c5b4bc1b30b81b9df72fb183bf4ee0f",
    "worker-2/shard_attempt_manifest.json": "8682dae42759ef0b24462f58a38eb6a347a82a29920ad0cd53b363691fcfb103",
    "worker-2/unit_attempts.parquet": "aad2cc18676140fa1bb1798e409628965c0c9a912f22c677be6e41d0914940d7",
}
_LOCAL_E2E_REQUIRED_RELATIVE_PATHS = (
    "execution-inputs/resolved_contract.json",
    "execution-inputs/resume_work_manifest.json",
    "execution-inputs/selected-config.json",
    "execution-inputs/recipe-dag/recipe_dag.parquet",
    "execution-inputs/recipe-dag/recipe_dag_manifest.json",
    "local-worker-qualification/run_plan.json",
    "local-worker-qualification/assignment-3.json",
    "local-worker-qualification/descriptor-3.json",
    "local-worker-qualification/checkpoint_policy.json",
    "local-worker-qualification/recovery-transport/recovery-descriptors/worker-003.json",
)
_LOCAL_E2E_AVAILABLE = (
    _PREPARED_INPUTS is not None
    and _COMPONENT_STORE_ROOT is not None
    and _RUNTIME_INPUT_PACK is not None
    and all(
        (_PREPARED_INPUTS / relative).is_file()
        for relative in _LOCAL_E2E_REQUIRED_RELATIVE_PATHS
    )
    and Path(_COMPONENT_STORE_ROOT).is_dir()
    and Path(_RUNTIME_INPUT_PACK).is_dir()
)


class RealCanaryFixture(TypedDict):
    plan_root: Path
    context_path: Path
    request: CatalogRunRequestV1
    context_sha256: str
    plan_sha256: str
    admission_token: str
    commit: str
    descriptor_path: Path
    assignment_path: Path
    recovery_descriptor_path: Path
    checkpoint_path: Path
    decision: CatalogFastLaunchDecisionV1
    authority_id: str
    campaign_id: str
    attempt_id: str


def _token(*, generation: int = 5, campaign_key: str = "catalog-fast-canary-v1") -> str:
    return build_canary_acceptance_token(
        campaign_key=campaign_key,
        generation=generation,
        context_sha256=_CONTEXT,
        request_sha256=_REQUEST,
        execution_plan_sha256=_PLAN,
    )


def _scope(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "enabled": "true",
        "campaign_key": "catalog-fast-canary-v1",
        "generation": 5,
        "context_sha256": _CONTEXT,
        "request_sha256": _REQUEST,
        "execution_plan_sha256": _PLAN,
        "acceptance_token": _token(),
        "worker_id": 3,
        "total_workers": 4,
        "checkpoint_slot_index": 1,
        "checkpoint_slot_count": 1,
        "strategy_ids": _IDS,
        "attempt_id": _ATTEMPT,
        "recovery_block_id": _BLOCK,
    }
    values.update(overrides)
    return values


def _reseal_plan(root: Path, *, request_sha256: str, decision_sha256: str,
                 protected_commit_sha: str) -> str:
    """Recompute the test copy's byte manifest after binding it to this run."""
    receipt_path = root / "execution_plan_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        request_sha256=request_sha256,
        decision_sha256=decision_sha256,
        protected_commit_sha=protected_commit_sha,
    )
    manifest = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == receipt_path:
            continue
        data = path.read_bytes()
        manifest.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    receipt["content_manifest"] = manifest
    receipt["content_manifest_sha256"] = canonical_sha256(tuple(manifest))
    identity = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(identity)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return str(receipt["execution_plan_sha256"])


def _real_canary_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> RealCanaryFixture:
    """Build an ephemeral authenticated gate around the prepared real worker inputs."""
    from aurora.tests.test_catalog_fast_canary_authentication import sign_canary_request

    assert _PREPARED_INPUTS is not None
    assert _SEALED_PLAN is not None
    plan_root = tmp_path / "sealed-plan"
    shutil.copytree(_SEALED_PLAN, plan_root)
    shutil.copy2(
        _PREPARED_INPUTS / "execution-inputs/resolved_contract.json",
        plan_root / "resolved_contract.json",
    )
    shutil.copy2(
        _PREPARED_INPUTS / "local-worker-qualification/run_plan.json",
        plan_root / "run_plan.json",
    )
    shutil.copy2(
        _PREPARED_INPUTS / "execution-inputs/resume_work_manifest.json",
        plan_root / "resume_work_manifest.json",
    )
    resolved = RunOptimizationContractV1.model_validate_json(
        (plan_root / "resolved_contract.json").read_text(encoding="utf-8")
    )
    run_plan_path = plan_root / "run_plan.json"
    run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    run_plan["contract_sha256"] = resolved.contract_sha256
    # The source bundle is qualification-only; the authenticated canary is a
    # full-plan gate, so only this ephemeral copy is promoted for the test.
    run_plan["qualification_only"] = False
    admission_token = hashlib.sha256(
        b"aurora-catalog-admission-v1\0"
        + resolved.contract_sha256.encode("ascii")
        + str(run_plan["evidence_sha256"]).encode("ascii")
    ).hexdigest()
    run_plan["admission_token_sha256"] = admission_token
    run_plan_path.write_text(
        json.dumps(run_plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    # Keep the historical fixture's transport copies aligned with the root
    # inputs required by the current verifier; only the temporary copy changes.
    payload_manifest_path = plan_root / "payload_bundle_manifest.json"
    payload_manifest = json.loads(payload_manifest_path.read_text(encoding="utf-8"))
    for payload in payload_manifest["payloads"]:
        if payload["member"] not in {"run_plan.json", "resume_work_manifest.json"}:
            continue
        target = plan_root / "payload_artifacts" / payload["artifact"] / payload["member"]
        shutil.copy2(plan_root / payload["member"], target)
        data = target.read_bytes()
        payload.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    payload_manifest["content_sha256"] = canonical_sha256(
        {key: value for key, value in payload_manifest.items() if key != "content_sha256"}
    )
    payload_manifest_path.write_text(
        json.dumps(payload_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    test_repo = tmp_path / "authenticated-repo"
    (test_repo / "config/catalog_campaign_definitions").mkdir(parents=True)
    for relative in (
        "catalog_controller_actors_v1.json",
        "catalog_campaign_registry_v1.json",
        "catalog_campaign_definitions/catalog-fast-canary-v1.manifest.json",
        "catalog_fast_canary_optimization_policy_v1.json",
        "sp500_megarun_dehb_campaign_v1.json",
        "catalog_fast_canary_selected_v1.json",
        "sp500_catalog_admission_evidence_current_v1.json",
        "sp500_megarun_free_data_240.json",
        "sp500_megarun_feature_contract_240.json",
    ):
        shutil.copy2(_ROOT / "config" / relative, test_repo / "config" / relative)
    shutil.copytree(
        _ROOT / "tests/fixtures/catalog_fast_canary_v1/production-catalog",
        test_repo / "tests/fixtures/catalog_fast_canary_v1/production-catalog",
    )

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path = test_repo / "config/requester.pem"
    public_path.write_bytes(public)
    monkeypatch.setattr(canary, "_REPOSITORY_ROOT", test_repo)
    monkeypatch.setattr(canary, "_PUBLIC_KEY_PATH", public_path)

    definition = parse_catalog_campaign_definition_bytes(
        (_ROOT / "config/catalog_campaign_definitions/catalog-fast-canary-v1.manifest.json").read_bytes()
    )
    request = sign_canary_request(
        private,
        definition_sha256=definition.campaign_definition_sha256,
        generation=5,
    )
    commit = "a" * 40
    identity = CatalogPreparationIdentityV1(
        schema_version="1",
        campaign_key="catalog-fast-canary-v1",
        engine_id="optimized_catalog_v1",
        protected_commit_sha=commit,
        campaign_definition_sha256=definition.campaign_definition_sha256,
        scientific_contract_sha256="57a24398bba9779f2095d20dc50f15975cd04949964055ae322411a3d57906a2",
        dependency_lock_sha256="0" * 64,
        optimization_policy_sha256="0" * 64,
        data_contract_sha256="0" * 64,
        feature_contract_sha256="0" * 64,
        catalog_manifest_sha256="0" * 64,
        selected_config_sha256="0" * 64,
    )
    gate = tmp_path / "gate"
    gate.mkdir()
    context: dict[str, object] = {
        "schema_version": "1",
        "document_type": "catalog_fast_request_context_v1",
        "request_mode": "admit_new",
        "issue_number": 123,
        "request": request.model_dump(mode="json"),
        "protected_commit_sha": commit,
        "actor": _CANARY_ACTOR,
        "identity": identity.model_dump(mode="json"),
        "logical_recipe_count": 8,
    }
    context_sha256 = canonical_sha256(context)
    context["content_sha256"] = context_sha256
    context_path = gate / "context.json"
    context_path.write_text(json.dumps(context, sort_keys=True), encoding="utf-8")

    decision = CatalogFastLaunchDecisionV1.create(
        state="QUEUED",
        reason_code="CATALOG_FAST_PATH_ADMITTED",
        request_sha256=request.request_sha256,
        submission_key_sha256=request.submission_key_sha256,
        campaign_key=request.campaign_key,
        prepared_receipt_sha256=None,
        selected_workers=4,
        launch_required=True,
        existing_run_id=None,
        decided_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        expires_at=datetime(2026, 9, 10, 0, 30, tzinfo=timezone.utc),
    )
    (gate / "catalog-fast-decision-v1.json").write_text(
        decision.model_dump_json(), encoding="utf-8"
    )

    binding_path = plan_root / "controller_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    plan_receipt = json.loads(
        (plan_root / "execution_plan_receipt.json").read_text(encoding="utf-8")
    )
    plan_receipt["admission_token_sha256"] = admission_token
    (plan_root / "execution_plan_receipt.json").write_text(
        json.dumps(plan_receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    binding["binding"].update(
        request_sha256=request.request_sha256,
        campaign_definition_sha256=definition.campaign_definition_sha256,
        execution_plan_sha256=plan_receipt["execution_plan_sha256"],
        protected_commit_sha=commit,
    )
    binding_path.write_text(
        json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    plan_sha256 = _reseal_plan(
        plan_root,
        request_sha256=request.request_sha256,
        decision_sha256=decision.decision_sha256,
        protected_commit_sha=commit,
    )
    monkeypatch.setenv("GITHUB_SHA", commit)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    from aurora.infra.sp500_megarun.dehb_numeric_runtime import DEHB_NUMERIC_ENV

    for name, value in DEHB_NUMERIC_ENV.items():
        monkeypatch.setenv(name, value)
    return {
        "plan_root": plan_root,
        "context_path": context_path,
        "request": request,
        "context_sha256": context_sha256,
        "plan_sha256": plan_sha256,
        "admission_token": admission_token,
        "commit": commit,
        "descriptor_path": plan_root / "payload_artifacts/catalog-recipe-descriptors-bundle-3e25f8214bc07f9a7f34-00/recipe/worker-003.json",
        "assignment_path": plan_root / "payload_artifacts/catalog-recipe-assignments-bundle-3e25f8214bc07f9a7f34-00/recipe/worker-003.json",
        "recovery_descriptor_path": _PREPARED_INPUTS / "local-worker-qualification/recovery-transport/recovery-descriptors/worker-003.json",
        "checkpoint_path": plan_root / "checkpoint_policy.json",
        "decision": decision,
        "authority_id": "737218cb-07e1-5d94-a182-1d5bc98202ba",
        "campaign_id": "815b88b3afc713b6b7c8b7f1097ce77867dbfa9ea1beab2da0b61281d4107200",
        "attempt_id": "737218cb-07e1-5d94-a182-1d5bc98202ba:worker:003:attempt:1",
    }


def test_policy_is_literal_and_targets_prepared_four_by_two_shape() -> None:
    policy = load_canary_recovery_policy()
    assert canary_policy_sha256()
    assert policy["campaign_key"] == "catalog-fast-canary-v1"
    assert policy["expected_worker_count"] == 4
    assert policy["expected_strategy_count"] == 2
    assert policy["target_worker_id"] == 3
    assert policy["expected_checkpoint_slot_count"] == 1


def test_only_authenticated_generation_five_target_block_is_selected() -> None:
    with pytest.raises(ValueError, match="AUTHENTICATED_INPUTS_REQUIRED"):
        should_inject_canary_failure(**_scope())


@pytest.mark.parametrize(
    "overrides",
    [
        {"campaign_key": "sp500-optimized-catalog-v1"},
        {"generation": 2, "acceptance_token": _token(generation=2)},
        {"generation": 3, "acceptance_token": _token(generation=3)},
        {"generation": 4, "acceptance_token": _token(generation=4)},
        {"generation": 6, "acceptance_token": _token(generation=6)},
        {"worker_id": 2},
        {"checkpoint_slot_index": 2},
        {"checkpoint_slot_count": 2},
        {"strategy_ids": ("SCV1-000",)},
        {"attempt_id": "authority:worker:003:attempt:2"},
        {"recovery_block_id": None},
        {"context_sha256": "not-a-hash"},
        {"acceptance_token": "e" * 64},
    ],
)
def test_scope_or_binding_mutation_fails_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        should_inject_canary_failure(**_scope(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": ""},
        {"enabled": "false"},
        {"enabled": "false", "generation": 5},
        {
            "enabled": "",
            "campaign_key": "",
            "generation": 0,
            "context_sha256": "",
            "request_sha256": "",
            "execution_plan_sha256": "",
            "acceptance_token": "",
        },
    ],
)
def test_normal_and_recovery_invocations_do_not_activate_hook(
    overrides: dict[str, object],
) -> None:
    values = _scope(**overrides)
    if values["enabled"] in {"", "false"}:
        values.update(
            campaign_key="",
            context_sha256="",
            request_sha256="",
            execution_plan_sha256="",
            acceptance_token="",
        )
    assert should_inject_canary_failure(**values) is False


def test_controlled_exception_uses_existing_transient_classifier() -> None:
    reason, exit_code, exception_type = classify_worker_exception(
        ConnectionResetError("CATALOG_CANARY_CONTROLLED_TRANSIENT_FAILURE")
    )
    assert (reason, exit_code, exception_type) == (
        "CONNECTION_RESET",
        1,
        "ConnectionResetError",
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _LOCAL_E2E_AVAILABLE,
    reason=(
        "local E2E bundle unavailable; set CATALOG_FAST_E2E_INPUTS, "
        "CATALOG_FAST_E2E_COMPONENT_STORE, and "
        "CATALOG_FAST_E2E_RUNTIME_INPUT_PACK"
    ),
)
def test_controlled_exception_keeps_deliberate_marker_in_existing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aurora.infra.sp500_megarun.catalog_worker_failure import (
        CatalogWorkerFailureReceiptV1,
    )
    from scripts.run_catalog_recipe_worker_guarded import execute_guarded

    assert _PREPARED_INPUTS is not None
    assert _COMPONENT_STORE_ROOT is not None
    assert _RUNTIME_INPUT_PACK is not None
    prepared_inputs = _PREPARED_INPUTS
    component_store = Path(_COMPONENT_STORE_ROOT)
    runtime_input_pack = Path(_RUNTIME_INPUT_PACK)
    fixture = _real_canary_fixture(tmp_path, monkeypatch)
    failure = tmp_path / "failure.json"

    command = [
        "--campaign-contract",
        str(_ROOT / "config/sp500_megarun_dehb_campaign_v1.json"),
        "--catalog-dir",
        str(prepared_inputs / "catalog"),
        "--selected-config",
        str(prepared_inputs / "execution-inputs/selected-config.json"),
        "--component-store", str(component_store),
        "--runtime-input-pack",
        str(runtime_input_pack),
        "--resolved-contract",
        str(fixture["plan_root"] / "resolved_contract.json"),
        "--run-plan",
        str(fixture["plan_root"] / "run_plan.json"),
        "--resume-work-manifest",
        str(prepared_inputs / "execution-inputs/resume_work_manifest.json"),
        "--recipe-dag",
        str(prepared_inputs / "execution-inputs/recipe-dag/recipe_dag.parquet"),
        "--recipe-dag-manifest",
        str(prepared_inputs / "execution-inputs/recipe-dag/recipe_dag_manifest.json"),
        "--admission-token",
        str(fixture["admission_token"]),
        "--shard-index", "3",
        "--total-shards", "4",
        "--payload-descriptor", str(fixture["descriptor_path"]),
        "--assignment-file", str(fixture["assignment_path"]),
        "--checkpoint-policy", str(fixture["checkpoint_path"]),
        "--checkpoint-slot-index", "1",
        "--checkpoint-slot-count", "1",
        "--output-dir", str(tmp_path / "worker-3"),
        "--canary-acceptance-enabled", "true",
        "--canary-acceptance-campaign-key", "catalog-fast-canary-v1",
        "--canary-acceptance-generation", "5",
        "--canary-acceptance-context-sha256", str(fixture["context_sha256"]),
        "--canary-acceptance-request-sha256", fixture["request"].request_sha256,
        "--canary-acceptance-plan-sha256", str(fixture["plan_sha256"]),
        "--canary-acceptance-token", build_canary_acceptance_token(
            campaign_key="catalog-fast-canary-v1",
            generation=5,
            context_sha256=str(fixture["context_sha256"]),
            request_sha256=fixture["request"].request_sha256,
            execution_plan_sha256=str(fixture["plan_sha256"]),
        ),
        "--canary-acceptance-context", str(fixture["context_path"]),
        "--canary-acceptance-sealed-plan", str(fixture["plan_root"]),
    ]
    result = execute_guarded(
        [
            "--failure-receipt", str(failure),
            "--authority-id", str(fixture["authority_id"]),
            "--campaign-id", str(fixture["campaign_id"]),
            "--execution-plan-sha256", str(fixture["plan_sha256"]),
            "--protected-commit-sha", str(fixture["commit"]),
            "--worker-id", "3",
            "--attempt-id", str(fixture["attempt_id"]),
            "--",
            *command,
        ],
    )
    receipt = CatalogWorkerFailureReceiptV1.model_validate_json(
        failure.read_text()
    )
    assert result == 1
    assert receipt.reason_code == "CONNECTION_RESET"
    assert receipt.failure_class.value == "transient_network"
    assert receipt.worker_id == 3
    assert receipt.attempt_id == fixture["attempt_id"]
    assert receipt.source_error_code == (
        "CATALOG_CANARY_CONTROLLED_TRANSIENT_FAILURE"
    )

    assignment = json.loads(Path(fixture["assignment_path"]).read_text())
    checkpoint_policy = json.loads(Path(fixture["checkpoint_path"]).read_text())
    recovered_block = resolve_recovery_block(
        checkpoint_policy,
        science_sha256=receipt.execution_plan_sha256 and checkpoint_policy["science_sha256"],
        worker_id=3,
        slot_index=1,
        strategy_ids=assignment["strategy_ids"],
    )
    assert recovered_block == _EXPECTED_RECOVERY_BLOCK_ID
    recovery = decide_catalog_worker_recovery(
        expected_worker_ids=(0, 1, 2, 3),
        completed_worker_ids=(0, 1, 2),
        failure_receipts=(receipt,),
        current_wave=0,
        max_waves=2,
        now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert recovery.status == "retry"
    assert {item.worker_id: item.action for item in recovery.decisions} == {
        0: "complete", 1: "complete", 2: "complete", 3: "retry"
    }

    recovery_output = tmp_path / "recovery-output"
    baseline_root = prepared_inputs / "local-worker-qualification"
    for worker_id in range(3):
        shutil.copytree(
            baseline_root / f"worker-{worker_id}",
            recovery_output / f"worker-{worker_id}",
        )
    for relative, expected_hash in _PRESERVED_WORKER_HASHES.items():
        baseline = baseline_root / Path(relative)
        assert hashlib.sha256(baseline.read_bytes()).hexdigest() == expected_hash

    retry_command = [
        "--campaign-contract",
        str(_ROOT / "config/sp500_megarun_dehb_campaign_v1.json"),
        "--catalog-dir",
        str(prepared_inputs / "catalog"),
        "--selected-config",
        str(prepared_inputs / "execution-inputs/selected-config.json"),
        "--component-store", str(component_store),
        "--runtime-input-pack", str(runtime_input_pack),
        "--resolved-contract",
        str(fixture["plan_root"] / "resolved_contract.json"),
        "--run-plan",
        str(fixture["plan_root"] / "run_plan.json"),
        "--resume-work-manifest",
        str(prepared_inputs / "execution-inputs/resume_work_manifest.json"),
        "--recipe-dag",
        str(prepared_inputs / "execution-inputs/recipe-dag/recipe_dag.parquet"),
        "--recipe-dag-manifest",
        str(prepared_inputs / "execution-inputs/recipe-dag/recipe_dag_manifest.json"),
        "--admission-token",
        str(fixture["admission_token"]),
        "--shard-index", "3",
        "--total-shards", "4",
        "--payload-descriptor", str(fixture["recovery_descriptor_path"]),
        "--assignment-file", str(fixture["assignment_path"]),
        "--checkpoint-policy", str(fixture["checkpoint_path"]),
        "--checkpoint-slot-index", "1",
        "--checkpoint-slot-count", "1",
        "--output-dir", str(recovery_output / "worker-3"),
        "--canary-acceptance-enabled", "false",
    ]
    retry_result = execute_guarded(
        [
            "--failure-receipt", str(tmp_path / "retry-failure.json"),
            "--authority-id", str(fixture["authority_id"]),
            "--campaign-id", str(fixture["campaign_id"]),
            "--execution-plan-sha256", str(fixture["plan_sha256"]),
            "--protected-commit-sha", str(fixture["commit"]),
            "--worker-id", "3",
            "--attempt-id",
            "737218cb-07e1-5d94-a182-1d5bc98202ba:worker:003:attempt:2",
            "--",
            *retry_command,
        ],
    )
    assert retry_result == 0
    for relative, expected_hash in _PRESERVED_WORKER_HASHES.items():
        preserved = recovery_output / Path(relative)
        assert hashlib.sha256(preserved.read_bytes()).hexdigest() == expected_hash
    retry_manifest = json.loads(
        (recovery_output / "worker-3/shard_attempt_manifest.json").read_text()
    )
    assert retry_manifest["attempt_id"] == (
        "737218cb-07e1-5d94-a182-1d5bc98202ba:worker:003:attempt:2"
    )


def test_protected_workflows_wire_gate_to_initial_worker_only() -> None:
    controller = (_ROOT / ".github/workflows/catalog-fast-controller.yml").read_text()
    run = (_ROOT / ".github/workflows/catalog-optimized-run.yml").read_text()
    worker = (_ROOT / ".github/workflows/catalog-optimized-worker.yml").read_text()
    recovery = (_ROOT / ".github/workflows/catalog-recovery-wave.yml").read_text()
    assert "canary_gate_artifact" in controller
    assert "Download the authenticated canary gate context" in run
    assert "canary_acceptance_token" in run
    assert '"policy_sha256": canonical_sha256(policy)' in run
    source = (_ROOT / "scripts/run_sp500_optimized_recipe_worker.py").read_text()
    assert "canary-acceptance-token" in worker
    assert "catalog_fast_canary_acceptance" in source
    assert "ConnectionResetError" in source
    assert "canary_acceptance" not in recovery

    from aurora.infra.github_performance.preflight import load_github_yaml

    parsed = load_github_yaml(_ROOT / ".github/workflows/catalog-optimized-run.yml")
    for component_job in (
        "build_components_a",
        "build_components_b",
        "materialize_cached_components_a",
        "materialize_cached_components_b",
    ):
        assert not any(
            name.startswith("canary_")
            for name in parsed["jobs"][component_job].get("with", {})
        )
    for recipe_job in ("evaluate_a", "evaluate_b", "evaluate_c"):
        assert parsed["jobs"][recipe_job]["with"]["canary_fault_enabled"]


def test_inspect_output_uses_request_property_not_a_dumped_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real inspector's context omits request_sha256 from request.model_dump."""

    from aurora.tests.test_inspect_catalog_fast_request import (
        COMMIT,
        _signed_request,
    )
    from scripts import inspect_catalog_fast_request as inspector

    root = tmp_path / "repo"
    runner = tmp_path / "runner"
    (root / "config").mkdir(parents=True)
    (root / "keys").mkdir()
    runner.mkdir()
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (root / "keys/requester.pem").write_bytes(public)
    (root / "config/catalog_controller_actors_v1.json").write_text(
        '{"request_actors":["requester"],"requester_public_key_path":"keys/requester.pem"}',
        encoding="utf-8",
    )
    title, body = _signed_request(private)
    issue = runner / "issue.json"
    issue.write_text(
        json.dumps(
            {
                "number": 123,
                "title": title,
                "body": body,
                "created_at": "2026-09-04T12:00:00Z",
                "updated_at": "2026-09-04T12:00:00Z",
                "labels": [{"name": "catalog-run-terminal-v1"}],
                "user": {"login": "requester"},
            }
        ),
        encoding="utf-8",
    )
    output = runner / "context.json"
    gh_output = runner / "github-output.txt"
    monkeypatch.setenv("CATALOG_PROTECTED_COMMIT_SHA", COMMIT)
    monkeypatch.setenv("RUNNER_TEMP", str(runner))
    monkeypatch.setattr(
        inspector.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{COMMIT}\n"),
    )

    context = inspector.inspect_request(
        issue_path=issue,
        repo_root=root,
        output_path=output,
        github_output=gh_output,
    )
    request_payload = context["request"]
    assert isinstance(request_payload, dict)
    assert "request_sha256" not in request_payload
    parsed = parse_catalog_run_request(title, body, public)
    reconstructed = CatalogRunRequestV1.model_validate(request_payload)
    assert reconstructed.request_sha256 == parsed.request_sha256
