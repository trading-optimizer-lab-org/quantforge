from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from aurora.infra.github_performance.preflight import load_github_yaml


ROOT = Path(__file__).parents[1]
ACTION_PATH = (
    ROOT / ".github" / "actions" / "aurora-runtime-setup" / "action.yml"
)
WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "_aurora-future-run-v3.yml"
)
POLICY_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "github-performance-policy.yml"
)
RECOVERY_PLAN_ACTION_PATH = (
    ROOT / ".github" / "actions" / "aurora-shard-recovery-plan" / "action.yml"
)
RETRY_SHARD_ACTION_PATH = (
    ROOT / ".github" / "actions" / "aurora-retry-shard" / "action.yml"
)
MERGE_LEVEL_ACTION_PATH = (
    ROOT / ".github" / "actions" / "aurora-merge-level" / "action.yml"
)
MERGE_ONLY_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "github-performance-merge-only.yml"
)
REPLAN_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "github-performance-replan.yml"
)
REFERENCE_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "github-performance-reference.yml"
)
VALIDATION_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "github-performance-validation.yml"
)
CAPACITY_CALIBRATION_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "catalog-capacity-calibration.yml"
)
CATALOG_OPTIMIZED_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "catalog-optimized-run.yml"
)


def _load_action() -> dict[str, Any]:
    return yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))


def _locked_action(name: str) -> str:
    lock = json.loads(
        (ROOT / "config" / "official_actions_lock.json").read_text(
            encoding="utf-8"
        )
    )
    return f"{name}@{lock[name]}"


def test_runtime_setup_is_composite_and_pinned() -> None:
    action = _load_action()
    assert action["runs"]["using"] == "composite"
    uses = [
        step["uses"]
        for step in action["runs"]["steps"]
        if "uses" in step
    ]
    assert _locked_action("actions/setup-python") in uses
    assert all(
        value.rsplit("@", 1)[-1].isalnum() and
        len(value.rsplit("@", 1)[-1]) == 40
        for value in uses
    )


def test_runtime_setup_requires_one_exact_runtime_object_and_lock() -> None:
    action = _load_action()
    required = {
        "python-version",
        "runtime-mode",
        "runtime-object-path",
        "runtime-manifest-path",
        "dependency-lock-path",
        "expected-runtime-identity-sha256",
        "expected-source-sha",
        "expected-numeric-profile-sha256",
    }
    assert set(action["inputs"]) == required
    assert all(action["inputs"][name]["required"] is True for name in required)


def test_runtime_setup_installs_offline_without_resolution_or_building() -> None:
    text = ACTION_PATH.read_text(encoding="utf-8")
    assert "--no-index" in text
    assert "--require-hashes" in text
    assert "--no-deps" in text
    assert "pip wheel" not in text
    assert "--upgrade pip" not in text
    assert "https://" not in text
    assert "http://" not in text
    assert "wheelhouse_manifest.json" in text
    assert "RUNTIME_DEPENDENCY_LOCK_HASH_INVALID" in text
    assert "RUNTIME_FILE_HASH_INVALID" in text
    assert "RUNTIME_OBJECT_HASH_INVALID" in text


def test_catalog_workers_pin_numeric_threads_without_oversubscription() -> None:
    worker_texts = [
        (ROOT / ".github/workflows/catalog-optimized-worker.yml").read_text(
            "utf-8"
        ),
        (ROOT / ".github/workflows/catalog-component-worker.yml").read_text(
            "utf-8"
        ),
    ]
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert all(f'{variable}: "1"' in text for text in worker_texts)


def test_runtime_setup_has_no_credential_or_larger_runner_escape() -> None:
    action_text = ACTION_PATH.read_text(encoding="utf-8").lower()
    assert "persist-credentials" not in action_text
    assert "larger" not in action_text
    assert "gpu" not in action_text
    assert "self-hosted" not in action_text


def _workflow() -> dict[str, Any]:
    return dict(load_github_yaml(WORKFLOW_PATH))


def _needs(job: dict[str, Any]) -> set[str]:
    value = job.get("needs", ())
    if isinstance(value, str):
        return {value}
    return set(value)


def test_reusable_workflow_has_complete_dependency_spine() -> None:
    jobs = _workflow()["jobs"]
    assert _needs(jobs["prepare_environment"]) == set()
    assert _needs(jobs["validate"]) == {"prepare_environment"}
    assert _needs(jobs["prepare_data"]) == {
        "validate",
        "sp500_data_plan",
        "sp500_prepare_data",
    }
    assert _needs(jobs["freeze_contract"]) == {
        "validate",
        "prepare_environment",
        "prepare_data",
    }
    assert _needs(jobs["smoke"]) == {"freeze_contract"}
    assert _needs(jobs["pilot"]) == {"smoke"}
    assert _needs(jobs["plan"]) == {"pilot"}
    assert _needs(jobs["fanout_a"]) == {"plan"}
    assert _needs(jobs["fanout_b"]) == {"plan"}
    assert _needs(jobs["campaign_initialize"]) == {
        "plan",
        "freeze_contract",
    }
    assert jobs["smoke"]["if"] == (
        "always() && needs.freeze_contract.result == 'success'"
    )
    assert jobs["pilot"]["if"] == "always() && needs.smoke.result == 'success'"
    assert jobs["plan"]["if"] == "always() && needs.pilot.result == 'success'"
    assert jobs["fanout_a"]["if"] == "always() && needs.plan.result == 'success'"
    assert jobs["fanout_b"]["if"] == (
        "always() && needs.plan.result == 'success' && "
        "needs.plan.outputs.has_matrix_b == 'true'"
    )
    assert jobs["campaign_initialize"]["if"] == (
        "always() && needs.plan.result == 'success' && "
        "needs.freeze_contract.result == 'success'"
    )
    assert _needs(jobs["recovery_plan"]) == {
        "plan",
        "fanout_a",
        "fanout_b",
        "campaign_initialize",
    }
    assert _needs(jobs["retry_a"]) == {"plan", "recovery_plan"}
    assert _needs(jobs["retry_b"]) == {"plan", "recovery_plan"}
    assert jobs["retry_a"]["if"] == (
        "always() && needs.recovery_plan.result == 'success' && "
        "needs.recovery_plan.outputs.status == 'retry' && "
        "needs.recovery_plan.outputs.has_matrix_a == 'true'"
    )
    assert jobs["retry_b"]["if"] == (
        "always() && needs.recovery_plan.result == 'success' && "
        "needs.recovery_plan.outputs.status == 'retry' && "
        "needs.recovery_plan.outputs.has_matrix_b == 'true'"
    )
    previous_a = "retry_a"
    previous_b = "retry_b"
    previous_recovery = "recovery_plan"
    for wave in range(1, 6):
        recovery = f"recovery_plan_{wave}"
        assert _needs(jobs[recovery]) == {
            "plan",
            previous_recovery,
            previous_a,
            previous_b,
        }
        previous_recovery = recovery
        if wave < 5:
            next_a = f"retry_{wave + 1}_a"
            next_b = f"retry_{wave + 1}_b"
            assert _needs(jobs[next_a]) == {"plan", recovery}
            assert _needs(jobs[next_b]) == {"plan", recovery}
            assert jobs[next_a]["if"] == (
                f"always() && needs.{recovery}.result == 'success' && "
                f"needs.{recovery}.outputs.status == 'retry' && "
                f"needs.{recovery}.outputs.has_matrix_a == 'true'"
            )
            assert jobs[next_b]["if"] == (
                f"always() && needs.{recovery}.result == 'success' && "
                f"needs.{recovery}.outputs.status == 'retry' && "
                f"needs.{recovery}.outputs.has_matrix_b == 'true'"
            )
            previous_a = next_a
            previous_b = next_b
    assert _needs(jobs["merge_level_0"]) == {
        "plan",
        "fanout_a",
        "fanout_b",
        "recovery_plan",
        "retry_a",
        "retry_b",
        "recovery_plan_1",
        "retry_2_a",
        "retry_2_b",
        "recovery_plan_2",
        "retry_3_a",
        "retry_3_b",
        "recovery_plan_3",
        "retry_4_a",
        "retry_4_b",
        "recovery_plan_4",
        "retry_5_a",
        "retry_5_b",
        "recovery_plan_5",
    }
    assert _needs(jobs["merge_level_1"]) == {
        "plan",
        "merge_level_0",
    }
    assert _needs(jobs["merge_level_2"]) == {
        "plan",
        "merge_level_1",
    }
    assert _needs(jobs["merge_level_3"]) == {
        "plan",
        "merge_level_2",
    }
    assert _needs(jobs["final_merge"]) == {
        "plan",
        "freeze_contract",
        "merge_level_0",
        "merge_level_1",
        "merge_level_2",
        "merge_level_3",
    }
    assert _needs(jobs["collect_timeline"]) == {"final_merge"}
    assert _needs(jobs["seal_final_artifact"]) == {
        "final_merge",
        "collect_timeline",
    }
    assert _needs(jobs["verify"]) == {
        "final_merge",
        "seal_final_artifact",
    }
    assert _needs(jobs["publish"]) == {"verify"}
    assert _needs(jobs["conclude"]) == {"plan", "publish"}


def test_reusable_workflow_conclusion_is_bound_to_verified_publication() -> None:
    workflow = _workflow()
    outputs = workflow["on"]["workflow_call"]["outputs"]
    jobs = workflow["jobs"]
    conclude = jobs["conclude"]

    assert outputs["selected_jobs"]["value"] == (
        "${{ jobs.conclude.outputs.selected_jobs }}"
    )
    assert outputs["profile_reused"]["value"] == (
        "${{ jobs.conclude.outputs.profile_reused }}"
    )
    assert outputs["verification_passed"]["value"] == (
        "${{ jobs.conclude.outputs.verification_passed }}"
    )
    assert conclude["if"] == "always()"
    assert conclude["runs-on"] == "ubuntu-24.04"
    assert _needs(conclude) == {"plan", "publish"}
    verdict = next(
        step
        for step in conclude["steps"]
        if step["name"] == "Bind reusable result to verified publication"
    )
    assert verdict["env"]["PLAN_RESULT"] == "${{ needs.plan.result }}"
    assert verdict["env"]["PUBLISH_RESULT"] == "${{ needs.publish.result }}"
    assert "verification_passed=true" in verdict["run"]


def test_universal_workflow_supports_direct_dispatch_without_a_caller() -> None:
    workflow = _workflow()
    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    call_inputs = workflow["on"]["workflow_call"]["inputs"]

    assert dispatch_inputs == call_inputs
    assert {"spec_path", "workload", "run_label"} <= set(dispatch_inputs)
    assert dispatch_inputs["spec_path"]["required"] is True
    assert dispatch_inputs["workload"]["required"] is True
    assert dispatch_inputs["run_label"]["required"] is True


def test_reusable_workflow_respects_standard_runner_limits() -> None:
    jobs = _workflow()["jobs"]
    for job in jobs.values():
        if "uses" in job:
            assert str(job["uses"]).startswith("./.github/workflows/")
        else:
            assert job["runs-on"] == "ubuntu-24.04"
    matrix_limits = {
        "fanout_a": 256,
        "fanout_b": 104,
        "retry_a": 256,
        "retry_b": 104,
    }
    for wave in range(2, 6):
        matrix_limits[f"retry_{wave}_a"] = 256
        matrix_limits[f"retry_{wave}_b"] = 104
    for name, limit in matrix_limits.items():
        strategy = jobs[name]["strategy"]
        assert strategy["fail-fast"] is False
        assert strategy["max-parallel"] == limit
        assert limit <= 256
    assert (
        jobs["fanout_a"]["strategy"]["max-parallel"]
        + jobs["fanout_b"]["strategy"]["max-parallel"]
        == 360
    )
    for wave in range(1, 6):
        a = "retry_a" if wave == 1 else f"retry_{wave}_a"
        b = "retry_b" if wave == 1 else f"retry_{wave}_b"
        assert (
            jobs[a]["strategy"]["max-parallel"]
            + jobs[b]["strategy"]["max-parallel"]
            == 360
        )


def test_recovery_is_durable_and_iterates_to_maximum_retry_budget() -> None:
    jobs = _workflow()["jobs"]
    recovery_jobs = (
        "recovery_plan",
        "recovery_plan_1",
        "recovery_plan_2",
        "recovery_plan_3",
        "recovery_plan_4",
        "recovery_plan_5",
    )
    for wave, name in enumerate(recovery_jobs):
        job = jobs[name]
        assert job["runs-on"] == "ubuntu-24.04"
        recovery = next(
                step
                for step in job["steps"]
                if step.get("uses")
                == "./.github/actions/aurora-shard-recovery-plan"
            )
        assert recovery["with"]["current-wave"] == wave
        assert recovery["with"]["max-waves"] == 6
    assert "campaign-update" in WORKFLOW_PATH.read_text(encoding="utf-8")
    recovery_text = RECOVERY_PLAN_ACTION_PATH.read_text(encoding="utf-8")
    assert "aurora github recovery-loop" in recovery_text
    assert "campaign_state_latest.json" in recovery_text
    retry_text = RETRY_SHARD_ACTION_PATH.read_text(encoding="utf-8")
    assert "aurora github run-shard" in retry_text
    assert "matrix" not in retry_text
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "_aurora-recovery-plan-v3.yml" not in workflow_text
    assert "_aurora-retry-shard-v3.yml" not in workflow_text
    for wave in range(2, 6):
        for suffix in ("a", "b"):
            job = jobs[f"retry_{wave}_{suffix}"]
            assert job["runs-on"] == "ubuntu-24.04"
            assert any(
                step.get("uses")
                == "./.github/actions/aurora-retry-shard"
                for step in job["steps"]
            )


def test_universal_merge_only_workflow_reuses_verified_artifacts_only() -> None:
    workflow = dict(load_github_yaml(MERGE_ONLY_WORKFLOW_PATH))
    assert "workflow_call" in workflow["on"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert {
        "source_run_id",
        "source_state_artifact_name",
        "run_label",
        "workload",
        "retention_days",
    } <= set(inputs)
    text = MERGE_ONLY_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "aurora github merge-only" in text
    assert "aurora github final-merge" in text
    assert "aurora github seal-final-artifact" in text
    assert "aurora github verify" in text
    assert "campaign_state_latest.json" in text
    assert "verified_source_artifacts" in text
    assert "gh run download" in text
    assert "MERGE_ONLY_SOURCE_TELEMETRY_MISSING" in text
    assert "merge_only_verification.json" in text
    assert "scientific_outputs_equal" in text
    assert "aurora github run-shard" not in text
    assert '["scientific_output"]' in text
    assert "reference_results.parquet" not in text
    steps = workflow["jobs"]["merge_only"]["steps"]
    names = [step["name"] for step in steps]
    assert names.index(
        "Rehydrate telemetry and prove scientific equivalence"
    ) < names.index("Seal complete rebuilt artifact")
    assert names.index(
        "Seal complete rebuilt artifact"
    ) < names.index("Independently verify merge-only artifact")
    publish = next(
        step
        for step in steps
        if step["name"] == "Publish merge-only final artifact"
    )
    assert "always()" in publish["if"]
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"


def test_universal_replan_workflow_changes_only_operational_plan() -> None:
    workflow = dict(load_github_yaml(REPLAN_WORKFLOW_PATH))
    assert "workflow_call" in workflow["on"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert {
        "source_run_id",
        "source_state_artifact_name",
        "run_label",
        "requested_jobs",
        "operational_overrides",
        "retention_days",
    } <= set(inputs)
    text = REPLAN_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "aurora github replan-pending" in text
    assert "campaign_state_latest.json" in text
    assert "work_unit_manifest.json" in text
    assert "unit_attempts.parquet" in text
    assert "operational_overrides" in text
    assert "aurora github run-shard" not in text
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"


def test_registered_reference_workflow_routes_recovery_operations() -> None:
    workflow = dict(load_github_yaml(REFERENCE_WORKFLOW_PATH))
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["operation"]["options"] == [
        "full",
        "transient_recovery",
        "merge_only",
        "replan",
        "replan_fixture",
    ]
    jobs = workflow["jobs"]
    assert "full_forced_job_count" in inputs
    assert jobs["reference"]["uses"] == (
        "./.github/workflows/_aurora-future-run-v3.yml"
    )
    full_jobs = jobs["reference"]["with"]["forced_job_count"]
    assert "fromJSON" in full_jobs
    assert "inputs.full_forced_job_count" in full_jobs
    assert jobs["merge_only"]["uses"] == (
        "./.github/workflows/github-performance-merge-only.yml"
    )
    assert jobs["replan"]["uses"] == (
        "./.github/workflows/github-performance-replan.yml"
    )
    requested_jobs = jobs["replan"]["with"]["requested_jobs"]
    assert "fromJSON" in requested_jobs
    assert "format('{0}', inputs.requested_jobs)" in requested_jobs
    assert jobs["replan_fixture_source"]["uses"] == (
        "./.github/workflows/_aurora-future-run-v3.yml"
    )
    fixture = jobs["build_replan_fixture"]
    assert fixture["runs-on"] == "ubuntu-24.04"
    fixture_text = REFERENCE_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "OUT_OF_MEMORY" not in fixture_text
    assert "build_github_performance_replan_fixture.py" in fixture_text


def test_registered_validation_workflow_routes_three_real_families() -> None:
    workflow = dict(load_github_yaml(VALIDATION_WORKFLOW_PATH))
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["workload_family"]["options"] == [
        "candidate_sweep",
        "event_study",
        "robustness",
    ]
    assert inputs["execution_mode"]["options"] == [
        "baseline",
        "optimized",
    ]
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "candidate_sweep",
        "event_study",
        "robustness",
    }
    expected = {
        "candidate_sweep": (
            "config/github_performance_candidate_sweep.yaml",
            "aurora.infra.github_performance.workloads."
            "candidate_sweep:WORKLOAD",
        ),
        "event_study": (
            "config/github_performance_event_study.yaml",
            "aurora.infra.github_performance.workloads."
            "event_study:WORKLOAD",
        ),
        "robustness": (
            "config/github_performance_robustness.yaml",
            "aurora.infra.github_performance.workloads."
            "robustness:WORKLOAD",
        ),
    }
    for name, job in jobs.items():
        assert job["uses"] == "./.github/workflows/_aurora-future-run-v3.yml"
        assert job["with"]["spec_path"] == expected[name][0]
        assert job["with"]["workload"] == expected[name][1]
        assert job["with"]["execution_mode"] == "${{ inputs.execution_mode }}"
        assert "fromJSON" in job["with"]["forced_job_count"]
    text = VALIDATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "self-hosted" not in text
    assert "ubuntu-latest" not in text
    assert "C:\\" not in text


def test_replan_restores_verified_source_runtime_across_orchestration_fixes() -> None:
    text = REPLAN_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "SOURCE_SCIENTIFIC_RUNTIME_MISMATCH" in text
    assert "source_runtime_verified" in text
    assert 'wheelhouse["code_sha"]' in text
    assert 'contract["identity"]["code_sha"]' in text
    assert "orchestration_code_sha" in text
    assert "SOURCE_CODE_SHA_MISMATCH" not in text


def test_reusable_workflow_preserves_salvage_and_bounded_merge() -> None:
    jobs = _workflow()["jobs"]
    for name in ("recovery_plan", "merge_level_0", "final_merge", "verify"):
        assert "always()" in jobs[name]["if"]
    for name in ("fanout_a", "fanout_b", "retry_a", "retry_b"):
        upload_steps = [
            step
            for step in jobs[name]["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        assert upload_steps
        assert all("always()" in step["if"] for step in upload_steps)
    final_download = next(
        step
        for step in jobs["final_merge"]["steps"]
        if step["name"] == "Download merge root only"
    )
    assert final_download["with"]["name"] == (
        "${{ needs.plan.outputs.merge_root_artifact }}"
    )
    assert "pattern" not in final_download["with"]


def test_checkpoint_salvage_detects_files_outside_github_workspace() -> None:
    workflows = (
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        RETRY_SHARD_ACTION_PATH.read_text(encoding="utf-8"),
    )
    for text in workflows:
        assert "find \"$RUNNER_TEMP/attempt\"" in text
        assert "-name checkpoint_manifest.json" in text
        assert "steps.checkpoint.outputs.exists == 'true'" in text
        assert "hashFiles(format('{0}/attempt" not in text


def test_reusable_workflow_executes_every_bounded_merge_plan_level() -> None:
    jobs = _workflow()["jobs"]
    level_names = [f"merge_level_{level_index}" for level_index in range(4)]
    assert all(level_name in jobs for level_name in level_names)
    for level_name in level_names:
        job = jobs[level_name]
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["strategy"]["fail-fast"] is False
        assert job["strategy"]["max-parallel"] <= 256
        assert any(
            step.get("uses")
            == "./.github/actions/aurora-merge-level"
            for step in job["steps"]
        )
        merge_step = next(
            step
            for step in job["steps"]
            if step.get("uses") == "./.github/actions/aurora-merge-level"
        )
        assert merge_step["with"]["prepared-artifact-name"] == (
            "${{ env.AURORA_PREPARED_ARTIFACT_NAME }}"
        )
        assert merge_step["with"]["prepared-artifact-run-id"] == (
            "${{ env.AURORA_PREPARED_ARTIFACT_RUN_ID }}"
        )
    for level_index in range(1, 4):
        assert _needs(jobs[f"merge_level_{level_index}"]) == {
            "plan",
            f"merge_level_{level_index - 1}",
        }
    assert _needs(jobs["final_merge"]) == {
        "plan",
        "freeze_contract",
        *level_names,
    }
    final_steps = jobs["final_merge"]["steps"]
    prepared_download = next(
        step
        for step in final_steps
        if step.get("name") == "Download prepared inputs"
    )
    assert prepared_download["with"] == {
        "name": "${{ env.AURORA_PREPARED_ARTIFACT_NAME }}",
        "path": "${{ runner.temp }}/prepared",
        "github-token": "${{ github.token }}",
        "run-id": "${{ env.AURORA_PREPARED_ARTIFACT_RUN_ID }}",
    }
    final_build = next(
        step
        for step in final_steps
        if step.get("name") == "Build final artifact and reconciliation"
    )
    assert final_build["env"]["AURORA_PREPARED_ROOT"] == (
        "${{ runner.temp }}/prepared"
    )
    merge_text = MERGE_LEVEL_ACTION_PATH.read_text(encoding="utf-8")
    assert "aurora github merge-plan-group" in merge_text
    assert "inputs.output-artifact" in merge_text
    assert "inputs.prepared-artifact-name" in merge_text
    assert "inputs.prepared-artifact-run-id" in merge_text
    assert "AURORA_PREPARED_ROOT: ${{ runner.temp }}/prepared" in merge_text
    final_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "_aurora-merge-level-v3.yml" not in final_text
    assert "needs.plan.outputs.merge_root_artifact" in final_text
    assert "Download merge root only" in final_text


def test_reusable_workflow_uses_compact_unique_attempt_artifacts() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "plan_outputs.json" in text
    assert "matrix.attempt_id" in text
    assert (
        "shard-${{ matrix.merge_group }}-${{ matrix.shard_id }}"
        "-${{ matrix.attempt_id }}"
    ) in text
    assert "compression-level: 0" in text
    assert "if-no-files-found: error" in text
    assert text.count('--assignment-root "$RUNNER_TEMP/plan"') == 4
    assert text.count('--prepared-root "$RUNNER_TEMP/prepared"') == 5


def test_reusable_workflow_inputs_and_permissions_are_minimal() -> None:
    workflow = _workflow()
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert set(inputs) == {
        "execution_mode",
        "forced_job_count",
        "prepared_artifact_name",
        "prepared_artifact_run_id",
        "wheelhouse_artifact_name",
        "performance_profile_run_id",
        "performance_profile_artifact_name",
        "fault_injection_shard_id",
        "fault_injection_after_units",
        "spec_path",
        "workload",
        "run_label",
        "retention_days",
        "autonomous_batch_id",
        "autonomous_candidate_count",
        "autonomous_previous_trial_count",
        "autonomous_prior_ledger_artifact_name",
        "autonomous_prior_ledger_run_id",
    }
    assert inputs["fault_injection_shard_id"]["default"] == ""
    assert inputs["fault_injection_after_units"]["default"] == 0
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert "push" not in workflow["on"]
    assert "pull_request" not in workflow["on"]


def test_reusable_workflow_builds_aurora_and_wheelhouse_exactly_once() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    prepare_text = str(jobs["prepare_environment"])
    assert "build_github_performance_wheelhouse.py" in prepare_text
    assert text.count("build_github_performance_wheelhouse.py") == 1
    assert text.count("pip wheel") <= 1
    assert "requirements/github-performance.lock" in prepare_text
    upload = next(
        step
        for step in jobs["prepare_environment"]["steps"]
        if step["name"] == "Upload immutable wheelhouse"
    )
    assert "wheelhouse" in str(upload["with"]["path"])
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["compression-level"] == 0


def test_reusable_workflow_can_reuse_exact_wheelhouse_artifact() -> None:
    workflow = _workflow()
    prepare = workflow["jobs"]["prepare_environment"]
    shared_download = next(
        step
        for step in prepare["steps"]
        if step["name"] == "Download shared immutable wheelhouse"
    )
    build = next(
        step
        for step in prepare["steps"]
        if step["name"] == "Build immutable wheelhouse once"
    )
    upload = next(
        step
        for step in prepare["steps"]
        if step["name"] == "Upload immutable wheelhouse"
    )
    assert shared_download["if"] == "inputs.wheelhouse_artifact_name != ''"
    assert build["if"] == "inputs.wheelhouse_artifact_name == ''"
    assert upload["if"] == "inputs.wheelhouse_artifact_name == ''"
    assert (
        shared_download["with"]["name"]
        == "${{ env.AURORA_WHEELHOUSE_ARTIFACT_NAME }}"
    )


def test_every_runtime_consumer_downloads_the_same_wheelhouse_first() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    runtime_jobs = {
        name
        for name, job in jobs.items()
        if any(
            step.get("uses")
            == "./.github/actions/aurora-runtime-setup"
            for step in job.get("steps", ())
        )
    }
    assert runtime_jobs
    for name in runtime_jobs:
        steps = jobs[name]["steps"]
        setup_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses")
            == "./.github/actions/aurora-runtime-setup"
        )
        download_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Download immutable wheelhouse"
        )
        assert download_index < setup_index, name
        setup = steps[setup_index]
        assert (
            setup["with"]["wheelhouse-path"]
            == "${{ runner.temp }}/wheelhouse"
        )
        assert (
            setup["with"]["dependency-lock-path"]
            == "requirements/github-performance.lock"
        )


def test_prepare_data_retries_the_critical_wheelhouse_download() -> None:
    workflow = _workflow()
    download = next(
        step
        for step in workflow["jobs"]["prepare_data"]["steps"]
        if step.get("name") == "Download immutable wheelhouse"
    )
    assert download["uses"] == "./.github/actions/download-artifact-retry"
    assert download["with"] == {
        "artifact-name": "${{ env.AURORA_WHEELHOUSE_ARTIFACT_NAME }}",
        "output-dir": "${{ runner.temp }}/wheelhouse",
        "attempts": "6",
    }


def test_reusable_workflow_can_reuse_exact_prepared_artifact() -> None:
    workflow = _workflow()
    prepared = workflow["jobs"]["prepare_data"]
    shared_download = next(
        step
        for step in prepared["steps"]
        if step["name"] == "Download shared immutable inputs"
    )
    prepare = next(
        step
        for step in prepared["steps"]
        if step["name"] == "Prepare immutable data once"
    )
    upload = next(
        step
        for step in prepared["steps"]
        if step["name"] == "Upload prepared inputs"
    )
    assert shared_download["if"] == (
        "inputs.prepared_artifact_name != '' || "
        "needs.sp500_data_plan.outputs.enabled == 'true'"
    )
    assert prepare["if"] == (
        "inputs.prepared_artifact_name == '' && "
        "needs.sp500_data_plan.outputs.enabled != 'true'"
    )
    assert upload["if"] == prepare["if"]
    assert (
        shared_download["with"]["name"]
        == "${{ inputs.prepared_artifact_name || env.AURORA_PREPARED_ARTIFACT_NAME }}"
    )
    assert "inputs.prepared_artifact_name" in str(
        workflow["env"]["AURORA_PREPARED_ARTIFACT_NAME"]
    )
    assert "github.run_attempt" in str(
        workflow["env"]["AURORA_PREPARED_ARTIFACT_NAME"]
    )
    assert "github.run_attempt" in str(
        workflow["env"]["AURORA_WHEELHOUSE_ARTIFACT_NAME"]
    )


def test_reusable_workflow_reuses_only_exact_prior_performance_profile() -> None:
    workflow = _workflow()
    pilot = workflow["jobs"]["pilot"]
    download = next(
        step
        for step in pilot["steps"]
        if step["name"] == "Download prior immutable performance profile"
    )
    resolve = next(
        step
        for step in pilot["steps"]
        if step["name"] == "Resolve exact profile or measure fresh pilot"
    )
    plan = workflow["jobs"]["plan"]
    build = next(
        step
        for step in plan["steps"]
        if step["name"] == "Build adaptive balanced plan"
    )

    assert "performance_profile_run_id" in str(download["if"])
    assert "performance_profile_artifact_name" in str(download["if"])
    assert download["with"]["run-id"] == (
        "${{ inputs.performance_profile_run_id }}"
    )
    assert download["with"]["github-token"] == "${{ github.token }}"
    assert "aurora github resolve-pilot" in resolve["run"]
    assert "--performance-profile" in resolve["run"]
    assert "planning_pilot_resolution.json" in resolve["run"]
    assert "--pilot-resolution" in build["run"]
    assert plan["outputs"]["profile_reused"] == (
        "${{ steps.profile.outputs.profile_reused }}"
    )


def test_transient_fault_is_limited_to_initial_fanout() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]

    for name in ("fanout_a", "fanout_b"):
        execute = next(
            step
            for step in jobs[name]["steps"]
            if step["name"] == "Execute shard"
        )
        assert execute["env"]["AURORA_FAULT_INJECTION_SHARD_ID"] == (
            "${{ inputs.fault_injection_shard_id }}"
        )
        assert execute["env"]["AURORA_FAULT_INJECTION_AFTER_UNITS"] == (
            "${{ inputs.fault_injection_after_units }}"
        )
        assert "continue-on-error" not in execute
        assert "--defer-technical-failure-to-recovery" in execute["run"]
        salvage = next(
            step
            for step in jobs[name]["steps"]
            if step["name"] == "Salvage attempt evidence"
        )
        assert salvage["continue-on-error"] is True

    retry_path = RETRY_SHARD_ACTION_PATH
    retry_text = retry_path.read_text(encoding="utf-8")
    assert "AURORA_FAULT_INJECTION_SHARD_ID" not in retry_text
    assert "AURORA_FAULT_INJECTION_AFTER_UNITS" not in retry_text
    retry = load_github_yaml(retry_path)["runs"]
    execute_retry = next(
        step
        for step in retry["steps"]
        if step["name"] == "Execute exact retry"
    )
    salvage_retry = next(
        step
        for step in retry["steps"]
        if step["name"] == "Salvage retry evidence"
    )
    assert "continue-on-error" not in execute_retry
    assert "--defer-technical-failure-to-recovery" in execute_retry["run"]
    assert salvage_retry["continue-on-error"] is True


def test_inline_retries_record_failure_without_poisoning_reusable_workflow() -> None:
    jobs = _workflow()["jobs"]
    for name in ("retry_a", "retry_b"):
        execute = next(
            step
            for step in jobs[name]["steps"]
            if step["name"] == "Execute retry"
        )
        assert "continue-on-error" not in execute
        assert "--defer-technical-failure-to-recovery" in execute["run"]


def test_timeline_collection_is_read_only_and_cannot_block_science() -> None:
    jobs = _workflow()["jobs"]
    collector = jobs["collect_timeline"]
    assert collector["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    step = next(
        item
        for item in collector["steps"]
        if item["name"] == "Collect read-only GitHub timing"
    )
    assert step["continue-on-error"] is True
    seal = jobs["seal_final_artifact"]
    assert any(
        "aurora github seal-final-artifact"
        in str(item.get("run", ""))
        for item in seal["steps"]
    )
    assert _needs(seal) == {"final_merge", "collect_timeline"}
    verify = jobs["verify"]
    assert _needs(verify) == {"final_merge", "seal_final_artifact"}
    publisher = jobs["publish"]
    assert "always()" in publisher["if"]
    assert not any(
        item["name"]
        == "Record incomplete timing without touching scientific files"
        for item in publisher["steps"]
    )
    assert not any(
        "timeline" in str(item.get("name", "")).lower()
        for item in publisher["steps"]
    )


def test_policy_workflow_is_lightweight_static_pr_enforcement() -> None:
    workflow = load_github_yaml(POLICY_WORKFLOW_PATH)
    assert "pull_request" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"workflow_policy"}
    job = jobs["workflow_policy"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert "strategy" not in job
    text = POLICY_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "validate_github_workflow_policy.py" in text
    assert "workflow_dispatch" in workflow["on"]


def test_catalog_capacity_calibration_is_fixed_synthetic_and_read_only() -> None:
    workflow = load_github_yaml(CAPACITY_CALIBRATION_WORKFLOW_PATH)
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["on"]["workflow_dispatch"] == {}
    assert workflow["on"]["schedule"] == [{"cron": "17 3 */3 * *"}]
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    jobs = workflow["jobs"]
    assert set(jobs) == {"synthetic_probe", "seal_receipt"}
    probe = jobs["synthetic_probe"]
    assert probe["runs-on"] == "ubuntu-24.04"
    assert probe["strategy"]["max-parallel"] == 11
    assert len(probe["strategy"]["matrix"]["probe_id"]) == 11
    assert jobs["seal_receipt"]["runs-on"] == "ubuntu-24.04"
    assert _needs(jobs["seal_receipt"]) == {"synthetic_probe"}

    text = CAPACITY_CALIBRATION_WORKFLOW_PATH.read_text("utf-8")
    lowered = text.lower()
    for forbidden in (
        "catalog-optimized-run",
        "catalog-component-worker",
        "catalog-optimized-worker",
        "catalog-recovery-wave",
        "run_sp500_optimized_recipe_worker",
        "build_sp500_component_store",
        "reduce_sp500_optimized_catalog",
        "git push",
        "github_capacity_profile.json",
    ):
        assert forbidden not in lowered
    assert "catalog_capacity_calibration_v1.json" in text
    assert '"p50"' in text
    assert '"p95"' in text
    assert '"p99"' in text
    assert "Download exact tiny probe evidence for timing" in text
    assert '"artifact_download_seconds": distribution(download_seconds)' in text
    assert "receipt_sha256" in text
    assert "retention-days: 90" in text


def test_catalog_reduction_uses_bound_merge_action_and_root_only() -> None:
    workflow = load_github_yaml(CATALOG_OPTIMIZED_WORKFLOW_PATH)
    jobs = workflow["jobs"]
    grouped = jobs["reduce_groups"]
    merge_step = next(
        step
        for step in grouped["steps"]
        if step.get("uses") == "./.github/actions/aurora-merge-level"
    )
    assert merge_step["with"]["mode"] == "catalog"
    assert merge_step["with"]["expected-level"] == "0"
    assert merge_step["with"]["expected-authority-id"] == (
        "${{ inputs.authority_id }}"
    )
    assert merge_step["with"]["expected-campaign-id"] == (
        "${{ inputs.campaign_id }}"
    )
    assert merge_step["with"]["expected-science-sha256"] == (
        "${{ inputs.science_sha256 }}"
    )
    assert merge_step["with"]["expected-execution-plan-sha256"] == (
        "${{ inputs.execution_plan_sha256 }}"
    )
    final_download = next(
        step
        for step in jobs["reduce"]["steps"]
        if step["name"] == "Download only bounded reduction groups"
    )
    assert "reduction_artifact_pattern" in final_download["with"]["pattern"]
    assert "checkpoint" not in final_download["with"]["pattern"]

    workflow_text = CATALOG_OPTIMIZED_WORKFLOW_PATH.read_text("utf-8")
    assert 'selected_mode == "central"' in workflow_text
    assert "expected_maximum_groups = 1" in workflow_text
    assert "expected_maximum_fan_in = 360" in workflow_text

    action_text = MERGE_LEVEL_ACTION_PATH.read_text("utf-8")
    for marker in (
        "Validate immutable reducer mode",
        "CATALOG_MERGE_MODE_INVALID",
        "CATALOG_MERGE_DIRECT_CHILD_SET_INVALID",
        "CATALOG_MERGE_PLAN_BINDING_INVALID",
        "CATALOG_MERGE_RESOURCE_MARGIN_UNPROVEN",
        "node_descriptor_sha256",
        "float(projection[field]) > 0.70",
        "float(projection[field]) < 0",
    ):
        assert marker in action_text
