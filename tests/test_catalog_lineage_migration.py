"""Maintenance prepares new bytes; the existing protected transaction applies them."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from aurora.infra.sp500_megarun.catalog_request_contract import CatalogLaunchTicketV1
from aurora.infra.sp500_megarun.catalog_requester import CatalogRequesterCampaignStatusV1
from aurora.infra.sp500_megarun.catalog_requester_broker import _ticket_journal
from tests.test_catalog_fast_authority import _lineage_boundary


NOW = datetime(2026, 9, 8, tzinfo=timezone.utc)


@pytest.mark.parametrize("campaign,generation,predecessor,installed_definition", [
    ("catalog-fast-canary-v1", 4,
     "b253c84105c6c221d1b8e2068cd5951c03cd564eaf9699d3fb5a841fa48eb37e",
     "fd0d9904b66458be9bd1ecfea5a8224f85669def9beaf379c7450d475cb26ace"),
    ("sp500-optimized-catalog-v1", 7,
     "1f73eadbb2404095072c61fb67f36f813cff8b119bc17bbb3d5df8852ad333f7",
     "684f349ba2e97f1f6fe03a78c7649f44baf22111691c11cc92909102cd7e0334"),
])
def test_release_transition_resolves_installed_unused_ticket_to_packaged_definition(
    campaign, generation, predecessor, installed_definition,
):
    from aurora.infra.sp500_megarun.catalog_campaign_definition_contract import parse_catalog_campaign_definition_bytes
    from aurora.infra.sp500_megarun.catalog_lineage_transition import load_lineage_transition

    root = Path(__file__).resolve().parents[1]
    prompt_hash = hashlib.sha256((root / "docs/runbooks/CATALOG_RUN_MASTER_PROMPT.md").read_bytes()).hexdigest()
    ticket = CatalogLaunchTicketV1(
        schema_version="1", request_id="018f47a2-6e91-7c34-8000-000000000002",
        campaign_key=campaign, launch_generation=generation,
        previous_terminal_request_sha256=predecessor,
        campaign_definition_sha256=installed_definition, prompt_sha256=prompt_hash,
    )
    approval = load_lineage_transition(root, ticket)
    assert approval is not None
    definition = parse_catalog_campaign_definition_bytes(
        (root / f"config/catalog_campaign_definitions/{campaign}.manifest.json").read_bytes())
    assert approval.previous_request_sha256 == predecessor
    assert approval.target_definition_sha256 == definition.campaign_definition_sha256
    assert approval.target_prompt_sha256 == prompt_hash
    if campaign == "sp500-optimized-catalog-v1":
        assert (installed_definition, prompt_hash) in {
            (context.campaign_definition_sha256, context.prompt_sha256)
            for context in approval.source_ticket_contexts
        }


def _available_models():
    state, request, approval = _lineage_boundary()
    previous = state.campaigns[0].request
    ticket = CatalogLaunchTicketV1(
        schema_version="1", request_id=request.request_id, campaign_key=previous.campaign_key,
        launch_generation=7, previous_terminal_request_sha256=previous.request_sha256,
        campaign_definition_sha256=previous.campaign_definition_sha256,
        prompt_sha256=previous.prompt_sha256,
    )
    journal = _ticket_journal(ticket=ticket, state="available", submission_key_sha256=None,
        request_sha256=None, issue_number=None, created_at=NOW, updated_at=NOW)
    status = CatalogRequesterCampaignStatusV1.create(campaign_key=ticket.campaign_key,
        state="ticket_available", launch_generation=7, launch_ticket_sha256=ticket.launch_ticket_sha256,
        updated_at=NOW)
    return previous, approval, ticket, journal, status


def test_available_migration_preserves_identity_predecessor_and_old_models():
    from aurora.infra.sp500_megarun.catalog_lineage_migration import prepare_available_lineage_models

    previous, approval, ticket, journal, status = _available_models()
    originals = [model.model_dump_json() for model in (previous, ticket, journal, status)]
    updated = prepare_available_lineage_models(previous_request=previous, transition=approval,
        ticket=ticket, journal=journal, status=status, observed_at=NOW + timedelta(seconds=1))
    next_ticket, next_journal, next_status = updated
    assert next_ticket.request_id == ticket.request_id
    assert next_ticket.launch_generation == 7
    assert next_ticket.previous_terminal_request_sha256 == previous.request_sha256
    assert next_ticket.campaign_definition_sha256 == "a" * 64
    assert next_ticket.prompt_sha256 == "b" * 64
    assert next_journal.ticket == next_ticket
    assert next_journal.created_at == NOW
    assert next_journal.state == "available"
    assert next_status.launch_ticket_sha256 == next_ticket.launch_ticket_sha256
    assert next_status.state == "ticket_available"
    assert [model.model_dump_json() for model in (previous, ticket, journal, status)] == originals
    assert prepare_available_lineage_models(previous_request=previous, transition=approval,
        ticket=next_ticket, journal=next_journal, status=next_status,
        observed_at=NOW + timedelta(seconds=2)) == updated


@pytest.mark.parametrize("permission", ["exact", "missing", "wrong_definition", "wrong_prompt",
    "wrong_generation", "wrong_predecessor", "claimed"])
def test_unused_intermediate_ticket_requires_exact_protected_source_permission(permission):
    from aurora.infra.sp500_megarun.catalog_lineage_migration import prepare_available_lineage_models

    previous, approval, ticket, _, _ = _available_models()
    ticket = CatalogLaunchTicketV1.model_validate({**ticket.model_dump(),
        "campaign_definition_sha256": "c" * 64, "prompt_sha256": "d" * 64})
    if permission == "wrong_generation":
        ticket = CatalogLaunchTicketV1.model_validate({**ticket.model_dump(), "launch_generation": 8})
    elif permission == "wrong_predecessor":
        ticket = CatalogLaunchTicketV1.model_validate({**ticket.model_dump(), "previous_terminal_request_sha256": "e" * 64})
    journal = _ticket_journal(ticket=ticket, state="claiming" if permission == "claimed" else "available",
        submission_key_sha256="e" * 64 if permission == "claimed" else None,
        request_sha256=None, issue_number=None, created_at=NOW, updated_at=NOW)
    status = CatalogRequesterCampaignStatusV1.create(campaign_key=ticket.campaign_key,
        state="ticket_available", launch_generation=ticket.launch_generation, launch_ticket_sha256=ticket.launch_ticket_sha256,
        updated_at=NOW)
    if permission != "missing":
        approval = type(approval).model_validate({**approval.model_dump(), "source_ticket_contexts": [{
            "campaign_definition_sha256": ("e" if permission == "wrong_definition" else "c") * 64,
            "prompt_sha256": ("e" if permission == "wrong_prompt" else "d") * 64,
        }]})
    before = [model.model_dump_json() for model in (previous, ticket, journal, status)]
    def prepare():
        return prepare_available_lineage_models(previous_request=previous, transition=approval,
            ticket=ticket, journal=journal, status=status, observed_at=NOW + timedelta(seconds=1))
    if permission == "exact":
        migrated, next_journal, next_status = prepare()
        assert migrated.request_id == ticket.request_id
        assert migrated.launch_generation == 7
        assert migrated.previous_terminal_request_sha256 == previous.request_sha256
        assert (migrated.campaign_definition_sha256, migrated.prompt_sha256) == ("a" * 64, "b" * 64)
        assert next_journal.state == "available"
        assert next_journal.ticket == migrated
        assert next_status.launch_ticket_sha256 == migrated.launch_ticket_sha256
        assert not approval.authorizes(previous, ticket)
        assert approval.authorizes(previous, migrated)
    else:
        with pytest.raises(ValueError, match="REQUESTER_LINEAGE_MIGRATION_INVALID"):
            prepare()
    assert [model.model_dump_json() for model in (previous, ticket, journal, status)] == before


@pytest.mark.parametrize("defect", ["duplicate", "oversized", "extra_field", "invalid_digest"])
def test_source_ticket_permission_rejects_ambiguous_or_malformed_contexts(defect):
    _, approval, _, _, _ = _available_models()
    context = {"campaign_definition_sha256": "c" * 64, "prompt_sha256": "d" * 64}
    contexts = [context]
    if defect == "duplicate":
        contexts *= 2
    elif defect == "oversized":
        contexts = [{**context, "campaign_definition_sha256": f"{index:064x}"} for index in range(17)]
    elif defect == "extra_field":
        contexts = [{**context, "allow_any_source": True}]
    else:
        contexts = [{**context, "prompt_sha256": "invalid"}]
    with pytest.raises(ValueError):
        type(approval).model_validate({**approval.model_dump(), "source_ticket_contexts": contexts})


@pytest.mark.parametrize("defect", ["claiming", "status_ticket", "ticket_context", "predecessor", "time"])
def test_available_migration_rejects_inconsistent_or_claimed_state(defect):
    from aurora.infra.sp500_megarun.catalog_lineage_migration import prepare_available_lineage_models

    previous, approval, ticket, journal, status = _available_models()
    if defect == "claiming":
        journal = _ticket_journal(ticket=ticket, state="claiming", submission_key_sha256="e" * 64,
            request_sha256=None, issue_number=None, created_at=NOW, updated_at=NOW)
    elif defect == "status_ticket":
        status = CatalogRequesterCampaignStatusV1.create(campaign_key=ticket.campaign_key,
            state="ticket_available", launch_generation=7, launch_ticket_sha256="f" * 64, updated_at=NOW)
    elif defect in {"ticket_context", "predecessor"}:
        field = "campaign_definition_sha256" if defect == "ticket_context" else "previous_terminal_request_sha256"
        ticket = CatalogLaunchTicketV1.model_validate({**ticket.model_dump(), field: "f" * 64})
    with pytest.raises(ValueError, match="REQUESTER_LINEAGE_MIGRATION_INVALID"):
        prepare_available_lineage_models(previous_request=previous, transition=approval,
            ticket=ticket, journal=journal, status=status,
            observed_at=NOW - timedelta(seconds=1) if defect == "time" else NOW)


def _disk_state(tmp_path, *, use_request_hash_as_submission_key=False):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from aurora.infra.sp500_megarun.catalog_request_contract import canonical_model_bytes, canonical_sha256
    from aurora.infra.sp500_megarun.catalog_requester import CatalogRequesterConfigV1
    from aurora.infra.sp500_megarun.catalog_requester_broker import CatalogBrokerProcessingRecordV1
    from aurora.infra.sp500_megarun.catalog_run_request import parse_catalog_run_request
    from aurora.infra.sp500_megarun.catalog_lineage_transition import CatalogLineageTransitionV1
    from tests.test_inspect_catalog_fast_request import _signed_request

    config = CatalogRequesterConfigV1.model_validate_json(
        (Path(__file__).resolve().parents[1] / "config/catalog_requester_v1.json").read_text("utf-8"))
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    title, body = _signed_request(private, launch_generation=6, previous_terminal_request_sha256="e" * 64)
    previous = parse_catalog_run_request(title, body, public)
    submission_key = previous.intent.submission_key_sha256
    assert submission_key != previous.request_sha256
    if use_request_hash_as_submission_key:
        submission_key = previous.request_sha256
    record = CatalogBrokerProcessingRecordV1.model_construct(
        schema_version="1", stage="signed_before_post", title=title, body=body,
        intent_sha256=previous.intent_sha256, request_sha256=previous.request_sha256,
        request=previous, signed_at=NOW, processing_record_sha256="0" * 64,
    )
    record = CatalogBrokerProcessingRecordV1.model_validate({**record.model_dump(mode="json"),
        "processing_record_sha256": canonical_sha256(record)})
    old_ticket = CatalogLaunchTicketV1(schema_version="1", request_id=previous.request_id,
        campaign_key=previous.campaign_key, launch_generation=6,
        campaign_definition_sha256=previous.campaign_definition_sha256, prompt_sha256=previous.prompt_sha256,
        previous_terminal_request_sha256=previous.previous_terminal_request_sha256)
    ticket = CatalogLaunchTicketV1.model_validate({**old_ticket.model_dump(), "launch_generation": 7,
        "request_id": "018f47a2-6e91-7c34-8000-000000000002",
        "previous_terminal_request_sha256": previous.request_sha256})
    journal = _ticket_journal(ticket=ticket, state="available", submission_key_sha256=None,
        request_sha256=None, issue_number=None, created_at=NOW, updated_at=NOW)
    terminal = _ticket_journal(ticket=old_ticket, state="terminal",
        submission_key_sha256=submission_key, request_sha256=previous.request_sha256,
        issue_number=276, created_at=NOW, updated_at=NOW)
    status = CatalogRequesterCampaignStatusV1.create(campaign_key=ticket.campaign_key,
        state="ticket_available", launch_generation=7, launch_ticket_sha256=ticket.launch_ticket_sha256, updated_at=NOW)
    for relative in (config.broker.campaign_status, config.broker.launch_tickets, config.broker.processing, config.broker.inbox):
        (tmp_path / relative).mkdir(parents=True)
    files = {
        f"{config.broker.campaign_status}/{ticket.campaign_key}.journal.json": journal,
        f"{config.broker.campaign_status}/{ticket.campaign_key}.status.json": status,
        f"{config.broker.launch_tickets}/{ticket.campaign_key}.ticket.json": ticket,
        f"{config.broker.campaign_status}/{ticket.campaign_key}.generation-0000000006.terminal.json": terminal,
        f"{config.broker.processing}/{submission_key}.signed.json": record,
    }
    for relative, model in files.items():
        (tmp_path / relative).write_bytes(canonical_model_bytes(model) + b"\n")
    transition = CatalogLineageTransitionV1(campaign_key=ticket.campaign_key,
        previous_request_sha256=previous.request_sha256, next_generation=7,
        target_definition_sha256="a" * 64, target_prompt_sha256="b" * 64)
    return config, public, transition, files


@pytest.mark.parametrize("defect", [None, "pending_input", "unverified_signature", "missing_terminal", "request_hash_as_submission_key"])
def test_migration_file_proposal_authenticates_history_without_writing_spool(tmp_path, defect):
    from aurora.infra.sp500_megarun.catalog_lineage_migration import prepare_available_lineage_files

    config, public, transition, files = _disk_state(tmp_path,
        use_request_hash_as_submission_key=defect == "request_hash_as_submission_key")
    if defect == "pending_input":
        (tmp_path / config.broker.inbox / "pending.json").write_text("{}")
    elif defect == "unverified_signature":
        public = b"invalid public key"
    elif defect == "missing_terminal":
        next(tmp_path.rglob("*.terminal.json")).unlink()
    before = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    def prepare():
        return prepare_available_lineage_files(broker_root=tmp_path, config=config,
            transition=transition, public_key=public, observed_at=NOW + timedelta(seconds=1))
    if defect:
        with pytest.raises(ValueError, match="REQUESTER_LINEAGE_"):
            prepare()
    else:
        records = prepare()
        assert len(records) == 3
        assert {row["path"] for row in records} == {
            "CatalogRequester/" + relative for relative in files
            if relative.endswith((".journal.json", ".status.json", ".ticket.json"))
        }
        for row in records:
            old = before[row["path"].removeprefix("CatalogRequester/")]
            assert row["expected_old_sha256"] == hashlib.sha256(old).hexdigest()
            assert row["sha256"] == hashlib.sha256(row["content"]).hexdigest()
            assert json.loads(row["content"])["schema_version"] == "1"
    assert {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("defect", [None, "target_definition", "target_prompt", "public_key", "approval_missing"])
def test_candidate_migration_binds_approval_to_packaged_context(tmp_path, defect, intermediate):
    from aurora.infra.sp500_megarun.catalog_lineage_migration import prepare_candidate_lineage_files
    from aurora.infra.sp500_megarun.catalog_campaign_definition_contract import parse_catalog_campaign_definition_bytes

    live = tmp_path / "live"
    live.mkdir()
    config, public, transition, files = _disk_state(live)
    if intermediate:
        from aurora.infra.sp500_megarun.catalog_request_contract import canonical_model_bytes

        ticket = next(model for relative, model in files.items() if relative.endswith(".ticket.json"))
        ticket = CatalogLaunchTicketV1.model_validate({**ticket.model_dump(),
            "campaign_definition_sha256": "c" * 64, "prompt_sha256": "d" * 64})
        updated = {
            f"{config.broker.launch_tickets}/{ticket.campaign_key}.ticket.json": ticket,
            f"{config.broker.campaign_status}/{ticket.campaign_key}.journal.json": _ticket_journal(
                ticket=ticket, state="available", submission_key_sha256=None, request_sha256=None,
                issue_number=None, created_at=NOW, updated_at=NOW),
            f"{config.broker.campaign_status}/{ticket.campaign_key}.status.json": CatalogRequesterCampaignStatusV1.create(
                campaign_key=ticket.campaign_key, state="ticket_available", launch_generation=7,
                launch_ticket_sha256=ticket.launch_ticket_sha256, updated_at=NOW),
        }
        for relative, model in updated.items():
            (live / relative).write_bytes(canonical_model_bytes(model) + b"\n")
        transition = type(transition).model_validate({**transition.model_dump(), "source_ticket_contexts": [{
            "campaign_definition_sha256": "c" * 64, "prompt_sha256": "d" * 64,
        }]})
    source = Path(__file__).resolve().parents[1]
    candidate = tmp_path / "candidate"
    registry = json.loads((source / "config/catalog_campaign_registry_v1.json").read_bytes())
    entry = next(row for row in registry["campaigns"] if row["campaign_key"] == transition.campaign_key)
    manifest_bytes = (source / entry["definition_manifest_path"]).read_bytes()
    manifest = parse_catalog_campaign_definition_bytes(manifest_bytes)
    prompt = (source / "docs/runbooks/CATALOG_RUN_MASTER_PROMPT.md").read_bytes()
    transition = transition.model_copy(update={"target_definition_sha256": manifest.campaign_definition_sha256,
        "target_prompt_sha256": hashlib.sha256(prompt).hexdigest()})
    if defect in {"target_definition", "target_prompt"}:
        transition = transition.model_copy(update={defect + "_sha256": "f" * 64})
    assets = {
        "config/catalog_requester_v1.json": config.model_dump_json().encode(),
        "config/catalog_requester_public_key_v1.pem": public,
        "config/catalog_campaign_registry_v1.json": json.dumps({"schema_version": "1", "campaigns": [entry]}).encode(),
        entry["definition_manifest_path"]: manifest_bytes,
        "docs/runbooks/CATALOG_RUN_MASTER_PROMPT.md": prompt,
        "config/catalog_lineage_transitions_v1.json": json.dumps({"schema_version": "1", "transitions":
            [] if defect == "approval_missing" else [transition.model_dump(mode="json")]}).encode(),
    }
    for relative, content in assets.items():
        path = candidate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (live / "config").mkdir()
    for relative in ("config/catalog_requester_v1.json", "config/catalog_requester_public_key_v1.pem"):
        (live / relative).write_bytes(assets[relative])
    if defect == "public_key":
        (candidate / "config/catalog_requester_public_key_v1.pem").write_bytes(b"different key")
    before = {path.relative_to(live): path.read_bytes() for path in live.rglob("*") if path.is_file()}
    if defect:
        with pytest.raises(ValueError, match="REQUESTER_LINEAGE_"):
            prepare_candidate_lineage_files(broker_root=live, candidate_root=candidate, observed_at=NOW)
    else:
        records = prepare_candidate_lineage_files(broker_root=live, candidate_root=candidate, observed_at=NOW)
        assert len(records) == 3
        ticket = json.loads(next(row["content"] for row in records if row["path"].endswith(".ticket.json")))
        assert ticket["campaign_definition_sha256"] == manifest.campaign_definition_sha256
        assert ticket["prompt_sha256"] == hashlib.sha256(prompt).hexdigest()
    assert {path.relative_to(live): path.read_bytes() for path in live.rglob("*") if path.is_file()} == before
    if defect is None:
        for row in records:
            (live / row["path"].removeprefix("CatalogRequester/")).write_bytes(row["content"])
        assert prepare_candidate_lineage_files(broker_root=live, candidate_root=candidate,
            observed_at=NOW + timedelta(seconds=1)) == ()
