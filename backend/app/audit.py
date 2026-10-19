"""Bayora Tamper-Evident Audit Logging Module.

This module implements an append-only, cryptographic hash-chained audit logging
system for Bayora, an AI safety testing infrastructure prototype developed for
Hack in Hills '26.

Architecture & Security Model:
------------------------------
1. Audit Event Model:
   - Immutable, typed `AuditEvent` records tracking actor, action, tenant, resource,
     timestamps, outcome, and cryptographic hash links.
   - All events are deterministically serialized to canonical JSON before hashing.

2. Cryptographic Hash-Chain Integrity:
   - Every event computes a SHA-256 digest over its canonical fields plus the
     preceding event's hash (`previous_hash`).
   - The initial event links to a fixed `GENESIS_HASH` (64 zeros).
   - Any modification, deletion, reordering, or insertion breaks the chain
     and is detected by `verify_audit_chain`.
   - Realistic Security Posture & Limitations:
     A hash chain stored on a standard writable local filesystem guarantees
     tamper-evidence against accidental modification or low-privileged actors,
     but CANNOT independently prevent an attacker with full filesystem write/root
     privileges from rewriting the entire log file and recomputing all hashes.
     In a production system, hashes must be periodically anchored to an external
     immutable destination (e.g. WORM storage, Sigstore/Rekor transparency log,
     or append-only S3 bucket with Object Lock).

3. Append-Only Storage & Concurrency:
   - Audit logs are stored in a dedicated file (`audit/audit_log.jsonl`) under the
     project's configured storage root (`BAYORA_STORAGE_DIR`).
   - Intra-process concurrency is serialized using a reentrant lock (`threading.RLock`)
     so concurrent appends cannot interleave or break the sequence/hash link.
   - Every append flushes and fsyncs to disk (`os.fsync`) to prevent partial line writes.
   - Concurrency Limitation: Cross-process concurrent append safety requires an external
     distributed database or OS-level file locking not present in this local JSON prototype.

4. Privacy, Least Privilege & Redaction:
   - Secrets, API keys, passwords, authorization headers, and raw attack/model
     payloads are strictly scrubbed (`_sanitize_metadata`) and never stored in audit logs.
   - Tenant scoping: Non-admin actors can list only audit records belonging to their
     authenticated `principal.tenant_id`.
   - Admin boundaries: Admins can inspect operational audit metadata, but this does
     NOT grant them access to private tenant artifact contents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

try:
    from backend.app.auth import Permission, Principal, Role
except ImportError:
    from .auth import Permission, Principal, Role

try:
    from backend.app.storage import get_storage_root, validate_identifier
except ImportError:
    from .storage import get_storage_root, validate_identifier

logger = logging.getLogger(__name__)

# Fixed genesis hash for the initial entry in the audit ledger
GENESIS_HASH = "0" * 64

# Subdirectory and file name for audit log ledger
AUDIT_SUBDIR_NAME = "audit"
AUDIT_LOG_FILENAME = "audit_log.jsonl"

# Concurrency lock for serializing file appends within the Python process
_AUDIT_LOCK = threading.RLock()

# Patterns of sensitive metadata keys that must never be recorded in audit logs
SENSITIVE_KEY_PATTERN = re.compile(
    r"(key|secret|token|password|auth|credential|payload|prompt|content|bearer)",
    re.IGNORECASE,
)


# ============================================================================
# Audit Exceptions
# ============================================================================

class AuditError(Exception):
    """Base exception for all audit logging errors."""


class AuditAccessDeniedError(AuditError):
    """Raised when an actor lacks authorization to inspect audit records."""


class AuditCorruptedRecordError(AuditError):
    """Raised when an audit log entry contains corrupted or unparseable JSON."""


class AuditVerificationError(AuditError):
    """Raised when an audit chain verification check fails."""


# ============================================================================
# Audit Event Model
# ============================================================================

@dataclass(frozen=True)
class AuditEvent:
    """Immutable record of an auditable system event with hash chaining.

    Attributes:
        event_id: Unique event identifier (UUID-based).
        sequence_number: Monotonically increasing 1-indexed sequence counter.
        timestamp: ISO 8601 UTC timestamp of event generation.
        actor_id: Authenticated actor identifier (or 'system').
        actor_role: Assigned role of the actor at the time of the event.
        action: Standardized action name (e.g., 'artifact.create', 'rule.delete').
        outcome: Outcome status ('success' or 'failure').
        previous_hash: SHA-256 digest of the immediately preceding event (or GENESIS_HASH).
        current_hash: SHA-256 digest computed over this event's canonical fields.
        tenant_id: Owning tenant identifier, if applicable.
        resource_type: Type of resource affected (e.g., 'red_team_artifact', 'blue_team_rule').
        resource_id: Identifier of the resource affected, if applicable.
        metadata: Safe, non-sensitive context metadata.
    """

    event_id: str
    sequence_number: int
    timestamp: str
    actor_id: str
    actor_role: str
    action: str
    outcome: str
    previous_hash: str
    current_hash: str
    tenant_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the audit event to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditEvent:
        """Instantiate an AuditEvent from dictionary data.

        Raises:
            AuditCorruptedRecordError: If required fields are missing.
        """
        required_fields = {
            "event_id",
            "sequence_number",
            "timestamp",
            "actor_id",
            "actor_role",
            "action",
            "outcome",
            "previous_hash",
            "current_hash",
        }
        missing = required_fields - set(data.keys())
        if missing:
            raise AuditCorruptedRecordError(f"Missing required audit event fields: {sorted(missing)}")

        return cls(
            event_id=str(data["event_id"]),
            sequence_number=int(data["sequence_number"]),
            timestamp=str(data["timestamp"]),
            actor_id=str(data["actor_id"]),
            actor_role=str(data["actor_role"]),
            action=str(data["action"]),
            outcome=str(data["outcome"]),
            previous_hash=str(data["previous_hash"]),
            current_hash=str(data["current_hash"]),
            tenant_id=str(data["tenant_id"]) if data.get("tenant_id") is not None else None,
            resource_type=str(data["resource_type"]) if data.get("resource_type") is not None else None,
            resource_id=str(data["resource_id"]) if data.get("resource_id") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class AuditVerificationResult:
    """Outcome of an audit chain verification check.

    Attributes:
        is_valid: True if all hashes, links, and sequences are verified.
        verified_count: Total number of events inspected.
        first_sequence: Sequence number of the first verified event (if any).
        last_sequence: Sequence number of the last verified event (if any).
        last_hash: Digest of the final verified event (if any).
        error_message: Human-readable explanation if verification failed.
        failing_event_id: Event ID where discrepancy was discovered (if invalid).
        failing_sequence: Sequence number where discrepancy was discovered (if invalid).
    """

    is_valid: bool
    verified_count: int
    first_sequence: Optional[int] = None
    last_sequence: Optional[int] = None
    last_hash: Optional[str] = None
    error_message: Optional[str] = None
    failing_event_id: Optional[str] = None
    failing_sequence: Optional[int] = None


# ============================================================================
# Cryptographic Hashing Helpers
# ============================================================================

def compute_event_hash(
    event_id: str,
    sequence_number: int,
    timestamp: str,
    actor_id: str,
    actor_role: str,
    action: str,
    outcome: str,
    previous_hash: str,
    tenant_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Compute the deterministic SHA-256 digest over canonical event fields.

    Uses sorted keys, compact separators, and UTF-8 encoding to guarantee
    cross-platform determinism.
    """
    canonical_payload = {
        "action": str(action),
        "actor_id": str(actor_id),
        "actor_role": str(actor_role),
        "event_id": str(event_id),
        "metadata": dict(metadata or {}),
        "outcome": str(outcome),
        "previous_hash": str(previous_hash),
        "resource_id": str(resource_id) if resource_id is not None else None,
        "resource_type": str(resource_type) if resource_type is not None else None,
        "sequence_number": int(sequence_number),
        "tenant_id": str(tenant_id) if tenant_id is not None else None,
        "timestamp": str(timestamp),
    }

    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def _sanitize_metadata(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Scrub sensitive keys and truncate excessively large values from metadata.

    Guarantees that credentials, raw payloads, and adversarial attack bodies
    are never stored in the audit log.
    """
    if not metadata:
        return {}

    cleaned: dict[str, Any] = {}
    for key, val in metadata.items():
        key_str = str(key)
        if SENSITIVE_KEY_PATTERN.search(key_str):
            cleaned[key_str] = "[REDACTED]"
            continue

        if isinstance(val, str):
            # Truncate strings longer than 256 characters to prevent log bloat / payload leakage
            if len(val) > 256:
                cleaned[key_str] = val[:253] + "..."
            else:
                cleaned[key_str] = val
        elif isinstance(val, (int, float, bool)) or val is None:
            cleaned[key_str] = val
        elif isinstance(val, Mapping):
            cleaned[key_str] = _sanitize_metadata(val)
        else:
            cleaned[key_str] = str(val)[:256]

    return cleaned


# ============================================================================
# Storage File Resolution
# ============================================================================

def get_audit_log_path(storage_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the canonical filesystem Path to the audit log JSONL file.

    Args:
        storage_root: Optional custom storage root path.

    Returns:
        Canonical Path to the audit log file.
    """
    root = get_storage_root(storage_root)
    audit_dir = root / AUDIT_SUBDIR_NAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / AUDIT_LOG_FILENAME


# ============================================================================
# Append Operation
# ============================================================================

def append_audit_event(
    action: str,
    outcome: str = "success",
    principal: Optional[Principal] = None,
    actor_id: Optional[str] = None,
    actor_role: Optional[Union[Role, str]] = None,
    tenant_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    storage_root: Optional[Union[str, Path]] = None,
) -> AuditEvent:
    """Record and cryptographically chain a new audit event in the append-only log.

    When an authenticated `principal` is provided, actor identity, role, and tenant
    scope are derived exclusively from the principal rather than caller-supplied arguments.

    Args:
        action: Descriptive action string (e.g. 'artifact.create', 'rule.delete').
        outcome: Status outcome ('success' or 'failure').
        principal: Authenticated caller identity from auth module (preferred).
        actor_id: Fallback actor identifier if principal is omitted.
        actor_role: Fallback actor role if principal is omitted.
        tenant_id: Fallback tenant identifier if principal is omitted.
        resource_type: Category of resource affected (optional).
        resource_id: Identifier of resource affected (optional).
        metadata: Supplementary contextual data (sanitized automatically).
        storage_root: Optional custom storage directory.

    Returns:
        The created, hashed, and persisted AuditEvent.

    Raises:
        AuditError: If persisting the audit entry fails.
    """
    # 1. Establish verified identity
    if principal is not None:
        effective_actor_id = principal.actor_id
        effective_actor_role = principal.role.value if hasattr(principal.role, "value") else str(principal.role)
        effective_tenant_id = principal.tenant_id
    else:
        effective_actor_id = str(actor_id).strip() if actor_id else "system"
        if isinstance(actor_role, Role):
            effective_actor_role = actor_role.value
        else:
            effective_actor_role = str(actor_role).strip() if actor_role else "system"
        effective_tenant_id = str(tenant_id).strip() if tenant_id else None

    # Sanitize and validate inputs
    clean_action = str(action).strip()
    if not clean_action:
        clean_action = "unknown.action"

    clean_outcome = "success" if str(outcome).strip().lower() in ("success", "ok", "true") else "failure"
    safe_metadata = _sanitize_metadata(metadata)

    log_file = get_audit_log_path(storage_root)

    with _AUDIT_LOCK:
        last_seq = 0
        last_hash = GENESIS_HASH

        # Determine last event's hash and sequence number
        if log_file.exists() and log_file.stat().st_size > 0:
            try:
                # Read lines to locate the last valid record
                with log_file.open("r", encoding="utf-8") as f:
                    last_line = None
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            last_line = stripped

                    if last_line:
                        last_record = json.loads(last_line)
                        last_seq = int(last_record["sequence_number"])
                        last_hash = str(last_record["current_hash"])
            except Exception as err:
                logger.error("Failed reading existing audit log to determine chain state: %s", str(err))
                raise AuditError("Audit log reading failed during chain append.") from err

        next_seq = last_seq + 1
        event_id = f"evt_{uuid.uuid4().hex}"
        timestamp = datetime.now(timezone.utc).isoformat()

        current_hash = compute_event_hash(
            event_id=event_id,
            sequence_number=next_seq,
            timestamp=timestamp,
            actor_id=effective_actor_id,
            actor_role=effective_actor_role,
            action=clean_action,
            outcome=clean_outcome,
            previous_hash=last_hash,
            tenant_id=effective_tenant_id,
            resource_type=str(resource_type) if resource_type else None,
            resource_id=str(resource_id) if resource_id else None,
            metadata=safe_metadata,
        )

        event = AuditEvent(
            event_id=event_id,
            sequence_number=next_seq,
            timestamp=timestamp,
            actor_id=effective_actor_id,
            actor_role=effective_actor_role,
            action=clean_action,
            outcome=clean_outcome,
            previous_hash=last_hash,
            current_hash=current_hash,
            tenant_id=effective_tenant_id,
            resource_type=str(resource_type) if resource_type else None,
            resource_id=str(resource_id) if resource_id else None,
            metadata=safe_metadata,
        )

        try:
            line_to_write = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line_to_write)
                f.flush()
                os.fsync(f.fileno())
        except Exception as err:
            logger.error("Failed writing audit event '%s': %s", event_id, str(err))
            raise AuditError("Failed to persist audit event.") from err

    logger.debug("Appended audit event '%s' (seq=%d)", event_id, next_seq)
    return event


# ============================================================================
# List Operation with Authorization & Scoping
# ============================================================================

def list_audit_events(
    principal: Principal,
    tenant_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: Optional[int] = None,
    storage_root: Optional[Union[str, Path]] = None,
) -> list[AuditEvent]:
    """Retrieve authorized audit log records based on caller role and tenant scope.

    Access Rules:
    - Administrators (Role.ADMIN or Permission.READ_AUDIT_LOGS) can view all events
      across the system, or optionally filter by tenant_id.
    - Tenant actors (Role.RED_TEAM, Role.BLUE_TEAM) can ONLY view events belonging
      to their own `principal.tenant_id`. Requesting another tenant's records raises
      `AuditAccessDeniedError`.
    - Unauthenticated or unauthorized actors are denied with `AuditAccessDeniedError`.

    Args:
        principal: The authenticated caller identity.
        tenant_id: Optional tenant filter (admins only; ignored/enforced for tenant actors).
        action: Optional filter by action name.
        resource_type: Optional filter by resource type.
        limit: Optional maximum number of recent events to return.
        storage_root: Optional custom storage directory.

    Returns:
        List of matching AuditEvent instances.

    Raises:
        AuditAccessDeniedError: If the caller is not authorized.
        AuditCorruptedRecordError: If stored audit lines cannot be parsed.
    """
    is_admin = principal.has_role(Role.ADMIN) or principal.has_permission(Permission.READ_AUDIT_LOGS)

    # Determine effective tenant scope
    if is_admin:
        effective_tenant = tenant_id.strip() if tenant_id else None
    else:
        # Non-admin actors MUST have a tenant_id and can only inspect their own tenant
        if not principal.tenant_id or not principal.tenant_id.strip():
            raise AuditAccessDeniedError(
                f"Actor '{principal.actor_id}' has no assigned tenant and cannot inspect audit logs."
            )

        if tenant_id and tenant_id.strip() != principal.tenant_id:
            logger.warning(
                "Audit access denied: Actor '%s' (tenant '%s') attempted to view tenant '%s'",
                principal.actor_id,
                principal.tenant_id,
                tenant_id,
            )
            raise AuditAccessDeniedError("Access denied: cannot view audit logs of another tenant.")

        effective_tenant = principal.tenant_id

    log_file = get_audit_log_path(storage_root)
    if not log_file.exists():
        return []

    events: list[AuditEvent] = []

    with _AUDIT_LOCK:
        try:
            with log_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record_dict = json.loads(stripped)
                        event = AuditEvent.from_dict(record_dict)
                    except Exception as err:
                        logger.error("Audit log line %d corrupted: %s", line_num, str(err))
                        raise AuditCorruptedRecordError(
                            f"Corrupted record encountered at audit log line {line_num}."
                        ) from err

                    # Apply tenant filter
                    if effective_tenant is not None and event.tenant_id != effective_tenant:
                        continue

                    # Apply action filter
                    if action and event.action != action:
                        continue

                    # Apply resource_type filter
                    if resource_type and event.resource_type != resource_type:
                        continue

                    events.append(event)
        except AuditCorruptedRecordError:
            raise
        except Exception as err:
            logger.error("Failed reading audit log file: %s", str(err))
            raise AuditError("Could not read audit events from storage.") from err

    if limit is not None and limit > 0:
        events = events[-limit:]

    return events


# ============================================================================
# Cryptographic Chain Verification
# ============================================================================

def verify_audit_chain(
    events: Optional[Sequence[AuditEvent]] = None,
    expected_start_hash: Optional[str] = None,
    storage_root: Optional[Union[str, Path]] = None,
) -> AuditVerificationResult:
    """Verify cryptographic integrity, hash links, and sequence ordering of the audit chain.

    Verification checks:
    1. Recomputed SHA-256 over canonical fields matches `event.current_hash`.
    2. Sequence numbers are strictly consecutive (seq[i] == seq[i-1] + 1).
    3. The first event's previous_hash matches `expected_start_hash` (or `GENESIS_HASH` if seq=1).
    4. Each subsequent event's `previous_hash` matches `event[i-1].current_hash`.

    Subset Verification:
    - If `events` is a subset of the full log starting at seq > 1, provide `expected_start_hash`
      to verify linkage to the rest of the chain. Without `expected_start_hash`, verification
      confirms internal consistency among the provided events, but cannot prove connection
      to the genesis block.

    Args:
        events: Optional sequence of AuditEvent instances. If None, reads the entire file.
        expected_start_hash: Expected previous_hash for the first event in the verified sequence.
        storage_root: Optional custom storage directory.

    Returns:
        AuditVerificationResult with validation details and failure diagnostics.
    """
    event_list: list[AuditEvent] = []

    if events is None:
        log_file = get_audit_log_path(storage_root)
        if not log_file.exists():
            return AuditVerificationResult(
                is_valid=True,
                verified_count=0,
                error_message=None,
            )

        with _AUDIT_LOCK:
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            event_list.append(AuditEvent.from_dict(json.loads(stripped)))
                        except Exception as err:
                            return AuditVerificationResult(
                                is_valid=False,
                                verified_count=len(event_list),
                                error_message=f"Corrupted record at line {line_num}: {str(err)}",
                            )
            except Exception as err:
                return AuditVerificationResult(
                    is_valid=False,
                    verified_count=0,
                    error_message=f"Failed reading audit log: {str(err)}",
                )
    else:
        event_list = list(events)

    if not event_list:
        return AuditVerificationResult(
            is_valid=True,
            verified_count=0,
        )

    # 1. Check starting link
    first_event = event_list[0]

    if expected_start_hash is not None:
        if first_event.previous_hash != expected_start_hash:
            return AuditVerificationResult(
                is_valid=False,
                verified_count=0,
                first_sequence=first_event.sequence_number,
                error_message=(
                    f"First event previous_hash mismatch: expected '{expected_start_hash}', "
                    f"found '{first_event.previous_hash}'."
                ),
                failing_event_id=first_event.event_id,
                failing_sequence=first_event.sequence_number,
            )
    else:
        # Full chain verification must start at sequence 1 with GENESIS_HASH
        if events is None and first_event.sequence_number != 1:
            return AuditVerificationResult(
                is_valid=False,
                verified_count=0,
                first_sequence=first_event.sequence_number,
                error_message=f"Missing genesis event: expected sequence 1, found {first_event.sequence_number}.",
                failing_event_id=first_event.event_id,
                failing_sequence=first_event.sequence_number,
            )

        if first_event.sequence_number == 1 and first_event.previous_hash != GENESIS_HASH:
            return AuditVerificationResult(
                is_valid=False,
                verified_count=0,
                first_sequence=first_event.sequence_number,
                error_message=(
                    f"Genesis event previous_hash mismatch: expected '{GENESIS_HASH}', "
                    f"found '{first_event.previous_hash}'."
                ),
                failing_event_id=first_event.event_id,
                failing_sequence=first_event.sequence_number,
            )

    # 2. Sequential verification
    for i, event in enumerate(event_list):
        # Verify self-hash integrity
        expected_hash = compute_event_hash(
            event_id=event.event_id,
            sequence_number=event.sequence_number,
            timestamp=event.timestamp,
            actor_id=event.actor_id,
            actor_role=event.actor_role,
            action=event.action,
            outcome=event.outcome,
            previous_hash=event.previous_hash,
            tenant_id=event.tenant_id,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            metadata=event.metadata,
        )

        if event.current_hash != expected_hash:
            return AuditVerificationResult(
                is_valid=False,
                verified_count=i,
                first_sequence=first_event.sequence_number,
                last_sequence=event_list[i - 1].sequence_number if i > 0 else None,
                last_hash=event_list[i - 1].current_hash if i > 0 else None,
                error_message=f"Event content modified: current_hash does not match recomputed SHA-256 digest.",
                failing_event_id=event.event_id,
                failing_sequence=event.sequence_number,
            )

        # Verify chain linkage to preceding event
        if i > 0:
            prev_event = event_list[i - 1]
            if event.sequence_number != prev_event.sequence_number + 1:
                return AuditVerificationResult(
                    is_valid=False,
                    verified_count=i,
                    first_sequence=first_event.sequence_number,
                    last_sequence=prev_event.sequence_number,
                    last_hash=prev_event.current_hash,
                    error_message=(
                        f"Sequence break: expected seq {prev_event.sequence_number + 1}, "
                        f"found {event.sequence_number}."
                    ),
                    failing_event_id=event.event_id,
                    failing_sequence=event.sequence_number,
                )

            if event.previous_hash != prev_event.current_hash:
                return AuditVerificationResult(
                    is_valid=False,
                    verified_count=i,
                    first_sequence=first_event.sequence_number,
                    last_sequence=prev_event.sequence_number,
                    last_hash=prev_event.current_hash,
                    error_message=(
                        f"Chain linkage broken: previous_hash '{event.previous_hash}' does not "
                        f"match preceding event's current_hash '{prev_event.current_hash}'."
                    ),
                    failing_event_id=event.event_id,
                    failing_sequence=event.sequence_number,
                )

    return AuditVerificationResult(
        is_valid=True,
        verified_count=len(event_list),
        first_sequence=first_event.sequence_number,
        last_sequence=event_list[-1].sequence_number,
        last_hash=event_list[-1].current_hash,
    )
