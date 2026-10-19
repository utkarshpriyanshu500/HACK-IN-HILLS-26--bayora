"""Bayora Tenant-Scoped Storage Layer.

This module provides secure, tenant-scoped local JSON storage for Bayora, an AI
safety testing infrastructure prototype for Hack in Hills '26.

Architecture & Security Model:
------------------------------
1. Tenant Isolation & Access Control:
   - Every protected storage operation requires an authenticated `Principal` from
     `backend.app.auth`.
   - Tenant scope is derived exclusively from `principal.tenant_id`. Callers cannot
     override or supply an untrusted tenant identifier.
   - Operations requiring tenant scope fail immediately with `TenantRequiredError`
     if `principal.tenant_id` is missing or empty.
   - Strict role boundaries are enforced:
     * `red_team`: Authorized only for red-team artifacts.
     * `blue_team`: Authorized only for blue-team defense rules.
     * `admin`: Cannot access or list private red-team artifacts or blue-team
       defense rules unless explicitly granted by policy.
     * Cross-role access is denied with `StorageAccessDeniedError`.

2. Filesystem Safety & Path Traversal Prevention:
   - All records are confined to a dedicated storage root (`BAYORA_STORAGE_DIR`).
   - Tenant IDs and record IDs are strictly validated using a whitelist regex
     (`^[a-zA-Z0-9_-]{1,64}$`). Path separators, `..`, null bytes, and non-ASCII
     characters are rejected with `InvalidIdentifierError`.
   - Resolved filesystem paths are verified with `Path.is_relative_to` before any
     file read, write, or deletion to guarantee no path traversal is possible.

3. Concurrency & Atomic Writes:
   - Writes are executed atomically: records are written and flushed to a temporary
     file in the target directory and then atomically replaced (`os.replace`).
   - Intra-process concurrency is guarded using an reentrant lock (`threading.RLock`).
   - Note on Concurrency Limits: Atomic file replacement protects against file
     corruption and partial writes under concurrent access. For multi-node distributed
     deployments, external database persistence with row-level locking should replace
     this local JSON prototype.

4. Data Integrity & Confidentiality:
   - Corrupted or unparseable JSON files raise `CorruptedDataError`.
   - Exceptions and logs deliberately avoid exposing sensitive artifact or rule payloads.
   - Storage files are kept in a dedicated directory separate from public or static assets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

try:
    from backend.app.auth import Permission, Principal, Role
except ImportError:
    from .auth import Permission, Principal, Role

logger = logging.getLogger(__name__)

# Environment variable for custom storage root directory
STORAGE_DIR_ENV_VAR = "BAYORA_STORAGE_DIR"
DEFAULT_STORAGE_DIR = "data/storage"

# Whitelist pattern for tenant, artifact, and rule identifiers (1-64 characters)
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Process-level lock for concurrent write coordination
_STORAGE_LOCK = threading.RLock()


# ============================================================================
# Storage Exceptions Hierarchy
# ============================================================================

class StorageError(Exception):
    """Base exception for all storage layer errors."""


class TenantRequiredError(StorageError):
    """Raised when an operation requires an assigned tenant_id on the Principal."""


class StorageAccessDeniedError(StorageError):
    """Raised when an authenticated principal lacks required role or permission."""


class RecordNotFoundError(StorageError):
    """Raised when a requested record does not exist in the principal's tenant scope."""


class RecordAlreadyExistsError(StorageError):
    """Raised when attempting to create a record with an ID that already exists."""


class InvalidIdentifierError(StorageError):
    """Raised when a record or tenant identifier fails safety validation."""


class PathTraversalError(StorageError):
    """Raised when a resolved file path escapes its designated directory."""


class CorruptedDataError(StorageError):
    """Raised when stored record data is unparseable or corrupted."""


# ============================================================================
# Record Data Models
# ============================================================================

@dataclass(frozen=True)
class RedTeamArtifact:
    """Representation of a stored red-team testing artifact.

    Attributes:
        artifact_id: Unique record identifier.
        tenant_id: Owning tenant identifier.
        created_by: Authenticated actor ID of the creator.
        created_at: ISO 8601 UTC creation timestamp.
        updated_at: ISO 8601 UTC last update timestamp.
        name: Human-readable name or label.
        artifact_type: Classification (e.g., 'prompt_injection', 'jailbreak_template').
        content: The testing payload or artifact data.
        metadata: Additional arbitrary metadata.
    """

    artifact_id: str
    tenant_id: str
    created_by: str
    created_at: str
    updated_at: str
    name: str
    artifact_type: str
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize artifact to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RedTeamArtifact:
        """Instantiate an artifact from dictionary data.

        Raises:
            CorruptedDataError: If required fields are missing or invalid.
        """
        required_fields = {
            "artifact_id",
            "tenant_id",
            "created_by",
            "created_at",
            "updated_at",
            "name",
            "artifact_type",
            "content",
        }
        missing = required_fields - set(data.keys())
        if missing:
            raise CorruptedDataError(f"Missing required artifact fields: {sorted(missing)}")

        return cls(
            artifact_id=str(data["artifact_id"]),
            tenant_id=str(data["tenant_id"]),
            created_by=str(data["created_by"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            name=str(data["name"]),
            artifact_type=str(data["artifact_type"]),
            content=data["content"],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BlueTeamRule:
    """Representation of a stored blue-team defense rule.

    Attributes:
        rule_id: Unique record identifier.
        tenant_id: Owning tenant identifier.
        created_by: Authenticated actor ID of the creator.
        created_at: ISO 8601 UTC creation timestamp.
        updated_at: ISO 8601 UTC last update timestamp.
        name: Human-readable name or label.
        rule_type: Classification (e.g., 'input_guardrail', 'regex_filter').
        pattern_or_config: The rule specification, pattern, or configuration.
        action: Response action ('block', 'sanitize', 'flag', 'quarantine').
        enabled: Whether the rule is active.
        metadata: Additional arbitrary metadata.
    """

    rule_id: str
    tenant_id: str
    created_by: str
    created_at: str
    updated_at: str
    name: str
    rule_type: str
    pattern_or_config: Any
    action: str = "block"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize defense rule to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BlueTeamRule:
        """Instantiate a defense rule from dictionary data.

        Raises:
            CorruptedDataError: If required fields are missing or invalid.
        """
        required_fields = {
            "rule_id",
            "tenant_id",
            "created_by",
            "created_at",
            "updated_at",
            "name",
            "rule_type",
            "pattern_or_config",
        }
        missing = required_fields - set(data.keys())
        if missing:
            raise CorruptedDataError(f"Missing required rule fields: {sorted(missing)}")

        return cls(
            rule_id=str(data["rule_id"]),
            tenant_id=str(data["tenant_id"]),
            created_by=str(data["created_by"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            name=str(data["name"]),
            rule_type=str(data["rule_type"]),
            pattern_or_config=data["pattern_or_config"],
            action=str(data.get("action", "block")),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata") or {}),
        )


# ============================================================================
# Path & Identifier Safety Helpers
# ============================================================================

def get_storage_root(custom_dir: Optional[Union[str, Path]] = None) -> Path:
    """Retrieve and ensure existence of the dedicated storage root directory.

    Args:
        custom_dir: Optional override path for storage root.

    Returns:
        Canonical absolute Path pointing to the storage root.
    """
    if custom_dir:
        root_path = Path(custom_dir)
    else:
        env_path = os.environ.get(STORAGE_DIR_ENV_VAR)
        root_path = Path(env_path) if env_path else Path(DEFAULT_STORAGE_DIR)

    resolved = root_path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_identifier(identifier: str, field_name: str = "identifier") -> str:
    """Validate that an identifier conforms strictly to the safe identifier pattern.

    Rejects directory traversal sequences ('..', '/', '\\'), null bytes, control
    characters, whitespace, and excessive length.

    Args:
        identifier: The candidate identifier string.
        field_name: Contextual name of the field for error reporting.

    Returns:
        The validated clean identifier string.

    Raises:
        InvalidIdentifierError: If the identifier violates safety constraints.
    """
    if not isinstance(identifier, str):
        raise InvalidIdentifierError(f"{field_name} must be a string, received {type(identifier).__name__}.")

    clean_id = identifier.strip()
    if not clean_id:
        raise InvalidIdentifierError(f"{field_name} cannot be empty or whitespace.")

    if not SAFE_ID_REGEX.match(clean_id):
        raise InvalidIdentifierError(
            f"Invalid {field_name} format: must be 1-64 alphanumeric characters, dashes, or underscores."
        )

    return clean_id


def _is_subpath(child: Union[str, Path], parent: Union[str, Path]) -> bool:
    """Check whether child path is strictly confined within parent directory.

    Normalizes case, resolves symlinks, and strips Windows extended-path (\\\\?\\)
    prefixes to prevent false positive traversal errors on Windows NTFS.
    """
    c_str = str(Path(child).resolve())
    p_str = str(Path(parent).resolve())
    if c_str.startswith("\\\\?\\"):
        c_str = c_str[4:]
    if p_str.startswith("\\\\?\\"):
        p_str = p_str[4:]
    c_norm = os.path.normcase(os.path.abspath(c_str))
    p_norm = os.path.normcase(os.path.abspath(p_str))
    try:
        return os.path.commonpath([c_norm, p_norm]) == p_norm
    except ValueError:
        return False


def get_tenant_directory(storage_root: Path, tenant_id: str, category: str) -> Path:
    """Safely resolve the directory for a specific tenant and resource category.

    Args:
        storage_root: Canonical storage root path.
        tenant_id: Validated tenant identifier.
        category: Subdirectory category (e.g., 'red_team_artifacts', 'blue_team_rules').

    Returns:
        Canonical Path to the tenant's resource directory.

    Raises:
        PathTraversalError: If resolved path escapes the storage root.
    """
    valid_tenant = validate_identifier(tenant_id, "tenant_id")
    valid_category = validate_identifier(category, "category")

    target_dir = (storage_root / "tenants" / valid_tenant / valid_category).resolve()

    # Enforce confinement within storage_root
    if not _is_subpath(target_dir, storage_root):
        raise PathTraversalError("Resolved tenant directory escapes storage root.")

    with _STORAGE_LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def get_record_path(category_dir: Path, record_id: str) -> Path:
    """Safely resolve the file path for a specific record inside a category directory.

    Args:
        category_dir: Canonical directory for the resource category.
        record_id: Candidate record identifier.

    Returns:
        Canonical Path to the record's JSON file.

    Raises:
        PathTraversalError: If resolved path escapes the category directory.
        InvalidIdentifierError: If the record_id fails validation.
    """
    valid_record_id = validate_identifier(record_id, "record_id")
    record_file = (category_dir / f"{valid_record_id}.json").resolve()

    if not _is_subpath(record_file, category_dir):
        raise PathTraversalError("Resolved record file path escapes category directory.")

    return record_file


# ============================================================================
# Concurrency & Atomic File Operations
# ============================================================================

def _atomic_write_json(file_path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write JSON data to file_path using a temporary file and replace.

    Guarantees that readers never observe partially written files or corruption.
    Uses process-level locking for thread safety.

    Args:
        file_path: The target destination Path.
        data: Plain dictionary to serialize as JSON.

    Raises:
        StorageError: If the atomic write operation fails.
    """
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

    with _STORAGE_LOCK:
        # Create temporary file in the same directory to allow atomic os.replace across filesystem boundaries
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=parent_dir,
                prefix="tmp_bayora_",
                suffix=".json",
                delete=False,
            ) as tmp:
                temp_file = Path(tmp.name)
                tmp.write(json_bytes)
                tmp.flush()
                os.fsync(tmp.fileno())

            # Atomic replace of destination file
            os.replace(temp_file, file_path)
            temp_file = None
        except Exception as err:
            logger.error("Failed atomic write to '%s': %s", file_path.name, str(err))
            raise StorageError(f"Failed to securely persist record '{file_path.name}'.") from err
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass


def _read_json_file(file_path: Path) -> dict[str, Any]:
    """Read and deserialize JSON data from file_path.

    Args:
        file_path: Path to the target JSON file.

    Returns:
        Deserialized dictionary.

    Raises:
        RecordNotFoundError: If the file does not exist.
        CorruptedDataError: If the file contains invalid or unparseable JSON.
    """
    if not file_path.exists() or not file_path.is_file():
        raise RecordNotFoundError(f"Record '{file_path.stem}' not found.")

    with _STORAGE_LOCK:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as err:
            logger.error("Failed reading file '%s': %s", file_path.name, str(err))
            raise StorageError("Could not read record from filesystem.") from err

    try:
        parsed = json.loads(content)
    except Exception as err:
        logger.error("Corrupted JSON data in '%s': %s", file_path.name, str(err))
        raise CorruptedDataError(f"Corrupted or invalid JSON data in record '{file_path.stem}'.") from err

    if not isinstance(parsed, dict):
        raise CorruptedDataError(f"Record data in '{file_path.stem}' is not a JSON object.")

    return parsed


# ============================================================================
# Authorization & Scope Guards
# ============================================================================

def _require_tenant_id(principal: Principal) -> str:
    """Extract and validate tenant_id from the authenticated Principal.

    Enforces that tenant scope is derived server-side from the authenticated identity.

    Args:
        principal: Authenticated principal.

    Returns:
        Validated non-empty tenant identifier string.

    Raises:
        TenantRequiredError: If the principal does not possess an assigned tenant_id.
    """
    if not principal.tenant_id or not principal.tenant_id.strip():
        logger.warning("Storage access rejected: Actor '%s' lacks tenant identity.", principal.actor_id)
        raise TenantRequiredError(
            f"Authenticated actor '{principal.actor_id}' has no assigned tenant scope."
        )
    return validate_identifier(principal.tenant_id, "tenant_id")


def _require_red_team_role(principal: Principal) -> None:
    """Enforce that the caller holds the authorized red_team role or permission.

    Raises:
        StorageAccessDeniedError: If the caller is not authorized for red-team artifacts.
    """
    is_authorized = (
        principal.has_role(Role.RED_TEAM)
        or principal.has_permission(Permission.MANAGE_RED_TEAM_ARTIFACTS)
    )
    if not is_authorized:
        logger.warning(
            "Access denied: Actor '%s' with role '%s' attempted red-team artifact operation.",
            principal.actor_id,
            principal.role.value if hasattr(principal.role, "value") else str(principal.role),
        )
        raise StorageAccessDeniedError(
            f"Actor '{principal.actor_id}' is not authorized to access red-team artifacts."
        )


def _require_blue_team_role(principal: Principal) -> None:
    """Enforce that the caller holds the authorized blue_team role or permission.

    Raises:
        StorageAccessDeniedError: If the caller is not authorized for blue-team rules.
    """
    is_authorized = (
        principal.has_role(Role.BLUE_TEAM)
        or principal.has_permission(Permission.MANAGE_BLUE_TEAM_RULES)
    )
    if not is_authorized:
        logger.warning(
            "Access denied: Actor '%s' with role '%s' attempted blue-team rule operation.",
            principal.actor_id,
            principal.role.value if hasattr(principal.role, "value") else str(principal.role),
        )
        raise StorageAccessDeniedError(
            f"Actor '{principal.actor_id}' is not authorized to access blue-team defense rules."
        )


# ============================================================================
# Red-Team Artifacts Storage Operations
# ============================================================================

def create_red_team_artifact(
    principal: Principal,
    name: str,
    artifact_type: str,
    content: Any,
    artifact_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    storage_root: Optional[Union[str, Path]] = None,
) -> RedTeamArtifact:
    """Create and persist a new red-team testing artifact under the caller's tenant.

    Derives tenant scope exclusively from `principal.tenant_id`. Enforces red-team
    authorization and writes data atomically.

    Args:
        principal: Authenticated caller identity. Must have role RED_TEAM.
        name: Short descriptive name for the artifact.
        artifact_type: Category (e.g., 'prompt_injection', 'jailbreak').
        content: The artifact content or test payload.
        artifact_id: Optional custom identifier. Auto-generated if omitted.
        metadata: Optional dictionary of additional metadata.
        storage_root: Optional custom storage directory (for testing/overrides).

    Returns:
        The newly created RedTeamArtifact.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks red_team authorization.
        InvalidIdentifierError: If artifact_id fails safety validation.
        RecordAlreadyExistsError: If an artifact with the specified ID already exists.
    """
    _require_red_team_role(principal)
    tenant_id = _require_tenant_id(principal)

    if artifact_id is None:
        clean_artifact_id = f"art_{uuid.uuid4().hex[:16]}"
    else:
        clean_artifact_id = validate_identifier(artifact_id, "artifact_id")

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "red_team_artifacts")
    record_file = get_record_path(cat_dir, clean_artifact_id)

    if record_file.exists():
        raise RecordAlreadyExistsError(
            f"Red-team artifact with ID '{clean_artifact_id}' already exists in tenant '{tenant_id}'."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    artifact = RedTeamArtifact(
        artifact_id=clean_artifact_id,
        tenant_id=tenant_id,
        created_by=principal.actor_id,
        created_at=now_iso,
        updated_at=now_iso,
        name=str(name).strip() or "Unnamed Artifact",
        artifact_type=str(artifact_type).strip() or "general",
        content=content,
        metadata=dict(metadata or {}),
    )

    _atomic_write_json(record_file, artifact.to_dict())
    logger.info("Created red-team artifact '%s' for tenant '%s'", clean_artifact_id, tenant_id)
    return artifact


def get_red_team_artifact(
    principal: Principal,
    artifact_id: str,
    storage_root: Optional[Union[str, Path]] = None,
) -> RedTeamArtifact:
    """Retrieve a red-team artifact by identifier within the caller's tenant scope.

    Args:
        principal: Authenticated caller identity. Must have role RED_TEAM.
        artifact_id: Unique artifact identifier.
        storage_root: Optional custom storage directory.

    Returns:
        The matching RedTeamArtifact.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks red_team authorization.
        InvalidIdentifierError: If artifact_id fails safety validation.
        RecordNotFoundError: If artifact does not exist within the caller's tenant.
        CorruptedDataError: If the stored data is corrupted.
    """
    _require_red_team_role(principal)
    tenant_id = _require_tenant_id(principal)
    clean_id = validate_identifier(artifact_id, "artifact_id")

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "red_team_artifacts")
    record_file = get_record_path(cat_dir, clean_id)

    data = _read_json_file(record_file)
    return RedTeamArtifact.from_dict(data)


def list_red_team_artifacts(
    principal: Principal,
    storage_root: Optional[Union[str, Path]] = None,
) -> list[RedTeamArtifact]:
    """List all authorized red-team artifacts belonging strictly to the caller's tenant.

    Args:
        principal: Authenticated caller identity. Must have role RED_TEAM.
        storage_root: Optional custom storage directory.

    Returns:
        List of RedTeamArtifact instances, sorted by creation timestamp descending.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks red_team authorization.
    """
    _require_red_team_role(principal)
    tenant_id = _require_tenant_id(principal)

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "red_team_artifacts")

    artifacts: list[RedTeamArtifact] = []
    with _STORAGE_LOCK:
        for json_path in cat_dir.glob("*.json"):
            if not json_path.is_file():
                continue
            try:
                data = _read_json_file(json_path)
                artifacts.append(RedTeamArtifact.from_dict(data))
            except CorruptedDataError:
                logger.warning("Skipping corrupted artifact file: %s", json_path.name)
            except Exception as err:
                logger.error("Error reading artifact file '%s': %s", json_path.name, str(err))

    # Sort descending by created_at
    artifacts.sort(key=lambda a: a.created_at, reverse=True)
    return artifacts


def delete_red_team_artifact(
    principal: Principal,
    artifact_id: str,
    storage_root: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete a red-team artifact within the caller's tenant scope.

    Args:
        principal: Authenticated caller identity. Must have role RED_TEAM.
        artifact_id: Unique artifact identifier.
        storage_root: Optional custom storage directory.

    Returns:
        True if successfully deleted.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks red_team authorization.
        InvalidIdentifierError: If artifact_id fails safety validation.
        RecordNotFoundError: If artifact does not exist within the caller's tenant.
    """
    _require_red_team_role(principal)
    tenant_id = _require_tenant_id(principal)
    clean_id = validate_identifier(artifact_id, "artifact_id")

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "red_team_artifacts")
    record_file = get_record_path(cat_dir, clean_id)

    if not record_file.exists():
        raise RecordNotFoundError(f"Red-team artifact '{clean_id}' not found.")

    with _STORAGE_LOCK:
        try:
            record_file.unlink()
        except OSError as err:
            logger.error("Failed to delete artifact '%s': %s", clean_id, str(err))
            raise StorageError(f"Could not delete artifact '{clean_id}'.") from err

    logger.info("Deleted red-team artifact '%s' from tenant '%s'", clean_id, tenant_id)
    return True


# ============================================================================
# Blue-Team Defense Rules Storage Operations
# ============================================================================

def create_blue_team_rule(
    principal: Principal,
    name: str,
    rule_type: str,
    pattern_or_config: Any,
    rule_id: Optional[str] = None,
    action: str = "block",
    enabled: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    storage_root: Optional[Union[str, Path]] = None,
) -> BlueTeamRule:
    """Create and persist a new blue-team defense rule under the caller's tenant.

    Derives tenant scope exclusively from `principal.tenant_id`. Enforces blue-team
    authorization and writes data atomically.

    Args:
        principal: Authenticated caller identity. Must have role BLUE_TEAM.
        name: Short descriptive name for the rule.
        rule_type: Category (e.g., 'input_guardrail', 'regex_filter').
        pattern_or_config: The rule specification, pattern, or policy definition.
        rule_id: Optional custom identifier. Auto-generated if omitted.
        action: Mitigation action (default 'block').
        enabled: Initial rule state (default True).
        metadata: Optional dictionary of additional metadata.
        storage_root: Optional custom storage directory.

    Returns:
        The newly created BlueTeamRule.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks blue_team authorization.
        InvalidIdentifierError: If rule_id fails safety validation.
        RecordAlreadyExistsError: If a rule with the specified ID already exists.
    """
    _require_blue_team_role(principal)
    tenant_id = _require_tenant_id(principal)

    if rule_id is None:
        clean_rule_id = f"rule_{uuid.uuid4().hex[:16]}"
    else:
        clean_rule_id = validate_identifier(rule_id, "rule_id")

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "blue_team_rules")
    record_file = get_record_path(cat_dir, clean_rule_id)

    if record_file.exists():
        raise RecordAlreadyExistsError(
            f"Blue-team rule with ID '{clean_rule_id}' already exists in tenant '{tenant_id}'."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    rule = BlueTeamRule(
        rule_id=clean_rule_id,
        tenant_id=tenant_id,
        created_by=principal.actor_id,
        created_at=now_iso,
        updated_at=now_iso,
        name=str(name).strip() or "Unnamed Rule",
        rule_type=str(rule_type).strip() or "general",
        pattern_or_config=pattern_or_config,
        action=str(action).strip() or "block",
        enabled=bool(enabled),
        metadata=dict(metadata or {}),
    )

    _atomic_write_json(record_file, rule.to_dict())
    logger.info("Created blue-team rule '%s' for tenant '%s'", clean_rule_id, tenant_id)
    return rule


def get_blue_team_rule(
    principal: Principal,
    rule_id: str,
    storage_root: Optional[Union[str, Path]] = None,
) -> BlueTeamRule:
    """Retrieve a blue-team defense rule by identifier within the caller's tenant scope.

    Args:
        principal: Authenticated caller identity. Must have role BLUE_TEAM.
        rule_id: Unique rule identifier.
        storage_root: Optional custom storage directory.

    Returns:
        The matching BlueTeamRule.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks blue_team authorization.
        InvalidIdentifierError: If rule_id fails safety validation.
        RecordNotFoundError: If rule does not exist within the caller's tenant.
        CorruptedDataError: If the stored data is corrupted.
    """
    _require_blue_team_role(principal)
    tenant_id = _require_tenant_id(principal)
    clean_id = validate_identifier(rule_id, "rule_id")

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "blue_team_rules")
    record_file = get_record_path(cat_dir, clean_id)

    data = _read_json_file(record_file)
    return BlueTeamRule.from_dict(data)


def list_blue_team_rules(
    principal: Principal,
    storage_root: Optional[Union[str, Path]] = None,
) -> list[BlueTeamRule]:
    """List all authorized blue-team rules belonging strictly to the caller's tenant.

    Args:
        principal: Authenticated caller identity. Must have role BLUE_TEAM.
        storage_root: Optional custom storage directory.

    Returns:
        List of BlueTeamRule instances, sorted by creation timestamp descending.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks blue_team authorization.
    """
    _require_blue_team_role(principal)
    tenant_id = _require_tenant_id(principal)

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "blue_team_rules")

    rules: list[BlueTeamRule] = []
    with _STORAGE_LOCK:
        for json_path in cat_dir.glob("*.json"):
            if not json_path.is_file():
                continue
            try:
                data = _read_json_file(json_path)
                rules.append(BlueTeamRule.from_dict(data))
            except CorruptedDataError:
                logger.warning("Skipping corrupted rule file: %s", json_path.name)
            except Exception as err:
                logger.error("Error reading rule file '%s': %s", json_path.name, str(err))

    rules.sort(key=lambda r: r.created_at, reverse=True)
    return rules


def delete_blue_team_rule(
    principal: Principal,
    rule_id: str,
    storage_root: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete a blue-team defense rule within the caller's tenant scope.

    Args:
        principal: Authenticated caller identity. Must have role BLUE_TEAM.
        rule_id: Unique rule identifier.
        storage_root: Optional custom storage directory.

    Returns:
        True if successfully deleted.

    Raises:
        TenantRequiredError: If principal lacks an assigned tenant_id.
        StorageAccessDeniedError: If caller lacks blue_team authorization.
        InvalidIdentifierError: If rule_id fails safety validation.
        RecordNotFoundError: If rule does not exist within the caller's tenant.
    """
    _require_blue_team_role(principal)
    tenant_id = _require_tenant_id(principal)
    clean_id = validate_identifier(rule_id, "rule_id")

    root = get_storage_root(storage_root)
    cat_dir = get_tenant_directory(root, tenant_id, "blue_team_rules")
    record_file = get_record_path(cat_dir, clean_id)

    if not record_file.exists():
        raise RecordNotFoundError(f"Blue-team defense rule '{clean_id}' not found.")

    with _STORAGE_LOCK:
        try:
            record_file.unlink()
        except OSError as err:
            logger.error("Failed to delete rule '%s': %s", clean_id, str(err))
            raise StorageError(f"Could not delete rule '{clean_id}'.") from err

    logger.info("Deleted blue-team rule '%s' from tenant '%s'", clean_id, tenant_id)
    return True
