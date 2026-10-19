"""Bayora AI Safety Testing Platform - FastAPI Application Entrypoint.

This module exposes authenticated, role-protected REST APIs for the Bayora AI
safety testing infrastructure prototype (Hack in Hills '26).

Key Security Guarantees:
------------------------
1. Tenant Isolation:
   - All tenant-scoped operations derive tenant identity exclusively from the
     authenticated `Principal` (via `auth.py`). Request bodies and query parameters
     cannot override tenant scope.
2. Least Privilege:
   - Red team actors can only access red-team testing artifacts.
   - Blue team actors can only access blue-team defense rules.
   - Administrators cannot access private artifacts or rules by default.
3. Synchronous Audit Logging:
   - Security-relevant state modifications (creation, deletion, chain verification)
     are committed synchronously to the append-only cryptographic ledger (`audit.py`)
     before the API returns success.
4. Defense-in-Depth Exception Mapping:
   - Storage and audit domain exceptions are translated into standardized HTTP
     responses without leaking filesystem paths, internal stack traces, or credentials.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Sequence, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    from backend.app.auth import (
        Permission,
        Principal,
        Role,
        get_current_principal,
        require_admin,
        require_blue_team,
        require_evaluator,
        require_red_team,
        require_roles,
    )
except ImportError:
    from .auth import (
        Permission,
        Principal,
        Role,
        get_current_principal,
        require_admin,
        require_blue_team,
        require_evaluator,
        require_red_team,
        require_roles,
    )

try:
    from backend.app.storage import (
        CorruptedDataError,
        InvalidIdentifierError,
        PathTraversalError,
        RecordAlreadyExistsError,
        RecordNotFoundError,
        RedTeamArtifact,
        BlueTeamRule,
        StorageAccessDeniedError,
        StorageError,
        TenantRequiredError,
        create_blue_team_rule,
        create_red_team_artifact,
        delete_blue_team_rule,
        delete_red_team_artifact,
        get_blue_team_rule,
        get_red_team_artifact,
        list_blue_team_rules,
        list_red_team_artifacts,
    )
except ImportError:
    from .storage import (
        CorruptedDataError,
        InvalidIdentifierError,
        PathTraversalError,
        RecordAlreadyExistsError,
        RecordNotFoundError,
        RedTeamArtifact,
        BlueTeamRule,
        StorageAccessDeniedError,
        StorageError,
        TenantRequiredError,
        create_blue_team_rule,
        create_red_team_artifact,
        delete_blue_team_rule,
        delete_red_team_artifact,
        get_blue_team_rule,
        get_red_team_artifact,
        list_blue_team_rules,
        list_red_team_artifacts,
    )

try:
    from backend.app.audit import (
        AuditAccessDeniedError,
        AuditCorruptedRecordError,
        AuditEvent,
        AuditError,
        AuditVerificationError,
        AuditVerificationResult,
        append_audit_event,
        list_audit_events,
        verify_audit_chain,
    )
except ImportError:
    from .audit import (
        AuditAccessDeniedError,
        AuditCorruptedRecordError,
        AuditEvent,
        AuditError,
        AuditVerificationError,
        AuditVerificationResult,
        append_audit_event,
        list_audit_events,
        verify_audit_chain,
    )

logger = logging.getLogger(__name__)

# Application Metadata
APP_TITLE = "Bayora AI Safety Testing Infrastructure"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = (
    "Prototype AI safety testing infrastructure for Hack in Hills '26. "
    "Provides authenticated, tenant-isolated red-team artifact management, "
    "blue-team defense rules, and tamper-evident audit logging."
)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Optional CORS configuration (no wildcard allow_origins)
CORS_ORIGINS_ENV = os.environ.get("BAYORA_CORS_ALLOWED_ORIGINS")
if CORS_ORIGINS_ENV and CORS_ORIGINS_ENV.strip():
    allowed_origins = [o.strip() for o in CORS_ORIGINS_ENV.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
    )


# ============================================================================
# Pydantic Schemas for Requests and Responses
# ============================================================================

class HealthResponse(BaseModel):
    """Service status response model without sensitive environmental data."""
    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["bayora-backend"])
    version: str = Field(..., examples=["0.1.0"])


class WhoAmIResponse(BaseModel):
    """Authenticated caller introspection response model."""
    actor_id: str
    role: str
    tenant_id: Optional[str] = None
    permissions: list[str]


class CreateArtifactRequest(BaseModel):
    """Request schema for creating a red-team testing artifact."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128, description="Descriptive name")
    artifact_type: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Classification (e.g. prompt_injection, jailbreak_template)",
    )
    content: Any = Field(..., description="The testing payload or artifact data")
    artifact_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=64,
        description="Optional unique identifier (alphanumeric, dashes, underscores)",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Supplementary context metadata",
    )


class ArtifactResponse(BaseModel):
    """Response schema representing a stored red-team artifact."""
    artifact_id: str
    tenant_id: str
    created_by: str
    created_at: str
    updated_at: str
    name: str
    artifact_type: str
    content: Any
    metadata: dict[str, Any]


class CreateRuleRequest(BaseModel):
    """Request schema for creating a blue-team defense rule."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128, description="Descriptive rule name")
    rule_type: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Classification (e.g. input_guardrail, regex_filter)",
    )
    pattern_or_config: Any = Field(..., description="Filter pattern, regex, or policy configuration")
    rule_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=64,
        description="Optional unique identifier (alphanumeric, dashes, underscores)",
    )
    action: str = Field(default="block", min_length=1, max_length=32, description="Mitigation action")
    enabled: bool = Field(default=True, description="Whether the defense rule is active")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Supplementary context metadata")


class RuleResponse(BaseModel):
    """Response schema representing a stored blue-team defense rule."""
    rule_id: str
    tenant_id: str
    created_by: str
    created_at: str
    updated_at: str
    name: str
    rule_type: str
    pattern_or_config: Any
    action: str
    enabled: bool
    metadata: dict[str, Any]


class DeleteResponse(BaseModel):
    """Generic deletion confirmation response schema."""
    deleted: bool
    resource_id: str


class AuditEventResponse(BaseModel):
    """Response schema representing an individual audit event."""
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
    metadata: dict[str, Any]


class AuditVerifyResponse(BaseModel):
    """Response schema for audit ledger cryptographic verification."""
    is_valid: bool
    verified_count: int
    first_sequence: Optional[int] = None
    last_sequence: Optional[int] = None
    last_hash: Optional[str] = None
    error_message: Optional[str] = None
    failing_event_id: Optional[str] = None
    failing_sequence: Optional[int] = None


# ============================================================================
# Domain Exception Handlers
# ============================================================================

@app.exception_handler(RecordNotFoundError)
async def record_not_found_handler(request: Request, exc: RecordNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(RecordAlreadyExistsError)
async def record_already_exists_handler(request: Request, exc: RecordAlreadyExistsError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc)},
    )


@app.exception_handler(StorageAccessDeniedError)
async def storage_access_denied_handler(request: Request, exc: StorageAccessDeniedError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": "Access forbidden: insufficient role permissions for this resource."},
    )


@app.exception_handler(AuditAccessDeniedError)
async def audit_access_denied_handler(request: Request, exc: AuditAccessDeniedError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": "Access forbidden: caller not authorized to view requested audit records."},
    )


@app.exception_handler(TenantRequiredError)
async def tenant_required_handler(request: Request, exc: TenantRequiredError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": "Access forbidden: authenticated identity requires an assigned tenant scope."},
    )


@app.exception_handler(InvalidIdentifierError)
async def invalid_identifier_handler(request: Request, exc: InvalidIdentifierError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(PathTraversalError)
async def path_traversal_handler(request: Request, exc: PathTraversalError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid resource path: directory traversal attempt rejected."},
    )


@app.exception_handler(CorruptedDataError)
@app.exception_handler(AuditCorruptedRecordError)
async def corrupted_data_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Data corruption error: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal storage record corruption detected."},
    )


# ============================================================================
# Health & Identity Endpoints
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Returns service availability status without exposing internal configuration.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="bayora-backend",
        version=APP_VERSION,
    )


@app.get(
    "/api/v1/whoami",
    response_model=WhoAmIResponse,
    tags=["Authentication"],
    summary="Caller Identity Introspection",
    description="Inspect the authenticated caller identity, assigned role, and permissions.",
)
async def whoami(
    principal: Principal = Depends(get_current_principal),
) -> WhoAmIResponse:
    return WhoAmIResponse(
        actor_id=principal.actor_id,
        role=principal.role.value if hasattr(principal.role, "value") else str(principal.role),
        tenant_id=principal.tenant_id,
        permissions=[p.value if hasattr(p, "value") else str(p) for p in principal.permissions],
    )


# ============================================================================
# Red-Team Artifact APIs
# ============================================================================

@app.post(
    "/api/v1/red-team/artifacts",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Red Team"],
    summary="Create Red-Team Artifact",
    description="Create and persist a new red-team testing artifact under the authenticated tenant.",
    dependencies=[Depends(require_red_team)],
)
async def create_artifact_endpoint(
    payload: CreateArtifactRequest,
    principal: Principal = Depends(get_current_principal),
) -> ArtifactResponse:
    artifact = create_red_team_artifact(
        principal=principal,
        name=payload.name,
        artifact_type=payload.artifact_type,
        content=payload.content,
        artifact_id=payload.artifact_id,
        metadata=payload.metadata,
    )

    # Synchronous audit log recording (fail-closed if auditing fails)
    append_audit_event(
        action="artifact.create",
        outcome="success",
        principal=principal,
        resource_type="red_team_artifact",
        resource_id=artifact.artifact_id,
        metadata={"artifact_type": artifact.artifact_type, "name": artifact.name},
    )

    return ArtifactResponse(**artifact.to_dict())


@app.get(
    "/api/v1/red-team/artifacts",
    response_model=list[ArtifactResponse],
    tags=["Red Team"],
    summary="List Red-Team Artifacts",
    description="List all authorized red-team artifacts belonging strictly to the caller's tenant.",
    dependencies=[Depends(require_red_team)],
)
async def list_artifacts_endpoint(
    principal: Principal = Depends(get_current_principal),
) -> list[ArtifactResponse]:
    artifacts = list_red_team_artifacts(principal=principal)
    return [ArtifactResponse(**a.to_dict()) for a in artifacts]


@app.get(
    "/api/v1/red-team/artifacts/{artifact_id}",
    response_model=ArtifactResponse,
    tags=["Red Team"],
    summary="Get Red-Team Artifact",
    description="Retrieve a single red-team artifact by identifier within the caller's tenant.",
    dependencies=[Depends(require_red_team)],
)
async def get_artifact_endpoint(
    artifact_id: str,
    principal: Principal = Depends(get_current_principal),
) -> ArtifactResponse:
    artifact = get_red_team_artifact(principal=principal, artifact_id=artifact_id)
    return ArtifactResponse(**artifact.to_dict())


@app.delete(
    "/api/v1/red-team/artifacts/{artifact_id}",
    response_model=DeleteResponse,
    tags=["Red Team"],
    summary="Delete Red-Team Artifact",
    description="Delete a red-team artifact within the caller's tenant scope.",
    dependencies=[Depends(require_red_team)],
)
async def delete_artifact_endpoint(
    artifact_id: str,
    principal: Principal = Depends(get_current_principal),
) -> DeleteResponse:
    delete_red_team_artifact(principal=principal, artifact_id=artifact_id)

    # Synchronous audit recording
    append_audit_event(
        action="artifact.delete",
        outcome="success",
        principal=principal,
        resource_type="red_team_artifact",
        resource_id=artifact_id,
    )

    return DeleteResponse(deleted=True, resource_id=artifact_id)


# ============================================================================
# Blue-Team Defense Rule APIs
# ============================================================================

@app.post(
    "/api/v1/blue-team/rules",
    response_model=RuleResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Blue Team"],
    summary="Create Defense Rule",
    description="Create and persist a new blue-team defense rule under the authenticated tenant.",
    dependencies=[Depends(require_blue_team)],
)
async def create_rule_endpoint(
    payload: CreateRuleRequest,
    principal: Principal = Depends(get_current_principal),
) -> RuleResponse:
    rule = create_blue_team_rule(
        principal=principal,
        name=payload.name,
        rule_type=payload.rule_type,
        pattern_or_config=payload.pattern_or_config,
        rule_id=payload.rule_id,
        action=payload.action,
        enabled=payload.enabled,
        metadata=payload.metadata,
    )

    # Synchronous audit log recording
    append_audit_event(
        action="rule.create",
        outcome="success",
        principal=principal,
        resource_type="blue_team_rule",
        resource_id=rule.rule_id,
        metadata={"rule_type": rule.rule_type, "action": rule.action},
    )

    return RuleResponse(**rule.to_dict())


@app.get(
    "/api/v1/blue-team/rules",
    response_model=list[RuleResponse],
    tags=["Blue Team"],
    summary="List Defense Rules",
    description="List all authorized defense rules belonging strictly to the caller's tenant.",
    dependencies=[Depends(require_blue_team)],
)
async def list_rules_endpoint(
    principal: Principal = Depends(get_current_principal),
) -> list[RuleResponse]:
    rules = list_blue_team_rules(principal=principal)
    return [RuleResponse(**r.to_dict()) for r in rules]


@app.get(
    "/api/v1/blue-team/rules/{rule_id}",
    response_model=RuleResponse,
    tags=["Blue Team"],
    summary="Get Defense Rule",
    description="Retrieve a single defense rule by identifier within the caller's tenant.",
    dependencies=[Depends(require_blue_team)],
)
async def get_rule_endpoint(
    rule_id: str,
    principal: Principal = Depends(get_current_principal),
) -> RuleResponse:
    rule = get_blue_team_rule(principal=principal, rule_id=rule_id)
    return RuleResponse(**rule.to_dict())


@app.delete(
    "/api/v1/blue-team/rules/{rule_id}",
    response_model=DeleteResponse,
    tags=["Blue Team"],
    summary="Delete Defense Rule",
    description="Delete a defense rule within the caller's tenant scope.",
    dependencies=[Depends(require_blue_team)],
)
async def delete_rule_endpoint(
    rule_id: str,
    principal: Principal = Depends(get_current_principal),
) -> DeleteResponse:
    delete_blue_team_rule(principal=principal, rule_id=rule_id)

    # Synchronous audit recording
    append_audit_event(
        action="rule.delete",
        outcome="success",
        principal=principal,
        resource_type="blue_team_rule",
        resource_id=rule_id,
    )

    return DeleteResponse(deleted=True, resource_id=rule_id)


# ============================================================================
# Audit Log APIs
# ============================================================================

@app.get(
    "/api/v1/audit/events",
    response_model=list[AuditEventResponse],
    tags=["Audit"],
    summary="List Audit Events",
    description=(
        "Retrieve audit events scoped to caller authorization. Admins can view all "
        "events or filter by tenant. Tenant actors can only view their own tenant events."
    ),
)
async def list_audit_events_endpoint(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID (admins only)"),
    action: Optional[str] = Query(None, description="Filter by action name"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Max events to return"),
    principal: Principal = Depends(get_current_principal),
) -> list[AuditEventResponse]:
    events = list_audit_events(
        principal=principal,
        tenant_id=tenant_id,
        action=action,
        resource_type=resource_type,
        limit=limit,
    )
    return [AuditEventResponse(**e.to_dict()) for e in events]


@app.post(
    "/api/v1/audit/verify",
    response_model=AuditVerifyResponse,
    tags=["Audit"],
    summary="Verify Audit Hash Chain",
    description=(
        "Cryptographically verify the integrity, sequence numbers, and SHA-256 hash "
        "chain links of the audit ledger from the genesis block."
    ),
    dependencies=[Depends(require_admin)],
)
async def verify_audit_chain_endpoint(
    principal: Principal = Depends(get_current_principal),
) -> AuditVerifyResponse:
    result = verify_audit_chain()

    # Record the verification operation in the audit ledger
    append_audit_event(
        action="audit.verify",
        outcome="success" if result.is_valid else "failure",
        principal=principal,
        metadata={
            "is_valid": result.is_valid,
            "verified_count": result.verified_count,
            "failing_sequence": result.failing_sequence,
        },
    )

    return AuditVerifyResponse(
        is_valid=result.is_valid,
        verified_count=result.verified_count,
        first_sequence=result.first_sequence,
        last_sequence=result.last_sequence,
        last_hash=result.last_hash,
        error_message=result.error_message,
        failing_event_id=result.failing_event_id,
        failing_sequence=result.failing_sequence,
    )
