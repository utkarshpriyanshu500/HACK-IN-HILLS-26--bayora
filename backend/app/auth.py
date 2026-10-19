"""Bayora Authentication and Role-Based Authorization Module.

This module provides security-critical authentication and role-based access
control (RBAC) for Bayora, an AI safety testing infrastructure prototype developed
for Hack in Hills '26.

Architecture & Security Model:
------------------------------
1. Authentication:
   - POC mechanism: API-key authentication using the `X-API-Key` HTTP request header.
   - Credentials are loaded securely from environment variables. Usable credentials
     are NEVER hardcoded in source code or defaults.
   - Validation uses `secrets.compare_digest` for constant-time comparison to guard
     against timing side-channel attacks.
   - Fail-closed design: Missing, blank, malformed, or unconfigured credentials
     immediately reject the request with HTTP 401 Unauthorized.
   - Confidentiality: Supplied keys, configured secrets, and sensitive tokens are
     never exposed in error messages, HTTP response bodies, or server logs.

2. Authorization & Least Privilege:
   - Distinct roles: `red_team`, `blue_team`, `evaluator`, and `admin`.
   - Granular permission matrix:
     * `red_team`: Submit and manage private red-team artifacts.
     * `blue_team`: Manage private blue-team defense rules.
     * `evaluator`: Run authorized evaluations and view evaluation results.
     * `admin`: Manage system configuration and view audit logs.
   - Administrative boundaries: An administrator does NOT automatically inherit
     unrestricted access to private red-team or blue-team tenant artifacts.
   - Direct Object Reference (IDOR) prevention: Callers cannot claim an identity,
     role, or tenant in the request body/parameters. The server establishes identity
     exclusively from the validated credential.
   - Tenant isolation helper: `validate_tenant_access` validates that the caller's
     authenticated tenant boundary matches requested resources.

3. Environment Configuration:
   - `BAYORA_API_KEY_RED_TEAM`: Comma-separated or single API key for `red_team`.
   - `BAYORA_API_KEY_BLUE_TEAM`: Comma-separated or single API key for `blue_team`.
   - `BAYORA_API_KEY_EVALUATOR`: Comma-separated or single API key for `evaluator`.
   - `BAYORA_API_KEY_ADMIN`: Comma-separated or single API key for `admin`.
   - `BAYORA_API_KEYS_JSON`: Optional JSON string mapping API keys to principal metadata:
     {
       "<api-key>": {
         "role": "red_team",
         "actor_id": "red-team-agent-1",
         "tenant_id": "tenant-alpha"
       },
       "<another-api-key>": "admin"
     }

Notice:
-------
This API-key mechanism is intended for proof-of-concept AI safety testing and
hackathon infrastructure. It does not provide cryptographic non-repudiation,
mutual TLS client certificates, or production-grade multi-tenant identity federation.
Transport Layer Security (TLS) must always be enforced in production deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import secrets
from typing import Callable, Mapping, Optional, Sequence, Union

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

# HTTP Header constant
API_KEY_HEADER_NAME = "X-API-Key"

# Security scheme for OpenAPI documentation
api_key_header_scheme = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="Bayora API Key for role-based authentication and authorization",
)


class Role(str, Enum):
    """Supported actor roles within the Bayora platform."""

    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    EVALUATOR = "evaluator"
    ADMIN = "admin"


class Permission(str, Enum):
    """Granular permissions enforcing least privilege across platform operations."""

    # Red-team operations
    MANAGE_RED_TEAM_ARTIFACTS = "red_team:artifacts:manage"

    # Blue-team operations
    MANAGE_BLUE_TEAM_RULES = "blue_team:rules:manage"

    # Evaluator operations
    RUN_EVALUATIONS = "evaluator:evaluations:run"
    READ_EVALUATION_RESULTS = "evaluator:results:read"

    # Administrative operations
    MANAGE_ADMIN_CONFIG = "admin:config:manage"
    READ_AUDIT_LOGS = "admin:audit:read"


# Strict role-to-permission mapping enforcing least privilege.
# Note: Admin role deliberately does NOT grant access to private red_team artifacts
# or blue_team defense rules.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.RED_TEAM: frozenset({
        Permission.MANAGE_RED_TEAM_ARTIFACTS,
    }),
    Role.BLUE_TEAM: frozenset({
        Permission.MANAGE_BLUE_TEAM_RULES,
    }),
    Role.EVALUATOR: frozenset({
        Permission.RUN_EVALUATIONS,
        Permission.READ_EVALUATION_RESULTS,
    }),
    Role.ADMIN: frozenset({
        Permission.MANAGE_ADMIN_CONFIG,
        Permission.READ_AUDIT_LOGS,
    }),
}


class AuthConfigError(ValueError):
    """Raised when authentication configuration is missing, malformed, or invalid.

    Error messages are intentionally safe and do not expose sensitive credentials.
    """


@dataclass(frozen=True)
class Principal:
    """Authenticated security identity representing the caller.

    Attributes:
        actor_id: Unique server-assigned identifier for the authenticated actor.
        role: The validated role assigned to this actor.
        tenant_id: Optional tenant identifier for resource scoping.
        permissions: Granular permissions granted to this principal.
    """

    actor_id: str
    role: Role
    tenant_id: Optional[str] = None
    permissions: frozenset[Permission] = field(default_factory=frozenset)

    def has_role(self, role: Union[Role, str]) -> bool:
        """Check if the principal holds the specified role."""
        if isinstance(role, Role):
            return self.role == role
        try:
            return self.role == Role(role)
        except ValueError:
            return False

    def has_permission(self, permission: Union[Permission, str]) -> bool:
        """Check if the principal possesses the specified permission."""
        if isinstance(permission, Permission):
            return permission in self.permissions
        try:
            return Permission(permission) in self.permissions
        except ValueError:
            return False


def load_configured_credentials(
    env: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, Principal]]:
    """Load and validate credentials from environment variables.

    Reads role-specific environment variables and optional structured JSON mapping.
    Validates role names and rejects blank or malformed keys.

    Args:
        env: Optional mapping of environment variables (defaults to os.environ).

    Returns:
        List of tuples containing (raw_api_key, Principal).

    Raises:
        AuthConfigError: If configuration contains syntax errors or invalid roles.
    """
    env_source = os.environ if env is None else env
    credentials: list[tuple[str, Principal]] = []

    # 1. Parse individual role environment variables
    role_env_mapping = {
        "BAYORA_API_KEY_RED_TEAM": Role.RED_TEAM,
        "BAYORA_API_KEY_BLUE_TEAM": Role.BLUE_TEAM,
        "BAYORA_API_KEY_EVALUATOR": Role.EVALUATOR,
        "BAYORA_API_KEY_ADMIN": Role.ADMIN,
    }

    for env_var_name, role in role_env_mapping.items():
        raw_val = env_source.get(env_var_name)
        if not raw_val:
            continue

        # Support comma-separated keys for multiple team members
        for idx, key_item in enumerate(raw_val.split(",")):
            key = key_item.strip()
            if not key:
                continue

            actor_id = f"{role.value}-{idx + 1}"
            principal = Principal(
                actor_id=actor_id,
                role=role,
                tenant_id=None,
                permissions=ROLE_PERMISSIONS.get(role, frozenset()),
            )
            credentials.append((key, principal))

    # 2. Parse structured JSON environment variable if present
    json_config = env_source.get("BAYORA_API_KEYS_JSON")
    if json_config and json_config.strip():
        try:
            parsed_data = json.loads(json_config.strip())
        except Exception as err:
            raise AuthConfigError(
                "Malformed JSON in BAYORA_API_KEYS_JSON environment variable."
            ) from err

        if not isinstance(parsed_data, dict):
            raise AuthConfigError(
                "BAYORA_API_KEYS_JSON must be a JSON object mapping API keys to roles or metadata."
            )

        for key_candidate, metadata in parsed_data.items():
            clean_key = str(key_candidate).strip()
            if not clean_key:
                raise AuthConfigError("Encountered empty or blank API key in BAYORA_API_KEYS_JSON.")

            role_str: str
            actor_id: str
            tenant_id: Optional[str] = None

            if isinstance(metadata, str):
                role_str = metadata.strip()
                actor_id = f"{role_str}-actor"
            elif isinstance(metadata, dict):
                role_val = metadata.get("role")
                if not role_val or not isinstance(role_val, str):
                    raise AuthConfigError(
                        "Each entry in BAYORA_API_KEYS_JSON must specify a string 'role'."
                    )
                role_str = role_val.strip()
                actor_id = str(metadata.get("actor_id") or f"{role_str}-actor")
                raw_tenant = metadata.get("tenant_id")
                tenant_id = str(raw_tenant).strip() if raw_tenant else None
            else:
                raise AuthConfigError(
                    "Invalid metadata format in BAYORA_API_KEYS_JSON. Expected string or dictionary."
                )

            try:
                role_enum = Role(role_str)
            except ValueError as err:
                valid_roles = [r.value for r in Role]
                raise AuthConfigError(
                    f"Invalid role '{role_str}' in BAYORA_API_KEYS_JSON. Valid roles are: {valid_roles}"
                ) from err

            principal = Principal(
                actor_id=actor_id,
                role=role_enum,
                tenant_id=tenant_id,
                permissions=ROLE_PERMISSIONS.get(role_enum, frozenset()),
            )
            credentials.append((clean_key, principal))

    return credentials


def validate_tenant_access(principal: Principal, requested_tenant_id: str) -> None:
    """Validate that the authenticated principal is authorized for the requested tenant.

    Enforces tenant scoping and prevents Insecure Direct Object Reference (IDOR) attacks.
    If the principal is bound to a specific tenant_id, the requested_tenant_id must match.

    Args:
        principal: The authenticated caller identity.
        requested_tenant_id: The tenant identifier from the resource or operation.

    Raises:
        HTTPException(400): If the requested tenant identifier is empty or whitespace.
        HTTPException(403): If the principal is not authorized for the requested tenant.
    """
    if not requested_tenant_id or not requested_tenant_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant identifier must be non-empty.",
        )

    clean_tenant = requested_tenant_id.strip()

    if principal.tenant_id is not None and principal.tenant_id != clean_tenant:
        logger.warning(
            "Tenant access denied for actor '%s': principal tenant '%s' != requested '%s'",
            principal.actor_id,
            principal.tenant_id,
            clean_tenant,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: caller not authorized for requested tenant.",
        )


async def get_current_principal(
    api_key: Optional[str] = Security(api_key_header_scheme),
) -> Principal:
    """FastAPI dependency to authenticate the incoming request via X-API-Key header.

    Validates the supplied credential using constant-time comparison against
    configured credentials. Fails closed if credentials are missing, malformed,
    or unconfigured.

    Args:
        api_key: The value of the X-API-Key header, injected by FastAPI.

    Returns:
        The authenticated Principal identity.

    Raises:
        HTTPException(401): If the credential is missing, blank, or invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key in 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    clean_key = api_key.strip()
    if not clean_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    try:
        configured_credentials = load_configured_credentials()
    except AuthConfigError as err:
        logger.error("Authentication configuration error: %s", str(err))
        # Fail closed on configuration error without revealing internal details
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service misconfigured or unavailable.",
            headers={"WWW-Authenticate": "ApiKey"},
        ) from err

    if not configured_credentials:
        logger.warning(
            "Authentication rejected: No API credentials configured on server (failing closed)."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unconfigured credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    matched_principal: Optional[Principal] = None

    # Perform constant-time comparison across all configured keys to prevent
    # character-level and key-position timing attacks.
    for configured_key, principal in configured_credentials:
        if secrets.compare_digest(configured_key, clean_key):
            matched_principal = principal

    if matched_principal is None:
        logger.warning("Authentication failed: Unrecognized or invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unrecognized API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return matched_principal


def require_roles(*allowed_roles: Union[Role, str]) -> Callable[[Principal], Principal]:
    """Dependency factory enforcing role-based access control (RBAC).

    Usage:
        @app.post("/artifacts", dependencies=[Depends(require_roles(Role.RED_TEAM))])
        def create_artifact(principal: Principal = Depends(require_roles(Role.RED_TEAM))):
            ...

    Args:
        *allowed_roles: One or more Role enum members or string role identifiers.

    Returns:
        A FastAPI dependency function that validates the caller's role and returns the Principal.

    Raises:
        ValueError: If no roles are provided or an unrecognized role string is passed.
    """
    target_roles: set[Role] = set()
    for role_item in allowed_roles:
        if isinstance(role_item, Role):
            target_roles.add(role_item)
        elif isinstance(role_item, str):
            try:
                target_roles.add(Role(role_item.strip()))
            except ValueError as err:
                valid = [r.value for r in Role]
                raise ValueError(
                    f"Unknown role '{role_item}'. Valid roles are: {valid}"
                ) from err
        else:
            raise TypeError(
                f"Role must be a Role enum or string, received: {type(role_item).__name__}"
            )

    if not target_roles:
        raise ValueError("require_roles requires at least one permitted role.")

    async def _role_dependency(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        if principal.role not in target_roles:
            logger.warning(
                "Access forbidden: actor '%s' with role '%s' lacks required role from %s",
                principal.actor_id,
                principal.role.value,
                [r.value for r in target_roles],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: insufficient role permissions.",
            )
        return principal

    return _role_dependency


def require_permissions(
    *required_permissions: Union[Permission, str],
) -> Callable[[Principal], Principal]:
    """Dependency factory enforcing granular permission-based access control.

    Args:
        *required_permissions: One or more Permission enum members or strings.

    Returns:
        A FastAPI dependency function that validates the caller's permissions and returns the Principal.

    Raises:
        ValueError: If no permissions are provided or an unrecognized permission is passed.
    """
    target_perms: set[Permission] = set()
    for perm_item in required_permissions:
        if isinstance(perm_item, Permission):
            target_perms.add(perm_item)
        elif isinstance(perm_item, str):
            try:
                target_perms.add(Permission(perm_item.strip()))
            except ValueError as err:
                raise ValueError(f"Unknown permission '{perm_item}'.") from err
        else:
            raise TypeError(
                f"Permission must be a Permission enum or str, received: {type(perm_item).__name__}"
            )

    if not target_perms:
        raise ValueError("require_permissions requires at least one required permission.")

    async def _permission_dependency(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        missing_perms = target_perms - principal.permissions
        if missing_perms:
            logger.warning(
                "Access forbidden: actor '%s' lacks required permissions %s",
                principal.actor_id,
                [p.value for p in missing_perms],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: insufficient permissions.",
            )
        return principal

    return _permission_dependency


# Pre-configured convenience dependencies for standard roles
require_red_team = require_roles(Role.RED_TEAM)
require_blue_team = require_roles(Role.BLUE_TEAM)
require_evaluator = require_roles(Role.EVALUATOR)
require_admin = require_roles(Role.ADMIN)
