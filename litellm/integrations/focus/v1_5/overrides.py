"""Per-deployment FOCUS 1.5 column overrides, read from a deployment's ``model_info.focus`` block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from ..settings import FocusBillingSettings
from .providers import provider_entities


class FocusDeploymentOverrides(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    include_spend: bool | None = None
    billing_account_id: str | None = None
    billing_account_name: str | None = None
    sub_account_id: str | None = None
    sub_account_name: str | None = None
    region_id: str | None = None
    region_name: str | None = None
    service_provider_name: str | None = None
    host_provider_name: str | None = None
    service_name: str | None = None
    invoice_issuer_name: str | None = None

    @model_validator(mode="after")
    def _region_is_set_as_a_pair(self) -> Self:
        if (self.region_id is None) != (self.region_name is None):
            raise ValueError("model_info.focus region_id and region_name must be set together")
        return self


class ResolveDeploymentOverrides(Protocol):
    def __call__(self, model_id: str, /) -> FocusDeploymentOverrides | None: ...


@dataclass(frozen=True, slots=True)
class ResolvedColumns:
    """The FOCUS 1.5 column values that come from configuration rather than SpendLogs."""

    include_spend: bool
    billing_account_id: str
    billing_account_name: str
    sub_account_id: str
    sub_account_name: str
    region_id: str | None
    region_name: str | None
    service_provider_name: str
    host_provider_name: str
    service_name: str
    invoice_issuer_name: str


def resolve_columns(
    billing: FocusBillingSettings, overrides: FocusDeploymentOverrides | None, provider: str | None
) -> ResolvedColumns:
    """Apply deployment overrides over FOCUS_* env settings, deriving provider names from the provider slug."""
    chosen: Final = overrides or FocusDeploymentOverrides()
    billing_account_name: Final = chosen.billing_account_name or billing.billing_account_name
    entities: Final = provider_entities(provider, operator_name=billing_account_name)
    include_spend: Final = billing.include_spend if chosen.include_spend is None else chosen.include_spend
    service_provider_name: Final = chosen.service_provider_name or entities.service_provider_name
    region_overridden: Final = chosen.region_id is not None
    return ResolvedColumns(
        include_spend=include_spend,
        billing_account_id=chosen.billing_account_id or billing.billing_account_id,
        billing_account_name=billing_account_name,
        sub_account_id=chosen.sub_account_id or billing.sub_account_id,
        sub_account_name=chosen.sub_account_name or billing.sub_account_name,
        region_id=chosen.region_id if region_overridden else billing.region_id,
        region_name=chosen.region_name if region_overridden else billing.region_name,
        service_provider_name=service_provider_name,
        host_provider_name=chosen.host_provider_name or entities.host_provider_name,
        service_name=chosen.service_name or entities.service_name,
        invoice_issuer_name=chosen.invoice_issuer_name
        or (billing_account_name if include_spend else service_provider_name),
    )


class _DeploymentInfo(Protocol):
    @property
    def model_extra(self) -> dict[str, object] | None: ...


class _Deployment(Protocol):
    @property
    def model_info(self) -> _DeploymentInfo: ...


class SupportsGetDeployment(Protocol):
    def get_deployment(self, model_id: str) -> _Deployment | None: ...


def deployment_overrides(router: SupportsGetDeployment | None, model_id: str) -> FocusDeploymentOverrides | None:
    """Read and validate ``model_info.focus`` for a deployment the router knows about."""
    deployment: Final = router.get_deployment(model_id) if router is not None else None
    extra: Final = deployment.model_info.model_extra if deployment is not None else None
    raw: Final = extra.get("focus") if extra is not None else None
    if raw is None:
        return None
    try:
        return FocusDeploymentOverrides.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid model_info.focus on deployment {model_id}: {exc}") from exc


def proxy_deployment_overrides(model_id: str) -> FocusDeploymentOverrides | None:
    from litellm.proxy.proxy_server import llm_router

    return deployment_overrides(llm_router, model_id)


__all__ = (
    "FocusDeploymentOverrides",
    "ResolveDeploymentOverrides",
    "ResolvedColumns",
    "SupportsGetDeployment",
    "deployment_overrides",
    "proxy_deployment_overrides",
    "resolve_columns",
)
