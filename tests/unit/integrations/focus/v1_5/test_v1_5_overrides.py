from __future__ import annotations

from dataclasses import dataclass

import pydantic
import pytest

from litellm import Router

from litellm.integrations.focus.settings import FocusBillingSettings
from litellm.integrations.focus.v1_5.overrides import (
    FocusDeploymentOverrides,
    ResolvedColumns,
    deployment_overrides,
    resolve_columns,
)

BILLING = FocusBillingSettings(
    include_spend=True,
    billing_account_id="billing-1",
    billing_account_name="Example AI Platform",
    sub_account_id="account-1",
    sub_account_name="Example Account",
    region_id="global",
    region_name="Global",
)


@dataclass(frozen=True)
class _Info:
    model_extra: dict[str, object] | None


@dataclass(frozen=True)
class _Deployment:
    model_info: _Info


@dataclass(frozen=True)
class _Router:
    deployments: dict[str, _Deployment]

    def get_deployment(self, model_id: str) -> _Deployment | None:
        return self.deployments.get(model_id)


def test_env_settings_and_provider_names_apply_without_overrides() -> None:
    assert resolve_columns(BILLING, None, "openai") == ResolvedColumns(
        include_spend=True,
        billing_account_id="billing-1",
        billing_account_name="Example AI Platform",
        sub_account_id="account-1",
        sub_account_name="Example Account",
        region_id="global",
        region_name="Global",
        service_provider_name="OpenAI",
        host_provider_name="OpenAI",
        service_name="OpenAI API",
        invoice_issuer_name="Example AI Platform",
    )


def test_region_override_replaces_both_region_columns() -> None:
    resolved = resolve_columns(BILLING, FocusDeploymentOverrides(region_id="us", region_name="US"), "openai")

    assert (resolved.region_id, resolved.region_name) == ("us", "US")


def test_observer_mode_names_the_service_provider_as_invoice_issuer() -> None:
    resolved = resolve_columns(BILLING, FocusDeploymentOverrides(include_spend=False), "anthropic")

    assert (resolved.include_spend, resolved.invoice_issuer_name) == (False, "Anthropic")


def test_unknown_override_keys_are_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="regoin_id"):
        FocusDeploymentOverrides.model_validate({"regoin_id": "us"})


@pytest.mark.parametrize("raw", ({"region_id": "us"}, {"region_name": "US"}))
def test_region_overrides_must_be_paired(raw: dict[str, str]) -> None:
    with pytest.raises(pydantic.ValidationError, match="region_id and region_name must be set together"):
        FocusDeploymentOverrides.model_validate(raw)


def test_router_deployment_focus_block_is_validated() -> None:
    router = _Router({"d1": _Deployment(_Info({"focus": {"region_id": "us", "region_name": "US"}}))})

    assert deployment_overrides(router, "d1") == FocusDeploymentOverrides(region_id="us", region_name="US")


@pytest.mark.parametrize(
    ("router", "model_id"),
    (
        (None, "d1"),
        (_Router({}), "d1"),
        (_Router({"d1": _Deployment(_Info(None))}), "d1"),
        (_Router({"d1": _Deployment(_Info({"other": 1}))}), "d1"),
    ),
)
def test_missing_router_deployment_or_focus_block_means_no_overrides(router: _Router | None, model_id: str) -> None:
    assert deployment_overrides(router, model_id) is None


def test_invalid_focus_block_names_the_deployment() -> None:
    router = _Router({"d1": _Deployment(_Info({"focus": {"include_spend": "sometimes"}}))})

    with pytest.raises(ValueError, match="deployment d1"):
        deployment_overrides(router, "d1")


def test_litellm_router_keeps_the_focus_block_from_model_info() -> None:
    router = Router(
        model_list=[
            {
                "model_name": "spark",
                "litellm_params": {"model": "hosted_vllm/spark", "api_base": "http://localhost:1"},
                "model_info": {"id": "spark-1", "focus": {"region_id": "on-prem", "region_name": "Home Lab"}},
            }
        ]
    )

    assert deployment_overrides(router, "spark-1") == FocusDeploymentOverrides(
        region_id="on-prem", region_name="Home Lab"
    )
