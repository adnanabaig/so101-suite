"""Request bodies refuse NaN and Infinity in every float field.

StrictJSONMiddleware rejects those tokens at the HTTP layer; the models refuse
them too, for any caller that does not pass through it.
"""

from __future__ import annotations

import inspect
import math

import pytest
from pydantic import BaseModel, ValidationError

from so101.bringup import schemas as bringup_schemas
from so101.services import schemas
from so101.services.schemas import RequestModel

NON_FINITE = (math.nan, math.inf, -math.inf)


def _request_models(module) -> list[type[BaseModel]]:
    return [
        cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if name.endswith("Request")
        and issubclass(cls, BaseModel)
        and cls.__module__ == module.__name__
    ]


@pytest.mark.parametrize("model", _request_models(schemas), ids=lambda cls: cls.__name__)
def test_every_service_request_model_is_a_request_model(model):
    assert issubclass(model, RequestModel)


# AttestRequest is the exception: hil.validate_attestation refuses non-finite
# evidence itself, naming the field (tests/test_bringup_hil.py).
BRINGUP_FINITE_EXEMPT = {"AttestRequest"}


@pytest.mark.parametrize(
    "model",
    [m for m in _request_models(bringup_schemas) if m.__name__ not in BRINGUP_FINITE_EXEMPT],
    ids=lambda cls: cls.__name__,
)
def test_every_bringup_request_model_refuses_non_finite_floats(model):
    assert model.model_config.get("allow_inf_nan") is False


@pytest.mark.parametrize("value", NON_FINITE)
def test_command_targets_refuse_non_finite_values(value):
    with pytest.raises(ValidationError):
        schemas.ControlCommandRequest.model_validate(
            {
                "lease_id": "lease",
                "source": "ui",
                "sequence": 1,
                "sent_at": "2026-10-06T00:00:00Z",
                "targets": {"shoulder_pan": value},
            }
        )


@pytest.mark.parametrize("wrist", [[0.1, math.nan, None], {"x": math.inf, "z": None}])
def test_tracking_wrist_refuses_non_finite_values(wrist):
    with pytest.raises(ValidationError):
        schemas.TrackingPreviewRequest.model_validate({"confidence": 0.9, "wrist": wrist})


def test_tracking_wrist_still_accepts_null_axes():
    request = schemas.TrackingPreviewRequest.model_validate(
        {"confidence": 0.9, "wrist": {"x": 0.1, "y": 0.2, "z": None}}
    )
    assert request.wrist == {"x": 0.1, "y": 0.2, "z": None}


@pytest.mark.parametrize("value", NON_FINITE)
def test_nested_request_models_refuse_non_finite_values(value):
    with pytest.raises(ValidationError):
        schemas.AgentConstraints.model_validate({"max_duration_seconds": value})
    with pytest.raises(ValidationError):
        bringup_schemas.ProbeRequest.model_validate({"duration_s": value})
