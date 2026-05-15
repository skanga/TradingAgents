import pytest
from pydantic import ValidationError

from service.routers.portfolio import PositionUpdateRequest


@pytest.mark.parametrize(
    "payload",
    [
        {"shares": 0},
        {"shares": -1},
        {"cost_basis_per_share": 0},
        {"cost_basis_per_share": -1},
    ],
)
def test_position_update_rejects_non_positive_numeric_values(payload):
    with pytest.raises(ValidationError):
        PositionUpdateRequest(**payload)


def test_position_update_allows_positive_numeric_values_when_present():
    req = PositionUpdateRequest(shares=1, cost_basis_per_share=10)

    assert req.shares == 1
    assert req.cost_basis_per_share == 10


def test_position_update_allows_omitted_numeric_values():
    req = PositionUpdateRequest(account="brokerage", notes="unchanged size")

    assert req.shares is None
    assert req.cost_basis_per_share is None
