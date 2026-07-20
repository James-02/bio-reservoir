"""Unit tests for optimization.reservoirs helpers."""

import pytest

from optimization.reservoirs import normalize_reservoir_type


class TestNormalizeReservoirType:
    @pytest.mark.parametrize(
        "value",
        ["bio", "biological", "genetic", "bioreservoir", "BioReservoir", " BIO "],
    )
    def test_bio_aliases(self, value):
        assert normalize_reservoir_type(value) == "bioreservoir"

    @pytest.mark.parametrize(
        "value",
        ["esn", "echo-state-network", "echo_state_network", "ESN"],
    )
    def test_esn_aliases(self, value):
        assert normalize_reservoir_type(value) == "esn"

    def test_none_defaults_to_bioreservoir(self):
        assert normalize_reservoir_type(None) == "bioreservoir"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_reservoir_type("quantum")
