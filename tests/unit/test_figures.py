"""training.figures: a run config's reservoir, built and driven as a run builds and drives it."""

import numpy as np
import pytest
from scipy import sparse

from training import figures


class TestConfig:
    def test_a_config_without_a_bio_reservoir_is_refused(self, small_design, tmp_path):
        config = tmp_path / "esn.yaml"
        config.write_text("name: toy\ntask: {rows: 100}\n"
                          "reservoir: {type: esn, params: {units: 10}}\n"
                          "readout: {classifier: RidgeCV}\n")
        with pytest.raises(ValueError, match="not a bio-reservoir config"):
            figures.run_config(config)
        assert figures.run_config(small_design("centre"))["reservoir"]["params"]["units"] == 16

    def test_a_beat_is_scaled_as_a_run_scales_it(self, small_design, ecg_data):
        cfg = figures.run_config(small_design("centre"))
        beat, name = figures.task_beat(cfg, 0)
        assert beat.shape == (187, 1) and name in figures.class_names(cfg["task"])
        assert np.mean(beat) == pytest.approx(0.0, abs=1e-9)  # sequence_zscore
        assert np.std(beat) == pytest.approx(1.0, abs=1e-6)

    def test_an_inducer_task_gives_a_trace_and_the_name_of_its_pattern_class(self,
                                                                            small_design):
        design = small_design("centre")
        config = design.with_name("inducer.yaml")
        config.write_text(f"base: {design.name}\n"
                          "task: {type: inducer, pattern: order, n_per_class: 5}\n")
        cfg = figures.run_config(config)
        beat, name = figures.task_beat(cfg, 0)
        assert beat.shape == (240, 1) and name == "A then B"

    def test_the_input_is_the_inducer_level_the_colonies_receive(self, small_design,
                                                                 ecg_data):
        cfg = figures.run_config(small_design("centre", noise_in=0.0))
        beat, _ = figures.task_beat(cfg, 0)
        level = figures.encoded_input(cfg, beat)
        node = figures.build(cfg, units=1, coupling="none", topology_type="lattice",
                             input_connectivity=1.0, input_sign="positive")
        node.initialize(beat[:1])
        node.diagnostics = True
        gain = cfg["reservoir"]["params"]["input_scaling"]
        for x, expected in zip(beat[:20], level[:20], strict=True):
            node(x.reshape(1, 1))
            assert node._last_inp_term[0, 0] == pytest.approx(gain * expected)
        assert level.min() >= 0.0
        signed = figures.run_config(small_design("abstract"))
        np.testing.assert_array_equal(figures.encoded_input(signed, beat), beat[:, 0])


class TestColony:
    def test_the_warm_up_ends_where_the_runs_start(self, small_design, ecg_data):
        cfg = figures.run_config(small_design("centre"))
        beat, _ = figures.task_beat(cfg, 0)
        warm, rest, driven = figures.colony_states(cfg, beat, 0, [0.1, 1.0])
        assert (warm.shape, rest.shape, list(driven)) == ((5, 4), (187, 4), [0.1, 1.0])
        alone = figures.build(cfg, units=1, coupling="none", topology_type="lattice",
                              param_heterogeneity_cv=0.0, input_connectivity=1.0,
                              input_sign="positive")
        np.testing.assert_array_equal(warm[-1], alone._warmup_states[-1][0])

    def test_a_larger_input_scaling_moves_the_colony_further(self, small_design, ecg_data):
        cfg = figures.run_config(small_design("centre"))
        beat, _ = figures.task_beat(cfg, 0)
        _, rest, driven = figures.colony_states(cfg, beat, 0, [0.01, 1.0])
        moved = {s: np.abs(states - rest).max() for s, states in driven.items()}
        assert 0 < moved[0.01] < moved[1.0]


class TestArray:
    def test_the_array_is_stepped_through_the_beat_as_a_run_runs_it(self, small_design,
                                                                     ecg_data):
        from training.simulate import beat_seed, reseed_noise

        cfg = figures.run_config(small_design("abstract"))
        beat, _ = figures.task_beat(cfg, 3)
        states, terms = figures.array_states(cfg, beat, 3)
        node = figures.build(cfg)
        reseed_noise(node, beat_seed(int(cfg["reservoir"]["seed"]), 3))
        np.testing.assert_array_equal(states.reshape(len(beat), -1),
                                      node.run(beat, reset=True))
        assert terms.shape == (187, 3) and np.all(terms[1:, 1] > 0)

    def test_without_the_abstract_coupling_there_is_no_recurrent_term(self, small_design,
                                                                      ecg_data):
        cfg = figures.run_config(small_design("centre"))
        beat, _ = figures.task_beat(cfg, 0)
        states, terms = figures.array_states(cfg, beat[:20], 0)
        assert states.shape == (20, 16, 4)
        assert np.all(terms[:, 1] == 0) and np.array_equal(terms[:, 0], terms[:, 2])


class TestCoupling:
    def test_a_lattice_is_a_grid_of_neighbours(self, small_design):
        positions, weights = figures.coupling(figures.run_config(small_design("centre")))
        np.testing.assert_allclose(positions[[0, 3, 4, 15]], [[0, 0], [1, 0], [0, 1 / 3],
                                                              [1, 1]])
        assert weights.nnz == 48  # the 24 edges of a 4 x 4 grid, both ways

    def test_every_link_of_a_lattice_that_is_not_square_joins_neighbours(self, small_design):
        positions, weights = figures.coupling(figures.run_config(small_design("centre",
                                                                              units=18)))
        lengths = np.linalg.norm(positions[weights.row] - positions[weights.col], axis=1)
        np.testing.assert_allclose(lengths, 0.5)  # 6 rows of 3 traps, 0.5 apart
        assert weights.nnz == 2 * (6 * 2 + 5 * 3)

    def test_a_random_topology_places_no_colonies(self, small_design):
        with pytest.raises(ValueError, match="places no colonies"):
            figures.coupling(figures.run_config(small_design("abstract", topology_type="random")))

    def test_a_kernel_places_its_colonies_and_no_coupling_has_no_weights(self, small_design):
        positions, weights = figures.coupling(figures.run_config(small_design("abstract")))
        assert positions.shape == (16, 2) and weights.shape == (16, 16)
        assert figures.coupling(figures.run_config(small_design("uncoupled")))[1] is None

    def test_the_raw_kernel_fades_to_alpha_at_the_interaction_distance(self, small_design):
        cfg = figures.run_config(small_design("abstract", units=200))
        distances, weights, nearest, reach = figures.raw_kernel(cfg)
        assert distances.shape == weights.shape == (200 * 199,)
        assert reach == pytest.approx(8.2 * nearest)
        np.testing.assert_allclose(weights, np.where(
            distances <= reach, cfg["reservoir"]["params"]["fade_alpha"] ** (
                (distances / reach) ** 2), 0.0), rtol=1e-6, atol=1e-12)

    def test_a_lattice_has_no_kernel(self, small_design):
        with pytest.raises(ValueError, match="Only a distance topology"):
            figures.raw_kernel(figures.run_config(small_design("centre")))

    def test_the_strongest_links_of_each_colony(self):
        weights = sparse.coo_matrix(np.array([[5.0, 1.0, -3.0, 2.0], [1.0, 0.0, 0.0, 0.0],
                                              [0.0, 4.0, 0.0, -6.0], [0.0, 0.0, 0.0, 0.0]]))
        keep = figures.strongest(weights, 2)
        links = zip(weights.row[keep], weights.col[keep], strict=True)
        assert sorted(links) == [(0, 2), (0, 3), (1, 0), (2, 1), (2, 3)]
