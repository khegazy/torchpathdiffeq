"""Unit tests for _get_sorted_indices and _insert_sorted_results."""

from __future__ import annotations

import torch
from _helpers import make_solver_for_unit_test


class TestGetSortedIndices:
    """Tests for _get_sorted_indices: compute merge positions via binary search."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_basic(self):
        """Interleaving [1,3,5] and [2,4] gives positions for [1,2,3,4,5]."""
        record = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        result = torch.tensor([2.0, 4.0], dtype=torch.float64)

        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        # Merged: [1, 2, 3, 4, 5]
        # record entries at positions 0, 2, 4
        # result entries at positions 1, 3
        assert torch.equal(idxs_keep, torch.tensor([0, 2, 4]))
        assert torch.equal(idxs_input, torch.tensor([1, 3]))

    def test_insert_at_start(self):
        """Inserting before all existing entries."""
        record = torch.tensor([5.0, 10.0], dtype=torch.float64)
        result = torch.tensor([1.0], dtype=torch.float64)

        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        # Merged: [1, 5, 10] -> result at 0, record at 1, 2
        assert torch.equal(idxs_input, torch.tensor([0]))
        assert torch.equal(idxs_keep, torch.tensor([1, 2]))

    def test_insert_at_end(self):
        """Inserting after all existing entries."""
        record = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = torch.tensor([10.0], dtype=torch.float64)

        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        # Merged: [1, 2, 10] -> record at 0, 1; result at 2
        assert torch.equal(idxs_input, torch.tensor([2]))
        assert torch.equal(idxs_keep, torch.tensor([0, 1]))

    def test_single_into_many(self):
        """Inserting one value into a larger record."""
        record = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
        result = torch.tensor([4.0], dtype=torch.float64)

        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        # Merged: [1, 3, 4, 5, 7] -> result at position 2
        assert torch.equal(idxs_input, torch.tensor([2]))
        assert len(idxs_keep) == 4

    def test_many_into_one(self):
        """Inserting multiple values around a single record entry."""
        record = torch.tensor([5.0], dtype=torch.float64)
        result = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64)

        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        # Merged: [1, 3, 5, 7] -> record at 2; result at 0, 1, 3
        assert torch.equal(idxs_keep, torch.tensor([2]))
        assert torch.equal(idxs_input, torch.tensor([0, 1, 3]))


class TestInsertSortedResults:
    """Tests for _insert_sorted_results: merge tensors at pre-computed positions."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_1d(self):
        """Merging 1D tensors produces correctly ordered output."""
        record = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        result = torch.tensor([2.0, 4.0], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        assert torch.allclose(merged, expected)

    def test_2d(self):
        """Merging 2D tensors [N, D] preserves correct ordering."""
        record = torch.tensor(
            [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=torch.float64
        )
        result = torch.tensor([[2.0, 20.0], [4.0, 40.0]], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._get_sorted_indices(
            record[:, 0], result[:, 0]
        )

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        assert merged.shape == (5, 2)
        expected_col0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        assert torch.allclose(merged[:, 0], expected_col0)

    def test_3d(self):
        """Merging 3D tensors [N, C, T] preserves correct ordering."""
        record = torch.tensor(
            [[[1.0, 1.5]], [[3.0, 3.5]], [[5.0, 5.5]]], dtype=torch.float64
        )  # [3, 1, 2]
        result = torch.tensor(
            [[[2.0, 2.5]], [[4.0, 4.5]]], dtype=torch.float64
        )  # [2, 1, 2]
        idxs_keep, idxs_input = self.solver._get_sorted_indices(
            record[:, 0, 0], result[:, 0, 0]
        )

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        assert merged.shape == (5, 1, 2)
        expected_first = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        assert torch.allclose(merged[:, 0, 0], expected_first)

    def test_values_preserved(self):
        """All original values from both record and result appear in merged output."""
        record = torch.tensor([10.0, 30.0, 50.0], dtype=torch.float64)
        result = torch.tensor([20.0, 40.0], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._get_sorted_indices(record, result)

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        for val in record:
            assert val in merged, f"Record value {val} missing from merged"
        for val in result:
            assert val in merged, f"Result value {val} missing from merged"
