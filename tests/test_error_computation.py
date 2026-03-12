"""Unit tests for error norm, _rec_remove, and error ratio computation."""
from __future__ import annotations

import pytest
import torch

from _helpers import make_solver_for_unit_test


# ---------------------------------------------------------------------------
# _error_norm
# ---------------------------------------------------------------------------

class TestErrorNorm:
    """Tests for _error_norm: RMS norm sqrt(mean(error², dim=-1))."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_1d_errors(self):
        """Single-dimension errors: RMS reduces to abs."""
        error = torch.tensor([[3.0], [-4.0]])  # [2, 1]
        result = self.solver._error_norm(error)
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_multidim(self):
        """Multi-dimension errors: RMS across D=2."""
        error = torch.tensor([[3.0, 4.0]])  # [1, 2]
        result = self.solver._error_norm(error)
        expected = torch.tensor([torch.sqrt(torch.tensor(12.5))])
        assert torch.allclose(result, expected)

    def test_zero(self):
        """Zero errors give zero norms."""
        error = torch.zeros(5, 3)
        result = self.solver._error_norm(error)
        assert torch.allclose(result, torch.zeros(5))

    def test_single_element(self):
        """Single element: RMS of one value is its abs."""
        error = torch.tensor([[7.0]])
        result = self.solver._error_norm(error)
        assert torch.allclose(result, torch.tensor([7.0]))

    def test_negative_values(self):
        """Negative values are squared, so result is always non-negative."""
        error = torch.tensor([[-3.0, -4.0]])
        result = self.solver._error_norm(error)
        # Same as [3, 4] case
        expected = torch.tensor([torch.sqrt(torch.tensor(12.5))])
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# _rec_remove
# ---------------------------------------------------------------------------

class TestRecRemove:
    """Tests for _rec_remove: ensure no adjacent True values in a boolean mask."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def _check_no_adjacent(self, mask):
        """Helper: verify no two adjacent True values."""
        if len(mask) < 2:
            return
        adjacent = mask[:-1] & mask[1:]
        assert not torch.any(adjacent), f"Adjacent Trues found in {mask}"

    def test_no_adjacent_trues(self):
        """Already valid mask is unchanged."""
        mask = torch.tensor([True, False, True, False, True])
        result = self.solver._rec_remove(mask.clone())
        assert torch.equal(result, mask)

    def test_pair_at_start(self):
        """Adjacent pair at start: second is removed."""
        mask = torch.tensor([True, True, False, True])
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        assert result[0] == True  # First kept

    def test_all_true(self):
        """All True: result alternates True/False."""
        mask = torch.ones(5, dtype=torch.bool)
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        # At least ceil(N/2) Trues remain
        assert result.sum() >= 3

    def test_all_false(self):
        """All False: unchanged."""
        mask = torch.zeros(3, dtype=torch.bool)
        result = self.solver._rec_remove(mask.clone())
        assert torch.equal(result, mask)

    def test_single_element(self):
        """Single True element is unchanged."""
        mask = torch.tensor([True])
        result = self.solver._rec_remove(mask.clone())
        assert result[0] == True

    def test_two_both_true(self):
        """Two adjacent Trues: second is removed."""
        mask = torch.tensor([True, True])
        result = self.solver._rec_remove(mask.clone())
        expected = torch.tensor([True, False])
        assert torch.equal(result, expected)

    def test_three_adjacent(self):
        """Three adjacent Trues: result is [T, F, T]."""
        mask = torch.tensor([True, True, True])
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        assert result[0] == True
        assert result[2] == True

    def test_long_alternating(self):
        """Already alternating mask of length 20 is unchanged."""
        mask = torch.tensor([i % 2 == 0 for i in range(20)])
        result = self.solver._rec_remove(mask.clone())
        assert torch.equal(result, mask)

    def test_long_all_true(self):
        """All True of length 20: result has no adjacent Trues."""
        mask = torch.ones(20, dtype=torch.bool)
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        assert result.sum() >= 10  # At least half remain


# ---------------------------------------------------------------------------
# _compute_error_ratios_absolute
# ---------------------------------------------------------------------------

class TestComputeErrorRatiosAbsolute:
    """Tests for absolute error ratio computation."""

    def _make_solver(self, atol=1e-3, rtol=1e-3):
        return make_solver_for_unit_test(atol=atol, rtol=rtol)

    def test_basic(self):
        """Error ratios correctly identify passing and failing steps."""
        solver = self._make_solver(atol=1e-3, rtol=1e-3)
        # error_tol = atol + rtol * |integral| = 1e-3 + 1e-3 * 1.0 = 2e-3
        sum_step_errors = torch.tensor([[0.01], [0.001]])  # [2, 1]
        integral = torch.tensor([1.0])

        error_ratio, error_ratio_2steps = solver._compute_error_ratios_absolute(
            sum_step_errors, integral
        )

        # 0.01 / 2e-3 = 5.0 (failing), 0.001 / 2e-3 = 0.5 (passing)
        assert error_ratio[0] > 1.0
        assert error_ratio[1] < 1.0

    def test_zero_integral(self):
        """Zero integral: error_tol = atol only, no NaN."""
        solver = self._make_solver(atol=1e-3, rtol=1e-3)
        sum_step_errors = torch.tensor([[1e-4]])
        integral = torch.tensor([0.0])

        error_ratio, _ = solver._compute_error_ratios_absolute(
            sum_step_errors, integral
        )

        assert torch.isfinite(error_ratio).all()
        # 1e-4 / 1e-3 = 0.1
        assert torch.allclose(error_ratio, torch.tensor([0.1]))

    def test_2steps_shape(self):
        """N=3 steps: error_ratio has len 3, error_ratio_2steps has len 2."""
        solver = self._make_solver()
        sum_step_errors = torch.tensor([[0.001], [0.002], [0.003]])
        integral = torch.tensor([1.0])

        error_ratio, error_ratio_2steps = solver._compute_error_ratios_absolute(
            sum_step_errors, integral
        )

        assert len(error_ratio) == 3
        assert len(error_ratio_2steps) == 2

    def test_single_step(self):
        """N=1: error_ratio has len 1, error_ratio_2steps has len 0."""
        solver = self._make_solver()
        sum_step_errors = torch.tensor([[0.001]])
        integral = torch.tensor([1.0])

        error_ratio, error_ratio_2steps = solver._compute_error_ratios_absolute(
            sum_step_errors, integral
        )

        assert len(error_ratio) == 1
        assert len(error_ratio_2steps) == 0

    def test_multidim(self):
        """D=2: error is reduced to scalar per step via _error_norm."""
        solver = self._make_solver()
        sum_step_errors = torch.tensor([[0.001, 0.002], [0.003, 0.004]])  # [2, 2]
        integral = torch.tensor([1.0, 1.0])

        error_ratio, error_ratio_2steps = solver._compute_error_ratios_absolute(
            sum_step_errors, integral
        )

        assert error_ratio.shape == (2,)
        assert error_ratio_2steps.shape == (1,)


# ---------------------------------------------------------------------------
# _compute_error_ratios_cumulative
# ---------------------------------------------------------------------------

class TestComputeErrorRatiosCumulative:
    """Tests for cumulative error ratio computation."""

    def _make_solver(self, atol=1e-3, rtol=1e-3):
        return make_solver_for_unit_test(atol=atol, rtol=rtol)

    def test_basic_with_sum_steps(self):
        """Cumulative error ratios computed from sum_steps."""
        solver = self._make_solver()
        sum_step_errors = torch.tensor([[0.001], [0.002]])
        sum_steps = torch.tensor([[1.0], [2.0]])

        error_ratio, error_ratio_2steps = solver._compute_error_ratios_cumulative(
            sum_step_errors, sum_steps=sum_steps
        )

        assert len(error_ratio) == 2
        assert len(error_ratio_2steps) == 1
        assert torch.isfinite(error_ratio).all()

    def test_with_cum_sum_steps(self):
        """Passing cum_sum_steps directly gives the same result."""
        solver = self._make_solver()
        sum_step_errors = torch.tensor([[0.001], [0.002]])
        sum_steps = torch.tensor([[1.0], [2.0]])
        cum_sum_steps = torch.cumsum(sum_steps, dim=0)

        r1, r2_1 = solver._compute_error_ratios_cumulative(
            sum_step_errors, sum_steps=sum_steps
        )
        r3, r2_2 = solver._compute_error_ratios_cumulative(
            sum_step_errors, cum_sum_steps=cum_sum_steps
        )

        assert torch.allclose(r1, r3)
        assert torch.allclose(r2_1, r2_2)

    def test_missing_args_raises(self):
        """Neither sum_steps nor cum_sum_steps raises ValueError."""
        solver = self._make_solver()
        sum_step_errors = torch.tensor([[0.001]])

        with pytest.raises(ValueError, match="Must give"):
            solver._compute_error_ratios_cumulative(sum_step_errors)
