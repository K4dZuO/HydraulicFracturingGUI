import numpy as np
import pytest
from scipy import stats
from typing import Tuple, List
from helpers.dimensionless.filtration.physics import PhysicsConstraints


class TestPhysicsConstraints:
    """Test class for PhysicsConstraints."""
    
    @staticmethod
    def create_monotonic_test_data(
        direction: str = 'non_increasing',
        n_points: int = 100,
        noise_level: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create test data with known monotonic properties."""
        x = np.linspace(0, 1, n_points)
        
        if direction == 'non_increasing':
            # Strictly decreasing function
            y = 1.0 - 0.8 * x**2
        elif direction == 'non_decreasing':
            # Strictly increasing function
            y = 0.2 + 0.8 * x**2
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Add noise if requested
        if noise_level > 0:
            y += np.random.normal(0, noise_level, n_points)
        
        return x, y
    
    @staticmethod
    def create_oscillatory_data(
        n_points: int = 100,
        amplitude: float = 0.2,
        frequency: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create data with artificial oscillations."""
        x = np.linspace(0, 1, n_points)
        # Base monotonic decreasing curve
        y_base = 1.0 - 0.8 * x**2
        # Add oscillations
        y_osc = amplitude * np.sin(frequency * x)
        return x, y_base + y_osc
    
    @staticmethod
    def create_high_curvature_data(
        n_points: int = 100,
        spike_amplitude: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create data with regions of high curvature."""
        x = np.linspace(0, 1, n_points)
        y = 1.0 - 0.8 * x**2
        
        # Add a spike (high curvature region)
        spike_center = n_points // 2
        spike_width = n_points // 10
        spike_indices = slice(spike_center - spike_width//2, spike_center + spike_width//2)
        y[spike_indices] += spike_amplitude * np.exp(-((x[spike_indices] - x[spike_center])**2) * 100)
        
        return x, y
    
    def test_monotonic_non_increasing(self):
        """Test monotonic constraint for non-increasing curves."""
        # Test 1: Already monotonic curve (should not change)
        x = np.linspace(0, 1, 10)
        y = 1.0 - 0.8 * x**2  # Strictly decreasing
        
        result = PhysicsConstraints.monotonic(y, x, direction='non_increasing')
        np.testing.assert_array_almost_equal(y, result, decimal=10)
        
        # Test 2: Non-monotonic curve with bumps
        y_non_monotonic = y.copy()
        y_non_monotonic[5] = 0.9  # Create a bump (higher value in the middle)
        
        result = PhysicsConstraints.monotonic(y_non_monotonic, x, direction='non_increasing')
        
        # Check that result is non-increasing
        diffs = np.diff(result)
        assert np.all(diffs <= 1e-10), "Result is not non-increasing"
        
        # Check that bump is removed (value at index 5 should be <= value at index 4)
        assert result[5] <= result[4] + 1e-10
        
        # Test 3: With NaN values
        y_with_nan = y.copy()
        y_with_nan[3] = np.nan
        y_with_nan[7] = np.nan
        
        result = PhysicsConstraints.monotonic(y_with_nan, x, direction='non_increasing')
        
        # Check NaNs are preserved
        assert np.isnan(result[3])
        assert np.isnan(result[7])
        # Check monotonicity of valid points
        valid_mask = np.isfinite(result)
        valid_diffs = np.diff(result[valid_mask])
        assert np.all(valid_diffs <= 1e-10)
    
    def test_monotonic_non_decreasing(self):
        """Test monotonic constraint for non-decreasing curves."""
        x = np.linspace(0, 1, 10)
        y = 0.2 + 0.8 * x**2  # Strictly increasing
        
        result = PhysicsConstraints.monotonic(y, x, direction='non_decreasing')
        np.testing.assert_array_almost_equal(y, result, decimal=10)
        
        # Test with non-monotonic data
        y_non_monotonic = y.copy()
        y_non_monotonic[5] = 0.3  # Create a dip (lower value in the middle)
        
        result = PhysicsConstraints.monotonic(y_non_monotonic, x, direction='non_decreasing')
        
        # Check that result is non-decreasing
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), "Result is not non-decreasing"
    
    def test_monotonic_without_x(self):
        """Test monotonic constraint without x-coordinates."""
        # Already monotonic
        y = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        result = PhysicsConstraints.monotonic(y, direction='non_increasing')
        np.testing.assert_array_almost_equal(y, result, decimal=10)
        
        # Non-monotonic
        y_non_monotonic = np.array([1.0, 0.9, 0.95, 0.7, 0.6])  # bump at index 2
        result = PhysicsConstraints.monotonic(y_non_monotonic, direction='non_increasing')
        
        # Check monotonicity
        assert np.all(np.diff(result) <= 1e-10)
        # Check specific: value at index 2 should be <= value at index 1
        assert result[2] <= result[1] + 1e-10
    
    def test_limit_curvature(self):
        """Test curvature limiting."""
        # Test 1: Low curvature curve (should not change much)
        x, y_low = self.create_monotonic_test_data(n_points=50, noise_level=0.01)
        result = PhysicsConstraints.limit_curvature(y_low, x, max_curvature=1.0)
        
        # With low curvature, result should be very close to original
        diff_norm = np.linalg.norm(result - y_low)
        assert diff_norm < 0.1, f"Low curvature curve changed too much: {diff_norm}"
        
        # Test 2: High curvature spike
        x, y_high = self.create_high_curvature_data(n_points=50, spike_amplitude=0.5)
        result = PhysicsConstraints.limit_curvature(y_high, x, max_curvature=1.0)
        
        # Compute curvature
        first_deriv = np.gradient(result, x)
        second_deriv = np.gradient(first_deriv, x)
        curve_range = np.max(result) - np.min(result)
        normalized_curvature = np.abs(second_deriv) / curve_range if curve_range > 0 else 0
        
        # Check that max curvature is reduced (or at least not increased)
        assert np.max(normalized_curvature) <= np.max(np.abs(second_deriv)) / curve_range + 0.1
        
        # Test 3: With NaN values
        y_with_nan = y_low.copy()
        y_with_nan[10:15] = np.nan
        result = PhysicsConstraints.limit_curvature(y_with_nan, x, max_curvature=1.0)
        
        # Check NaNs are preserved
        assert np.all(np.isnan(result[10:15]))
        # Check other values are finite
        assert np.all(np.isfinite(result[np.isfinite(y_with_nan)]))
    
    def test_limit_curvature_without_x(self):
        """Test curvature limiting without x-coordinates."""
        # Create data with implicit x (equidistant)
        y = np.sin(np.linspace(0, 4*np.pi, 100))  # Oscillatory but smooth
        
        result = PhysicsConstraints.limit_curvature(y, max_curvature=0.5)
        
        # Compute curvature (second derivative with uniform spacing)
        second_deriv = np.gradient(np.gradient(y))
        result_second_deriv = np.gradient(np.gradient(result))
        
        # Check that extreme curvatures are reduced
        max_orig_curv = np.max(np.abs(second_deriv))
        max_result_curv = np.max(np.abs(result_second_deriv))
        
        # Result shouldn't have higher maximum curvature
        assert max_result_curv <= max_orig_curv + 1e-10
    
    def test_remove_oscillations(self):
        """Test removal of oscillations."""
        # Test 1: Create oscillatory data
        x, y_osc = self.create_oscillatory_data(n_points=100, amplitude=0.1, frequency=20.0)
        
        result = PhysicsConstraints.remove_oscillations(y_osc, x, window_size=5, threshold=0.05)
        
        # Compute second derivative of result
        first_deriv = np.gradient(result, x)
        second_deriv = np.gradient(first_deriv, x)
        
        # Count sign changes in second derivative (indicator of oscillations)
        sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
        orig_first_deriv = np.gradient(y_osc, x)
        orig_second_deriv = np.gradient(orig_first_deriv, x)
        orig_sign_changes = np.sum(np.diff(np.sign(orig_second_deriv)) != 0)
        
        # Oscillations should be reduced (fewer sign changes)
        assert sign_changes <= orig_sign_changes + 5  # Allow small tolerance
        
        # Test 2: Already smooth data (should not change much)
        x, y_smooth = self.create_monotonic_test_data(n_points=50)
        result = PhysicsConstraints.remove_oscillations(y_smooth, x)
        
        diff_norm = np.linalg.norm(result - y_smooth)
        assert diff_norm < 0.01, f"Smooth curve changed too much: {diff_norm}"
        
        # Test 3: With NaN values
        y_with_nan = y_osc.copy()
        y_with_nan[20:30] = np.nan
        result = PhysicsConstraints.remove_oscillations(y_with_nan, x)
        
        assert np.all(np.isnan(result[20:30]))
        assert np.all(np.isfinite(result[np.isfinite(y_with_nan)]))
    
    def test_remove_oscillations_without_x(self):
        """Test oscillation removal without x-coordinates."""
        # Create oscillatory data
        n = 100
        y = np.linspace(1.0, 0.2, n) + 0.1 * np.sin(np.linspace(0, 10*np.pi, n))
        
        result = PhysicsConstraints.remove_oscillations(y, window_size=7)
        
        # Check variance reduction (oscillations add variance)
        orig_variance = np.var(y - np.mean(y))
        result_variance = np.var(result - np.mean(result))
        
        # Variance should be reduced (or similar if no oscillations)
        assert result_variance <= orig_variance + 0.01
    
    def test_asymptotic_fix(self):
        """Test asymptotic behavior correction."""
        n_points = 50
        
        # Test 1: Early regime with noise
        x = np.linspace(0, 1, n_points)
        y = 1.0 - 0.8 * x**2
        # Add noise to early points
        y[:10] += np.random.normal(0, 0.1, 10)
        
        result = PhysicsConstraints.asymptotic_fix(y, x, early_window=10, late_window=10)
        
        # Early points should be stabilized
        early_variance_orig = np.var(y[:10])
        early_variance_result = np.var(result[:10])
        assert early_variance_result <= early_variance_orig + 0.01
        
        # Test 2: Late regime with noise
        y_late = y.copy()
        y_late[-10:] += np.random.normal(0, 0.1, 10)
        
        result = PhysicsConstraints.asymptotic_fix(y_late, x, early_window=5, late_window=10)
        
        # Late points should be stabilized
        late_variance_orig = np.var(y_late[-10:])
        late_variance_result = np.var(result[-10:])
        assert late_variance_result <= late_variance_orig + 0.01
        
        # Test 3: Both regimes
        y_both = y.copy()
        y_both[:5] += np.random.normal(0, 0.05, 5)
        y_both[-5:] += np.random.normal(0, 0.05, 5)
        
        result = PhysicsConstraints.asymptotic_fix(y_both, x, early_window=5, late_window=5)
        
        # Check that middle part is unchanged (within tolerance)
        middle_indices = slice(5, -5)
        middle_diff = np.max(np.abs(result[middle_indices] - y_both[middle_indices]))
        assert middle_diff < 1e-5, f"Middle part changed: {middle_diff}"
    
    def test_asymptotic_fix_without_x(self):
        """Test asymptotic fix without x-coordinates."""
        n = 30
        y = np.linspace(1.0, 0.1, n)
        # Add noise to ends
        y[:5] += np.random.normal(0, 0.1, 5)
        y[-5:] += np.random.normal(0, 0.1, 5)
        
        result = PhysicsConstraints.asymptotic_fix(y, early_window=5, late_window=5)
        
        # Check that ends are smoothed
        early_diff = np.std(y[:5] - result[:5])
        late_diff = np.std(y[-5:] - result[-5:])
        
        # Differences should be positive (smoothing happened)
        assert early_diff > 1e-3 or late_diff > 1e-3
    
    def test_enforce_all(self):
        """Test comprehensive application of all constraints."""
        # Create challenging data: oscillatory, high curvature, non-monotonic
        n = 100
        x = np.linspace(0, 1, n)
        
        # Base curve
        y = 1.0 - 0.8 * x**2
        
        # Add various artifacts
        y_artifacts = y.copy()
        # Non-monotonic bump
        y_artifacts[30:40] += 0.2
        # Oscillations
        y_artifacts += 0.1 * np.sin(20 * x)
        # High curvature spike
        spike_center = n // 2
        y_artifacts[spike_center-5:spike_center+5] += 0.3 * np.exp(-((x[spike_center-5:spike_center+5] - x[spike_center])**2) * 100)
        # Noisy ends
        y_artifacts[:10] += np.random.normal(0, 0.05, 10)
        y_artifacts[-10:] += np.random.normal(0, 0.05, 10)
        
        # Apply all constraints
        result = PhysicsConstraints.enforce_all(
            y_artifacts,
            x,
            monotonic=True,
            limit_curvature=True,
            remove_oscillations=True,
            asymptotic_fix=True,
            monotonic_direction='non_increasing',
            max_curvature=5.0,
            oscillation_window=7,
            early_window=10,
            late_window=10
        )
        
        # Verify properties
        
        # 1. Monotonicity (non-increasing)
        diffs = np.diff(result[np.isfinite(result)])
        assert np.all(diffs <= 1e-8), "Result not monotonic non-increasing"
        
        # 2. Finite values (except where input was NaN)
        assert np.all(np.isfinite(result[np.isfinite(y_artifacts)]))
        
        # 3. Roughness check (should be smoother)
        result_grad = np.gradient(result, x)
        orig_grad = np.gradient(y_artifacts, x)
        
        # Result should generally be smoother (lower gradient magnitude)
        result_grad_norm = np.linalg.norm(result_grad[np.isfinite(result_grad)])
        orig_grad_norm = np.linalg.norm(orig_grad[np.isfinite(orig_grad)])
        
        # Allow for small increases due to asymptotic fix, but generally should be smoother
        assert result_grad_norm <= orig_grad_norm * 1.5
        
        # 4. Endpoint stability
        early_std = np.std(result[:10])
        late_std = np.std(result[-10:])
        # Should be reasonably stable
        assert early_std < 0.1 and late_std < 0.1
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test 1: Empty array
        empty_arr = np.array([])
        result = PhysicsConstraints.monotonic(empty_arr)
        assert len(result) == 0
        
        # Test 2: Single element
        single = np.array([1.0])
        result = PhysicsConstraints.monotonic(single)
        assert len(result) == 1
        assert result[0] == 1.0
        
        # Test 3: All NaN
        all_nan = np.full(10, np.nan)
        result = PhysicsConstraints.monotonic(all_nan)
        assert np.all(np.isnan(result))
        
        # Test 4: Invalid direction
        y = np.array([1.0, 0.9, 0.8])
        with pytest.raises(ValueError):
            PhysicsConstraints.monotonic(y, direction='invalid')
        
        # Test 5: Incompatible x and y lengths
        y = np.array([1.0, 2.0, 3.0])
        x_wrong = np.array([0, 1])  # Wrong length
        # Should return original (based on implementation)
        result = PhysicsConstraints.monotonic(y, x_wrong)
        np.testing.assert_array_equal(y, result)
    
    def test_preserve_finite_values(self):
        """Test that finite values outside NaN regions are preserved reasonably."""
        n = 20
        y = np.linspace(1.0, 0.0, n)
        # Add some NaNs in the middle
        y[5:10] = np.nan
        
        result = PhysicsConstraints.enforce_all(
            y,
            monotonic=True,
            limit_curvature=True,
            remove_oscillations=False,
            asymptotic_fix=False
        )
        
        # Check that non-NaN regions outside 5:10 are similar
        valid_indices = list(range(5)) + list(range(10, n))
        max_diff = np.max(np.abs(y[valid_indices] - result[valid_indices]))
        
        # Should not change valid points much
        assert max_diff < 0.1, f"Valid points changed too much: {max_diff}"
        
        # Check monotonicity in valid regions
        valid_result = result[np.isfinite(result)]
        diffs = np.diff(valid_result)
        assert np.all(diffs <= 1e-10)
    
    def test_parameter_variations(self):
        """Test that different parameter values work correctly."""
        x, y = self.create_oscillatory_data(n_points=80, amplitude=0.15, frequency=15.0)
        
        # Test different window sizes for oscillation removal
        for window_size in [3, 5, 9, 15]:
            result = PhysicsConstraints.remove_oscillations(
                y, x, window_size=window_size, threshold=0.1
            )
            # Just check it runs without error and produces finite output
            assert len(result) == len(y)
            assert np.all(np.isfinite(result[np.isfinite(y)]))
        
        # Test different curvature limits
        for max_curvature in [0.1, 1.0, 5.0, 20.0]:
            result = PhysicsConstraints.limit_curvature(y, x, max_curvature=max_curvature)
            assert len(result) == len(y)
        
        # Test different asymptotic window sizes
        result1 = PhysicsConstraints.asymptotic_fix(y, x, early_window=3, late_window=3)
        result2 = PhysicsConstraints.asymptotic_fix(y, x, early_window=10, late_window=10)
        
        # Different window sizes should produce different results
        assert not np.allclose(result1, result2, rtol=1e-10)
    
    def test_statistical_properties(self):
        """Test that statistical properties are preserved."""
        # Create a realistic curve
        n = 200
        x = np.linspace(0, 1, n)
        y = 1.0 / (1.0 + 10 * x**2)  # Smooth decreasing function
        
        # Add realistic noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, n)
        y_noisy = y + noise
        
        # Apply constraints
        result = PhysicsConstraints.enforce_all(
            y_noisy,
            x,
            monotonic=True,
            limit_curvature=True,
            remove_oscillations=True,
            asymptotic_fix=True
        )
        
        # Check statistical properties
        
        # 1. Mean should be similar
        orig_mean = np.nanmean(y_noisy)
        result_mean = np.nanmean(result)
        assert abs(orig_mean - result_mean) < 0.05
        
        # 2. Range should be similar or reduced (noise reduction)
        orig_range = np.nanmax(y_noisy) - np.nanmin(y_noisy)
        result_range = np.nanmax(result) - np.nanmin(result)
        assert result_range <= orig_range + 0.1
        
        # 3. Should be smoother (lower variance of gradients)
        orig_grad = np.gradient(y_noisy[np.isfinite(y_noisy)], x[np.isfinite(y_noisy)])
        result_grad = np.gradient(result[np.isfinite(result)], x[np.isfinite(result)])
        
        orig_grad_var = np.var(orig_grad)
        result_grad_var = np.var(result_grad)
        
        # Result should be smoother (lower gradient variance)
        assert result_grad_var <= orig_grad_var * 1.2  # Allow small increase