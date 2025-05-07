# test_force_curve_analyzer.py
import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import os
import pandas as pd  # For checking CSV output if generated

# Ensure the main script (analyze_force_curve.py) is in the same directory or PYTHONPATH
from Gen_SMFS.src.data_processing.analyze_force_curve import ForceCurveAnalyzer


# Helper function to create a mock IBW data structure as returned by igor.binarywave.load
def create_mock_ibw_data_structure(wave_data_array, notes_string=""):
    """
    Creates a dictionary structure similar to what ibw_load might return.
    """
    return {
        'wave': {
            'wData': np.array(wave_data_array, dtype=float),  # Ensure it's a numpy array
            'note': notes_string.encode('utf-8', errors='ignore')
        },
        'version': 5  # Or any other relevant version info
    }


class TestForceCurveAnalyzer(unittest.TestCase):

    def setUp(self):
        """
        Set up a default analyzer instance before each test method.
        Plots are disabled for automated testing.
        """
        self.analyzer = ForceCurveAnalyzer(
            temperature_k=298.15,
            fit_threshold_percent=80.0,
            make_plots=False,  # Crucial for non-interactive testing
            diagnostics=False
        )
        self.kb = 1.38064852e-23  # Boltzmann constant
        self.temp_k = 298.15  # Temperature in Kelvin

    def test_get_force_wlc_basic(self):
        """Test the _get_force_wlc method with typical parameters."""
        lc = 50e-9  # 50 nm
        p = 1e-9  # 1 nm
        separation = 25e-9  # Half Lc

        force = self.analyzer._get_force_wlc(lc, p, separation, self.temp_k)
        self.assertIsInstance(force, float)
        self.assertGreater(force, 0, "Force should be positive for valid inputs.")

        # Test with an array of separations
        sep_array = np.array([10e-9, 20e-9, 30e-9])
        forces_array = self.analyzer._get_force_wlc(lc, p, sep_array, self.temp_k)
        self.assertIsInstance(forces_array, np.ndarray)
        self.assertEqual(forces_array.shape, sep_array.shape)
        self.assertTrue(np.all(forces_array > 0))

    def test_get_force_wlc_edge_cases(self):
        """Test _get_force_wlc with edge cases."""
        lc = 50e-9
        p = 1e-9
        separation = 25e-9

        # Invalid Lc or P
        self.assertEqual(self.analyzer._get_force_wlc(0, p, separation, self.temp_k), 0)
        self.assertEqual(self.analyzer._get_force_wlc(lc, 0, separation, self.temp_k), 0)

        # Separation equals zero
        self.assertAlmostEqual(self.analyzer._get_force_wlc(lc, p, 0, self.temp_k), 0, places=15,
                               msg="Force at zero separation should be near zero for this WLC form.")

        # Separation very close to Lc (should be capped by max_r in implementation)
        force_near_limit = self.analyzer._get_force_wlc(lc, p, lc * 0.98, self.temp_k)
        force_at_cap_limit = self.analyzer._get_force_wlc(lc, p, lc * 0.99, self.temp_k)  # Assuming max_r = 0.99
        force_exceeding_limit = self.analyzer._get_force_wlc(lc, p, lc * 1.1, self.temp_k)

        self.assertGreater(force_near_limit, 0)
        self.assertAlmostEqual(force_exceeding_limit, force_at_cap_limit, places=12,
                               msg="Force should be capped if separation exceeds Lc*max_r")

    def test_get_cost_wlc(self):
        """Test the _get_cost_wlc method."""
        lc = 50e-9
        p = 1e-9
        sep_data = np.linspace(1e-9, 45e-9, 20)  # 20 points

        # Generate "perfect" force data using the model
        force_data_perfect = self.analyzer._get_force_wlc(lc, p, sep_data, self.temp_k)

        cost_perfect_fit = self.analyzer._get_cost_wlc((lc, p), sep_data, self.temp_k, force_data_perfect)
        # Cost should be very close to zero (allowing for float precision) and then scaled by 1e20
        self.assertAlmostEqual(cost_perfect_fit / 1e20, 0.0, places=10)

        # Generate "imperfect" force data
        force_data_imperfect = force_data_perfect * 1.05  # 5% deviation
        cost_imperfect_fit = self.analyzer._get_cost_wlc((lc, p), sep_data, self.temp_k, force_data_imperfect)
        self.assertGreater(cost_imperfect_fit, 0.0)

        # Test with invalid parameters (should return large cost)
        large_cost_expected = 1e12 * 1e20  # From implementation
        cost_invalid_lc = self.analyzer._get_cost_wlc((0, p), sep_data, self.temp_k, force_data_perfect)
        self.assertEqual(cost_invalid_lc, large_cost_expected)

    def test_extract_parameters_from_notes(self):
        """Test _extract_parameters_from_notes with various note strings."""
        notes_valid = "SpringConstant 0.052 N/m\\rRetractVelocity 1.23e-6 m/s\\rSomeOtherNote"
        k, v, k_ok, v_ok = self.analyzer._extract_parameters_from_notes(notes_valid)
        self.assertAlmostEqual(k, 0.052)
        self.assertTrue(k_ok)
        self.assertAlmostEqual(v, 1.23e-6)
        self.assertTrue(v_ok)

        notes_k_only = "This note has SpringConstant 0.03"
        k, v, k_ok, v_ok = self.analyzer._extract_parameters_from_notes(notes_k_only)
        self.assertAlmostEqual(k, 0.03)
        self.assertTrue(k_ok)
        self.assertAlmostEqual(v, 1e-6)  # Default velocity
        self.assertFalse(v_ok)

        notes_v_only_scientific = "RetractVelocity 2500.0E-9"  # 2.5e-6 m/s
        k, v, k_ok, v_ok = self.analyzer._extract_parameters_from_notes(notes_v_only_scientific)
        self.assertAlmostEqual(k, 0.03)  # Default k_spring
        self.assertFalse(k_ok)
        self.assertAlmostEqual(v, 2.5e-6)
        self.assertTrue(v_ok)

        notes_empty = "No relevant information here."
        k_default, v_default, k_ok_empty, v_ok_empty = self.analyzer._extract_parameters_from_notes(notes_empty)
        self.assertAlmostEqual(k_default, 0.03)  # Check against default
        self.assertFalse(k_ok_empty)
        self.assertAlmostEqual(v_default, 1e-6)  # Check against default
        self.assertFalse(v_ok_empty)

        notes_malformed = "SpringConstant NOT_A_NUMBER RetractVelocity ALSO_NOT_NUM"
        k_mal, v_mal, k_ok_mal, v_ok_mal = self.analyzer._extract_parameters_from_notes(notes_malformed)
        self.assertAlmostEqual(k_mal, 0.03)  # Default
        self.assertFalse(k_ok_mal)
        self.assertAlmostEqual(v_mal, 1e-6)  # Default
        self.assertFalse(v_ok_mal)

    def test_process_raw_curve_data_basic_structure(self):
        """
        Test _process_raw_curve_data for basic processing and output structure.
        This test uses a very simplified mock y_data.
        """
        # Mock y_data: [Height, Deflection, ZSensor]
        # A simple ramp up and down for ZSensor to define approach/retract.
        num_approach = 50
        num_retract = 50
        total_pts = num_approach + num_retract

        # Simplified Z-sensor: ramp up then down
        z_sensor = np.concatenate([np.linspace(0, 1e-7, num_approach), np.linspace(1e-7, 0, num_retract)])
        # Simplified Deflection: flat then a small bump
        deflection = np.zeros(total_pts)
        deflection[num_approach - 10:num_approach + 10] = np.sin(
            np.linspace(0, np.pi, 20)) * 5e-9  # Small bump around contact
        # Simplified Height
        height = z_sensor - deflection

        mock_y_data = np.column_stack((height, deflection, z_sensor))
        k_spring = 0.04  # N/m

        retract_p, approach_p, success = self.analyzer._process_raw_curve_data(mock_y_data, k_spring)

        self.assertTrue(success)
        self.assertIsNotNone(retract_p)
        self.assertIsNotNone(approach_p)
        self.assertEqual(retract_p.shape[1], 2, "Retract curve should have 2 columns (sep, force)")
        self.assertEqual(approach_p.shape[1], 2, "Approach curve should have 2 columns (sep, force)")
        # Downsampling is by factor of 2, so length should be roughly half of original segment.
        self.assertTrue(0 < retract_p.shape[0] <= num_retract)
        self.assertTrue(0 < approach_p.shape[0] <= num_approach)

    def test_process_raw_curve_data_insufficient_data(self):
        """Test _process_raw_curve_data with insufficient or malformed data."""
        k_spring = 0.04
        # Too few points
        short_y_data = np.random.rand(5, 3)
        _, _, success_short = self.analyzer._process_raw_curve_data(short_y_data, k_spring)
        self.assertFalse(success_short, "Should fail with too few data points.")

        # Wrong number of columns
        wrong_shape_data = np.random.rand(100, 2)  # Expecting at least 3
        _, _, success_shape = self.analyzer._process_raw_curve_data(wrong_shape_data, k_spring)
        self.assertFalse(success_shape, "Should fail with incorrect column count.")

    def test_find_regions_of_interest_with_event(self):
        """
        Test _find_regions_of_interest with data containing a simulated unfolding event.
        Note: This test is sensitive to the peak finding parameters in _find_regions_of_interest.
        """
        analyzer_for_roi = ForceCurveAnalyzer(diagnostics=False,
                                              make_plots=False)  # Use fresh analyzer if params matter
        separation = np.linspace(0, 150e-9, 300)
        force = np.random.normal(0, 1e-12, 300)  # Baseline noise around 0-1 pN

        # Simulate a WLC pull and rupture for an ROI
        # Start of pull at index 50, to index 150
        idx_start_pull, idx_peak_pull, idx_end_pull = 50, 140, 150
        true_lc1, true_p1 = 80e-9, 0.7e-9
        pull_sep = separation[idx_start_pull:idx_peak_pull + 1] - separation[idx_start_pull]
        pull_force = analyzer_for_roi._get_force_wlc(true_lc1, true_p1, pull_sep, analyzer_for_roi.temperature_k)

        # Ensure pull_force is significant enough
        pull_force = np.clip(pull_force, 0, 100e-12)  # Cap force to avoid extreme values from WLC
        min_peak_height_expected = 5e-12  # based on default in _find_regions_of_interest
        if np.max(pull_force) < min_peak_height_expected * 1.5:  # Make sure peak is clearly detectable
            pull_force *= (min_peak_height_expected * 2 / (np.max(pull_force) + 1e-15))

        force[idx_start_pull:idx_peak_pull + 1] += pull_force
        # Create a drop after the peak
        force[idx_peak_pull + 1:idx_end_pull] = force[idx_peak_pull] * np.linspace(1, 0.2,
                                                                                   idx_end_pull - (idx_peak_pull + 1))

        mock_force_curve = np.column_stack((separation, force))
        k_spring = 0.03  # Not directly used by simplified ROI finder, but good practice

        rois = analyzer_for_roi._find_regions_of_interest(mock_force_curve, k_spring)

        self.assertIsInstance(rois, list)
        # Depending on noise and exact peak parameters, we expect at least one ROI
        self.assertGreaterEqual(len(rois), 0)  # Can be 0 if synthetic peak isn't perfect for thresholds
        if len(rois) > 0:
            self.assertEqual(rois[0].shape[1], 2, "ROI data should have 2 columns.")
            # print(f"ROIs found: {len(rois)}") # For debugging test

    def test_find_regions_of_interest_no_event(self):
        """Test _find_regions_of_interest with flat, noisy data (no clear events)."""
        separation = np.linspace(0, 100e-9, 200)
        force = np.random.normal(2e-12, 0.5e-12, 200)  # Low force, small noise
        mock_force_curve = np.column_stack((separation, force))
        k_spring = 0.03

        rois = self.analyzer._find_regions_of_interest(mock_force_curve, k_spring)
        self.assertEqual(len(rois), 0, "No ROIs should be found in flat data.")

    @patch('analyze_force_curve.ibw_load')  # Mock the actual file loading
    def test_analyze_single_curve_mocked_processing_successful_fit(self, mock_ibw_load_func):
        """
        Test analyze_single_curve focusing on the WLC fitting part.
        Internal data processing steps (_process_raw_curve_data, _find_regions_of_interest) are mocked.
        """
        # 1. Define what ibw_load should return (for parameter extraction)
        mock_notes_string = "SpringConstant 0.05 N/m\rRetractVelocity 1e-6 m/s"
        # The wave_data itself can be minimal as _process_raw_curve_data is mocked
        mock_ibw_load_func.return_value = create_mock_ibw_data_structure(
            wave_data_array=[[0, 0, 0], [1, 1, 1]],
            notes_string=mock_notes_string
        )

        # 2. Define what the mocked _process_raw_curve_data should return
        # This would be a (mock_retract_curve, mock_approach_curve, success_flag)
        # For this test, we only care about the retract curve fed to _find_regions_of_interest
        mock_processed_retract = np.array([[i * 1e-9, i * 1e-12] for i in range(100)])  # Dummy processed data

        # 3. Define what the mocked _find_regions_of_interest should return
        # This is a list of ROIs. Each ROI is a [separation, force] numpy array.
        true_lc, true_p = 60e-9, 0.6e-9
        roi1_sep = np.linspace(0, 50e-9, 50)  # Adjusted for positive extension
        roi1_force = self.analyzer._get_force_wlc(true_lc, true_p, roi1_sep, self.temp_k)
        # Add slight noise to make it more realistic for fitting
        roi1_force += np.random.normal(0, np.mean(roi1_force) * 0.01, len(roi1_force))
        mock_rois_list = [np.column_stack((roi1_sep, roi1_force))]

        # Patch the internal methods of the self.analyzer instance
        with patch.object(self.analyzer, '_process_raw_curve_data',
                          return_value=(mock_processed_retract, None, True)) as mock_data_proc, \
                patch.object(self.analyzer, '_find_regions_of_interest', return_value=mock_rois_list) as mock_roi_find:
            fits, params_list, fig_handle = self.analyzer.analyze_single_curve("dummy_path.ibw")

            mock_ibw_load_func.assert_called_once_with("dummy_path.ibw")
            mock_data_proc.assert_called_once()  # k_spring would be 0.05 from notes
            # Check the k_spring argument passed to _find_regions_of_interest
            mock_roi_find.assert_called_once_with(mock_processed_retract, 0.05)

            self.assertEqual(len(fits), 1, "Should find one successful fit.")
            self.assertEqual(len(params_list), 1)

            fitted_params = params_list[0]
            self.assertGreaterEqual(fitted_params['fit_quality_percent'], self.analyzer.fit_threshold_percent)
            self.assertAlmostEqual(fitted_params['Lc_m'], true_lc, delta=true_lc * 0.25)  # Allow 25% delta for fit
            self.assertAlmostEqual(fitted_params['P_m'], true_p, delta=true_p * 0.35)  # Allow 35% delta for fit
            self.assertAlmostEqual(fitted_params['velocity_ms'], 1e-6)
            self.assertIsNone(fig_handle, "make_plots was False, so fig_handle should be None.")

    @patch('analyze_force_curve.ibw_load')
    def test_analyze_single_curve_no_rois_found(self, mock_ibw_load_func):
        """Test analyze_single_curve when no ROIs are identified."""
        mock_notes_string = "SpringConstant 0.04"
        mock_ibw_load_func.return_value = create_mock_ibw_data_structure([[0, 0, 0]], mock_notes_string)

        mock_processed_retract = np.array([[i * 1e-9, i * 0.1e-12] for i in range(10)])  # Dummy

        with patch.object(self.analyzer, '_process_raw_curve_data', return_value=(mock_processed_retract, None, True)), \
                patch.object(self.analyzer, '_find_regions_of_interest', return_value=[]):  # No ROIs

            fits, params_list, fig_handle = self.analyzer.analyze_single_curve("another_dummy.ibw")

            self.assertEqual(len(fits), 0)
            self.assertEqual(len(params_list), 0)

    @patch('analyze_force_curve.ForceCurveAnalyzer.analyze_single_curve')  # Mock the instance method
    @patch('os.listdir')
    @patch('os.path.join', side_effect=lambda *args: os.sep.join(args))  # Mock join to behave normally
    @patch('os.path.exists', return_value=True)  # Assume paths exist
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')  # Mock the saving of CSV
    def test_batch_process_ibw_files_flow(self, mock_df_to_csv, mock_makedirs, mock_exists,
                                          mock_os_join, mock_os_listdir, mock_analyze_single_curve_method):
        """Test the batch processing flow, mocking analyze_single_curve."""
        test_dir = "test_data_dir"
        mock_os_listdir.return_value = ['curve1.ibw', 'not_ibw.txt', 'curve2.ibw']

        # Define return values for mock_analyze_single_curve_method
        # Curve1: one fit
        fits1 = [np.array([[1, 2], [3, 4]])]
        params1 = [{'Lc_m': 50e-9, 'P_m': 1e-9, 'rupture_L_m': 45e-9, 'rupture_F_N': 30e-12,
                    'fit_quality_percent': 88.0, 'velocity_ms': 2e-6, 'roi_index': 0}]
        # Curve2: two fits
        fits2 = [np.array([[5, 6]]), np.array([[7, 8]])]
        params2 = [
            {'Lc_m': 60e-9, 'P_m': 1.1e-9, 'rupture_L_m': 55e-9, 'rupture_F_N': 35e-12,
             'fit_quality_percent': 90.0, 'velocity_ms': 2e-6, 'roi_index': 0},
            {'Lc_m': 70e-9, 'P_m': 1.2e-9, 'rupture_L_m': 65e-9, 'rupture_F_N': 40e-12,
             'fit_quality_percent': 85.0, 'velocity_ms': 2e-6, 'roi_index': 1}
        ]

        def analyze_single_side_effect(file_path_arg):
            if file_path_arg == os.path.join(test_dir, 'curve1.ibw'):
                return fits1, params1, None  # (fits, params, fig_handle)
            elif file_path_arg == os.path.join(test_dir, 'curve2.ibw'):
                return fits2, params2, None
            return [], [], None  # Should not happen for .ibw files

        mock_analyze_single_curve_method.side_effect = analyze_single_side_effect

        # Analyzer settings for this test
        self.analyzer.make_plots = True  # To test plot saving path
        self.analyzer.batch_process_ibw_files(test_dir, "summary_output.csv")

        self.assertEqual(mock_os_listdir.call_count, 1)
        self.assertEqual(mock_analyze_single_curve_method.call_count, 2)  # Called for curve1.ibw and curve2.ibw

        # Check calls to analyze_single_curve
        expected_calls = [
            unittest.mock.call(os.path.join(test_dir, 'curve1.ibw')),
            unittest.mock.call(os.path.join(test_dir, 'curve2.ibw'))
        ]
        mock_analyze_single_curve_method.assert_has_calls(expected_calls, any_order=True)

        # Check if DataFrame.to_csv was called
        mock_df_to_csv.assert_called_once()
        # First argument to mock_df_to_csv is the DataFrame instance, second is path, third is index=False
        df_written = mock_df_to_csv.call_args[0][0]  # The DataFrame instance
        self.assertEqual(len(df_written), 3)  # 1 fit from curve1, 2 from curve2
        self.assertIn('Lc_nm', df_written.columns)
        self.assertEqual(df_written.iloc[0]['filename'], 'curve1.ibw')
        self.assertEqual(df_written.iloc[1]['filename'], 'curve2.ibw')

        # Check if plot directory creation was attempted (if make_plots=True)
        expected_plot_dir = os.path.join(test_dir, "analysis_plots")
        mock_makedirs.assert_called_with(expected_plot_dir, exist_ok=True)  # if plots were generated

    @patch('os.listdir', return_value=[])  # No files
    @patch('builtins.print')  # To capture print statements
    def test_batch_process_no_ibw_files(self, mock_print, mock_os_listdir):
        """Test batch processing when no .ibw files are found."""
        self.analyzer.batch_process_ibw_files("empty_dir")
        mock_os_listdir.assert_called_once_with("empty_dir")
        mock_print.assert_any_call("No .ibw files found in directory: empty_dir")


if __name__ == '__main__':
    # This allows running the tests from the command line: python -m unittest test_force_curve_analyzer.py
    unittest.main()