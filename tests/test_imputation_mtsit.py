import unittest

import numpy as np

from imputegap.recovery.contamination import GenGap
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


class TestMTSIT(unittest.TestCase):

    def test_imputation_mtsit(self, name="mtsit"):
        """
        MTSIT through the standard ImputeGAP pipeline: only missing values are filled,
        and the imputation is clearly better than mean imputation on chlorine.
        """
        ts = TimeSeries()
        ts.load_series(utils.search_path("chlorine"), normalizer="z_score")
        incomp_data = GenGap.mcar(ts.data)

        algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
        algo.impute(params={"seq_len": 48, "epochs": 20, "batch_size": 64, "sliding_windows": 1, "d_model": 16, "n_heads": 4, "n_layers": 3, "d_ffn": 16, "masking_ratio": 0.15, "mean_mask_length": 3})
        algo.score(ts.data)
        ts.print_results(algo.metrics, algo.algorithm)

        missing = np.isnan(incomp_data)
        self.assertEqual(algo.recov_data.shape, incomp_data.shape)
        self.assertFalse(np.isnan(algo.recov_data).any())
        np.testing.assert_array_equal(algo.recov_data[~missing], incomp_data[~missing])

        baseline = Imputation.Statistics.MeanImpute(incomp_data)
        baseline.impute()
        baseline.score(ts.data)
        self.assertLess(algo.metrics["MAE"], baseline.metrics["MAE"], f"MTSIT MAE = {algo.metrics['MAE']}, mean imputation MAE = {baseline.metrics['MAE']}")

    def test_imputation_mtsit_default_parameters(self, name="mtsit"):
        """Default parameters are read from default_values.toml."""
        params = utils.load_parameters(query="default", algorithm=name)
        self.assertEqual(len(params), 10)


if __name__ == "__main__":
    unittest.main()
