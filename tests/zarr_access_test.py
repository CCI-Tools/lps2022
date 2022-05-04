import unittest

import numpy as np
import xarray as xr

from xcube.core.store import new_data_store


class ZarrAccessTest(unittest.TestCase):

    def test_that_xr_has_an_indexing_problem(self):
        x32 = xr.DataArray(np.linspace(0, 1, 100, dtype=np.float32), dims='x')
        x64 = xr.DataArray(np.linspace(0, 1, 100, dtype=np.float64), dims='x')

        x3 = x32.where(x64 > 0.5)
        self.assertEqual(len(x32), len(x3))

        v32 = xr.DataArray(np.random.random(100), dims='x', coords=dict(x=x32))
        v64 = xr.DataArray(np.random.random(100), dims='x', coords=dict(x=x64))

        v3 = v32.where(v64 > 0.5)
        self.assertEqual(len(v32), len(v3))
        # --> Assertion error, Expected :100, Actual :2

    def test_it(self):
        store = new_data_store('ccizarr')
        sm_dataset = store.open_data('ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-1978-2020-fv05.3.zarr')
        self.assertIsInstance(sm_dataset, xr.Dataset)
