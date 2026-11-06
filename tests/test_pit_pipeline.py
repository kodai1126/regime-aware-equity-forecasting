import sys
import unittest
from datetime import date
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pit_pipeline import feature_from_rows, month_ends, attach_realized_target

class AsOfTests(unittest.TestCase):
    def test_month_end_and_order(self):
        self.assertEqual(month_ends('2020-01','2020-03'),
                         [date(2020,1,31),date(2020,2,29),date(2020,3,31)])

    def test_no_future_obs_or_revision(self):
        rows = [
            {'date':'2020-01-01','value':'3.5','realtime_start':'2020-02-07','realtime_end':'2020-04-02'},
            {'date':'2020-01-01','value':'3.7','realtime_start':'2020-04-03','realtime_end':'9999-12-31'},
            {'date':'2020-02-01','value':'3.6','realtime_start':'2020-03-06','realtime_end':'9999-12-31'},
        ]
        feature = feature_from_rows(rows, date(2020,2,29),'level')
        self.assertEqual(feature['value'],3.5)
        self.assertEqual(feature['last_observation_date'],'2020-01-01')

    def test_yoy_transform_asof(self):
        rows = [
            {'date':'2019-01-01','value':'100','realtime_start':'2019-02-01','realtime_end':'9999-12-31'},
            {'date':'2020-01-01','value':'110','realtime_start':'2020-02-01','realtime_end':'9999-12-31'},
        ]
        feature = feature_from_rows(rows, date(2020,2,29),'yoy_pct')
        self.assertAlmostEqual(feature['value'], 10.0)

    def test_no_implicit_future_label(self):
        x = pd.DataFrame([{'forecast_origin':'2020-01-31'}, {'forecast_origin':'2020-02-29'}])
        y = pd.DataFrame([{'target_month':'2020-02', 'target_next_month_excess_return_pct':1.2},
                          {'target_month':'2020-03', 'target_next_month_excess_return_pct':-5.0}])
        z = attach_realized_target(x,y)
        self.assertEqual(list(z.target_month),['2020-02','2020-03'])
        self.assertEqual(list(z.target_next_month_excess_return_pct),[1.2,-5.0])

    def test_missing_history_not_filled_with_future(self):
        rows = [{'date':'2020-02-01','value':'3.5','realtime_start':'2020-03-06','realtime_end':'9999-12-31'}]
        feature = feature_from_rows(rows, date(2020,2,29),'level')
        self.assertIsNone(feature['value'])
        self.assertEqual(feature['status'],'no_asof_observation')

if __name__ == '__main__':
    unittest.main()
