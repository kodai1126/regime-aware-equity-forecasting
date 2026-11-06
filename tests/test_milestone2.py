import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import numpy as np
from milestone2_models import feature_engineering,walk_forward,performance,select_features

class TestMilestoneTwo(unittest.TestCase):
 def sample(self,n=20):
  return pd.DataFrame({'forecast_origin':pd.period_range('2020-01',periods=n,freq='M').to_timestamp('M').strftime('%Y-%m-%d'),
    'x':np.arange(n,dtype=float), 'target_next_month_excess_return_pct':np.arange(n,dtype=float)*2+1})
 def test_lagged_train_no_same_origin(self):
  out=walk_forward(self.sample(),min_train=5)
  self.assertEqual(out.iloc[0]['n_train'],5)
  self.assertEqual(out.iloc[0]['forecast_origin'],'2020-06-30')
  self.assertAlmostEqual(out.iloc[0]['HistoricalMean'],5.0)
 def test_future_change_cannot_change_earlier_forecast(self):
  a=self.sample();p=walk_forward(a,min_train=5)
  b=a.copy();b.loc[12:,'x']=90000.;b.loc[12:,'target_next_month_excess_return_pct']=88888.
  q=walk_forward(b,min_train=5)
  for c in ['HistoricalMean','OLS','Ridge','ElasticNet','Combination']:
   self.assertAlmostEqual(p.iloc[0][c],q.iloc[0][c])
 def test_derived_spread(self):
  df=self.sample();df['ten_year_treasury_pct']=4.0;df['fed_funds_rate_pct']=3.0
  self.assertEqual(feature_engineering(df).iloc[0]['treasury_minus_fedfunds_pp'],1.0)
 def test_target_excluded_from_features(self):
  self.assertNotIn('target_next_month_excess_return_pct',select_features(self.sample()))
 def test_metric_baseline_r2_zero(self):
  p=performance(walk_forward(self.sample(),min_train=5))
  self.assertAlmostEqual(p.set_index('model').loc['HistoricalMean','oos_r2_vs_historical_mean'],0.0)
 def test_duplicate_origin_fails(self):
  df=self.sample();df.loc[1,'forecast_origin']=df.loc[0,'forecast_origin']
  with self.assertRaises(ValueError):walk_forward(df,min_train=5)
if __name__=='__main__':unittest.main()
