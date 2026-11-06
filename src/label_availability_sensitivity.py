"""One-month target-label availability sensitivity test for baseline predictions.

At forecast origin t, this conservative case trains on targets up to month t-1.
This does not establish the exact historical publication date of the French factors.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from milestone2_models import feature_engineering, select_features, _model, performance, Y

def forecast_lagged_labels(frame, min_train=120):
    df=feature_engineering(frame)
    df['forecast_origin']=pd.to_datetime(df['forecast_origin'])
    df['target_month']=(df['forecast_origin'].dt.to_period('M')+1).astype(str)
    features=select_features(df)
    output=[]
    for i in range(len(df)):
        origin=df.iloc[i].forecast_origin
        max_label_month=origin.to_period('M') - 1
        hist=df.iloc[:i]
        train=hist.loc[(pd.PeriodIndex(hist.target_month,freq='M')<=max_label_month)&hist[Y].notna()]
        if len(train)<min_train or pd.isna(df.iloc[i][Y]): continue
        xtrain=train[features].astype(float); ytrain=train[Y].astype(float)
        valid=[c for c in features if xtrain[c].notna().any()]
        if not valid: continue
        xpred=df.loc[[i],valid].astype(float)
        rec={'forecast_origin':origin.date().isoformat(), 'target_month':df.iloc[i].target_month,
             'actual':float(df.iloc[i][Y]),'n_train':len(train),'n_features':len(valid),'HistoricalMean':float(ytrain.mean())}
        for k in ['OLS','Ridge','ElasticNet']:
            mdl=_model(k);mdl.fit(xtrain[valid],ytrain);rec[k]=float(mdl.predict(xpred)[0])
        rec['Combination']=float(np.mean([rec[k] for k in ['OLS','Ridge','ElasticNet']]))
        output.append(rec)
    return pd.DataFrame(output)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True)
    ap.add_argument('--original-predictions',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--min-train',type=int,default=120)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lag=forecast_lagged_labels(pd.read_csv(a.features),a.min_train)
    lag.to_csv(out/'walk_forward_label_lag1.csv',index=False)
    performance(lag).to_csv(out/'baseline_performance_label_lag1.csv',index=False)
    orig=pd.read_csv(a.original_predictions)
    common=orig.merge(lag,on=['forecast_origin','target_month'],suffixes=('_original','_lag1'))
    models=['HistoricalMean','OLS','Ridge','ElasticNet','Combination']
    rows=[]
    for name in models:
        actual=common.actual_original.to_numpy()
        for kind,suffix in [('original','_original'),('lag1','_lag1')]:
            predictions=common[name+suffix].to_numpy()
            rows.append({'model':name,'case':kind,'n_common':len(common),
                         'rmse_pct_points':float(np.sqrt(np.mean((actual-predictions)**2)))})
    pd.DataFrame(rows).to_csv(out/'baseline_original_vs_label_lag1_common.csv',index=False)
    print('Saved label-delay sensitivity results to',out)

if __name__=='__main__': main()
