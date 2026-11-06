"""Chronological baseline forecasting for Milestone 2.

Run on a monthly as-of features CSV with realized next-month market excess returns.
The fit, scaling and imputation are repeated using only the expanding training window.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

Y='target_next_month_excess_return_pct'
META={'forecast_origin','target_month',Y,'sample_type'}

def feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic transforms of the as-of input, without fitted hindsight statistics."""
    df=frame.copy().sort_values('forecast_origin').reset_index(drop=True)
    numeric=[c for c in df.columns if c not in META and not c.endswith('__obs_date') and pd.api.types.is_numeric_dtype(df[c])]
    if {'ten_year_treasury_pct','fed_funds_rate_pct'} <= set(df):
        df['treasury_minus_fedfunds_pp']=df['ten_year_treasury_pct']-df['fed_funds_rate_pct']
    if 'unemployment_rate_pp' in df:
        df['unemployment_snapshot_change_3m_pp']=df['unemployment_rate_pp'].diff(3)
    if 'vix_level' in df:
        df['vix_log1p']=np.log1p(df['vix_level'].clip(lower=0))
    if 'us_epu_level' in df:
        df['epu_log1p']=np.log1p(df['us_epu_level'].clip(lower=0))
    for c in numeric:
        if df[c].isna().any():
            df[c+'_missing']=df[c].isna().astype(int)
    return df

def select_features(df, explicit=None):
    cols=explicit or [c for c in df if c not in META and not c.endswith('__obs_date') and pd.api.types.is_numeric_dtype(df[c])]
    cols=[c for c in cols if c in df and c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: raise ValueError('No numeric modeling features')
    if any('target' in c.lower() for c in cols): raise ValueError('Target-like feature forbidden')
    return cols

def _model(kind: str):
    if kind=='OLS': estimator=LinearRegression()
    elif kind=='Ridge': estimator=Ridge(alpha=10.0)
    elif kind=='ElasticNet': estimator=ElasticNet(alpha=0.15,l1_ratio=0.5,max_iter=10000,random_state=17)
    else: raise ValueError(kind)
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),
                     ('scaler',StandardScaler()),('model',estimator)])

def walk_forward(df, *, min_train=120, feature_cols=None, models=('OLS','Ridge','ElasticNet')):
    """At forecast origin t, train only on earlier origins and observed target months <= t."""
    if Y not in df: raise ValueError(f'Missing realized outcome {Y}')
    df=df.sort_values('forecast_origin').reset_index(drop=True).copy()
    if df['forecast_origin'].duplicated().any(): raise ValueError('Duplicate forecast origins')
    df['forecast_origin']=pd.to_datetime(df['forecast_origin'])
    df['target_month']=(df['forecast_origin'].dt.to_period('M')+1).astype(str)
    features=select_features(df,feature_cols)
    output=[]
    for i in range(len(df)):
        origin=df.loc[i,'forecast_origin']
        train=df.iloc[:i].copy()
        train=train.loc[(pd.to_datetime(train['target_month']+'-01').dt.to_period('M') <= origin.to_period('M')) & train[Y].notna()]
        if len(train)<min_train or pd.isna(df.loc[i,Y]): continue
        xtrain=train[features].astype(float); ytrain=train[Y].astype(float)
        xpred=df.loc[[i],features].astype(float)
        valid=[c for c in features if xtrain[c].notna().any()]
        if not valid: continue
        record={'forecast_origin':origin.date().isoformat(),'target_month':df.loc[i,'target_month'],
                'actual':float(df.loc[i,Y]),'n_train':len(train),'n_features':len(valid),
                'HistoricalMean':float(ytrain.mean())}
        for kind in models:
            model=_model(kind)
            model.fit(xtrain[valid],ytrain)
            record[kind]=float(model.predict(xpred[valid])[0])
        record['Combination']=float(np.mean([record[k] for k in models]))
        output.append(record)
    return pd.DataFrame(output)

def performance(pred: pd.DataFrame):
    if pred.empty: return pd.DataFrame()
    baseline=pred['HistoricalMean'].to_numpy()
    actual=pred['actual'].to_numpy()
    denom=np.sum((actual-baseline)**2)
    metrics=[]
    for model in ['HistoricalMean','OLS','Ridge','ElasticNet','Combination']:
        if model not in pred: continue
        p=pred[model].to_numpy()
        metrics.append({'model':model,'n_oos':len(pred),'rmse_pct_points':float(np.sqrt(mean_squared_error(actual,p))),
            'mae_pct_points':float(mean_absolute_error(actual,p)),
            'oos_r2_vs_historical_mean':float(1-np.sum((actual-p)**2)/denom) if denom>0 else np.nan,
            'directional_accuracy':float(np.mean(np.sign(actual)==np.sign(p)))})
    return pd.DataFrame(metrics)

def describe(df):
    cols=select_features(df)
    descriptions=[]
    for c in [Y]+cols:
        if c not in df: continue
        s=pd.to_numeric(df[c],errors='coerce')
        descriptions.append({'variable':c,'n':int(s.notna().sum()),'missing_pct':round(float(s.isna().mean()*100),2),
            'mean':float(s.mean()) if s.notna().any() else np.nan,
            'sd':float(s.std()) if s.notna().sum()>1 else np.nan,
            'min':float(s.min()) if s.notna().any() else np.nan,
            'max':float(s.max()) if s.notna().any() else np.nan})
    return pd.DataFrame(descriptions)

def run(input_path,output_path,min_train=120,allow_synthetic=False):
    inp=Path(input_path); out=Path(output_path);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(inp)
    if 'sample_type' in df.columns and df['sample_type'].astype(str).str.contains('synthetic',case=False).any() and not allow_synthetic:
        raise ValueError('Synthetic input blocked; use --allow-synthetic and label results DEMONSTRATION')
    if Y not in df: raise ValueError('No outcome labels; first attach Fama/French realized target using Milestone 1 pipeline')
    df=feature_engineering(df)
    df.to_csv(out/'monthly_modeling_table.csv',index=False)
    describe(df).to_csv(out/'descriptive_statistics.csv',index=False)
    results=walk_forward(df,min_train=min_train)
    if results.empty: raise ValueError('Insufficient labeled history for walk-forward; reduce --min-train only if scientifically justified')
    results.to_csv(out/'walk_forward_predictions.csv',index=False)
    performance(results).to_csv(out/'baseline_performance.csv',index=False)
    manifest={'input':str(inp),'n_months':len(df),'n_oos':len(results),'min_train':min_train,
              'output_flag':'SYNTHETIC DEMONSTRATION - NOT EMPIRICAL' if 'sample_type' in df and df['sample_type'].astype(str).str.contains('synthetic',case=False).any() else 'USER-SUPPLIED DATA - VERIFY PIT AUDIT',
              'feature_columns':select_features(df),'models':['HistoricalMean','OLS','Ridge','ElasticNet','Combination'],
              'important_limits':['Hyperparameters fixed ex ante; not tuned.','No HMM or political ablation until Milestone 3/4.',
                    'Strict PIT compliance depends on verified provenance of input features.']}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True);ap.add_argument('--out',default='output/milestone2')
    ap.add_argument('--min-train',type=int,default=120);ap.add_argument('--allow-synthetic',action='store_true')
    a=ap.parse_args(); print(json.dumps(run(a.input,a.out,a.min_train,a.allow_synthetic),indent=2))
if __name__=='__main__': main()
