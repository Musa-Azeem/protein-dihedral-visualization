import numpy as np
import pandas as pd
import statsmodels.api as sm

def fit_linregr(ins):
    ins.grouped_preds = ins.grouped_preds.sort_values('protein_id')
    ins.grouped_preds_md = ins.grouped_preds_md.sort_values('protein_id')
    X = ins.grouped_preds_md.values
    y = ins.grouped_preds.RMS_CA.values
    X[np.isnan(X)] = 0

    if X.shape[1] > X.shape[0]:
        print('WARNING: More features than samples', X.shape[0], X.shape[1])

    X = sm.add_constant(X)
    ins.model = sm.OLS(y, X).fit()

    print(f'Model R-squared: {ins.model.rsquared:.6f}, Adj R-squared: {ins.model.rsquared_adj:.6f}, p-value: {ins.model.f_pvalue}')

    ins.grouped_preds['rms_pred'] = ins.model.predict(X)