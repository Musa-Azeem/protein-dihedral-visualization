from lib import DihedralAdherence
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def fit_lr(protein_ids: list, winsizes, kdews, pdbmine_url, project_dir, n_comp=1000):
    X = []
    y = []
    grouped_preds = []
    grouped_preds_da = []
    longest_protein = 0
    for protein_id in protein_ids:
        da = DihedralAdherence(protein_id, winsizes, pdbmine_url, project_dir, kdews=kdews)
        # da.compute_structures()
        # da.query_pdbmine()
        # da.load_results()
        # da.compute_das(replace=True)
        da.load_results_da()
        da.filter_nas()
        grouped_preds.append(da.grouped_preds.sort_values(['protein_id']))
        grouped_preds_da.append(da.grouped_preds_da.sort_values(['protein_id']))
        Xi = da.grouped_preds_da.sort_values('protein_id').values
        Xi[np.isnan(Xi)] = 0

        if Xi.shape[1] > longest_protein:
            longest_protein = Xi.shape[1]

        if Xi.shape[1] < n_comp:
            # Pad Xi if less then n_comp
            # Xi = np.sort(np.pad(Xi, ((0, 0), (0, n_comp - Xi.shape[1])), mode='constant', constant_values=0))[:,::-1]
            Xi = np.pad(Xi, ((0, 0), (0, n_comp - Xi.shape[1])), mode='constant', constant_values=0)
            # Xi = np.sort(Xi)[:,::-1] # sort in descending order
        else:
            # sort Xi in descending order
            # Xi = np.sort(Xi)[:,::-1]
            # truncate to n_comp
            Xi = Xi[:,:n_comp]
        yi = grouped_preds[-1]['RMS_CA'].values

        # normalize
        # Xi = (Xi - Xi.mean(axis=0)) / (Xi.std(axis=0) + 1e-8)
        # yi = (yi - yi.mean()) / (yi.std() + 1e-8)
        # normalize to be between 0 and 1
        # Xi = (Xi - Xi.min(axis=0)) / (Xi.max(axis=0) - Xi.min(axis=0) + 1e-8)
        # yi = (yi - yi.min()) / (yi.max() - yi.min() + 1e-8)

        X.append(Xi)
        y.append(yi)

    # truncate to longest protein length if its less than n_comp
    # X = [Xi[:,:min(longest_protein, n_comp)] for Xi in X]

    X = np.concatenate(X)
    y = np.concatenate(y)
    grouped_preds = pd.concat(grouped_preds)

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    grouped_preds['rms_pred'] = model.predict(X)
    grouped_preds['RMS_CA'] = y
    print(f'Model R-squared: {model.rsquared:.6f}, Adj R-squared: {model.rsquared_adj:.6f}, p-value: {model.f_pvalue}')
    return model, grouped_preds

def predict_lr(model, protein_ids, winsize, winsize_ctxt, pdbmine_url, project_dir, n_comp=1000):
    X = []
    y = []
    grouped_preds = []
    grouped_preds_da = []
    for protein_id in protein_ids:
        da = DihedralAdherence(protein_id, winsize, winsize_ctxt, pdbmine_url, project_dir)
        # da.compute_structures()
        # da.query_pdbmine()
        # da.compute_mds()
        da.load_results_md()
        da.filter_nas()
        grouped_preds.append(da.grouped_preds.sort_values(['protein_id']))
        grouped_preds_da.append(da.grouped_preds_da.sort_values(['protein_id']))
        Xi = da.grouped_preds_da.sort_values('protein_id').values
        Xi[np.isnan(Xi)] = 0
        

        if Xi.shape[1] < n_comp:
            # Pad Xi if less then n_comp
            Xi = np.pad(Xi, ((0, 0), (0, n_comp - Xi.shape[1])), mode='constant', constant_values=0)
        else:
            # truncate to n_comp
            Xi = Xi[:,:n_comp]
        yi = grouped_preds[-1]['RMS_CA'].values

        X.append(Xi)
        y.append(yi)

    X = np.concatenate(X)
    y = np.concatenate(y)
    grouped_preds = pd.concat(grouped_preds)

    X = sm.add_constant(X)
    grouped_preds['rms_pred'] = model.predict(X)
    grouped_preds['RMS_CA'] = y
    print(f'Model R-squared: {model.rsquared:.6f}, Adj R-squared: {model.rsquared_adj:.6f}, p-value: {model.f_pvalue}')
    return model, grouped_preds

def fit_rf(protein_ids: list, winsize, winsize_ctxt, pdbmine_url, project_dir, n_comp=900):
    grouped_preds = []
    X = []
    y = []
    for protein_id in protein_ids:
        da = DihedralAdherence(protein_id, winsize, winsize_ctxt, pdbmine_url, project_dir)
        da.load_results_md()
        da.filter_nas()
        grouped_preds.append(da.grouped_preds.sort_values(['protein_id']))
        Xi = da.grouped_preds_da.sort_values('protein_id').values
        Xi = np.pad(Xi, ((0, 0), (0, n_comp - Xi.shape[1])), mode='constant', constant_values=0)
        X.append(Xi)
        y.append(grouped_preds[-1]['RMS_CA'].values)

    X = np.concatenate(X)
    y = np.concatenate(y)
    grouped_preds = pd.concat(grouped_preds)
    X[np.isnan(X)] = 0

    rf = RandomForestRegressor(n_estimators=1, max_depth=50)
    rf.fit(X, y)
    grouped_preds['rms_pred'] = rf.predict(X)

    return rf, grouped_preds

def predict_rf(rf, protein_ids, winsize, winsize_ctxt, pdbmine_url, project_dir, n_comp=900):
    grouped_preds = []
    X = []
    y = []
    longest_protein = 0
    for protein_id in protein_ids:
        da = DihedralAdherence(protein_id, winsize, winsize_ctxt, pdbmine_url, project_dir)
        da.load_results_md()
        da.filter_nas()
        grouped_preds.append(da.grouped_preds.sort_values(['protein_id']))
        Xi = da.grouped_preds_da.sort_values('protein_id').values
        Xi = np.pad(Xi, ((0, 0), (0, n_comp - Xi.shape[1])), mode='constant', constant_values=0)

        if Xi.shape[1] > longest_protein:
            longest_protein = Xi.shape[1]
        
        X.append(Xi)
        y.append(grouped_preds[-1]['RMS_CA'].values)

    X = [np.pad(Xi, ((0, 0), (0, longest_protein - Xi.shape[1])), mode='constant', constant_values=0) for Xi in X]
    X = np.concatenate(X)
    y = np.concatenate(y)
    grouped_preds = pd.concat(grouped_preds)
    X[np.isnan(X)] = 0

    grouped_preds['rms_pred'] = rf.predict(X)

    return grouped_preds

def plot_md_vs_rmsd(grouped_preds, axlims=None):
    regr = linregress(grouped_preds.rms_pred, grouped_preds.RMS_CA)

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    fig, ax = plt.subplots(figsize=(8, 8))

    # sns.scatterplot(data=grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, marker='o', s=25, edgecolor='b', legend=True, hue='target')
    sns.scatterplot(data=grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, marker='o', s=25, edgecolor='b', legend=True)
    ax.plot(
        np.linspace(0, grouped_preds.rms_pred.max(), 100), 
        regr.intercept + regr.slope * np.linspace(0, grouped_preds.rms_pred.max(), 100), 
        color='red', lw=2, label='Regression Line')
    ax.set_xlabel('Regression-Aggregated Dihedral Adherence Score', fontsize=14, labelpad=15)
    ax.set_ylabel('Prediction Backbone RMSD', fontsize=14, labelpad=15)
    ax.set_title(r'Aggregated Dihedral Adherence vs RMSD ($C_{\alpha}$) for each prediction', fontsize=16, pad=20)
    ax.text(0.85, 0.10, r'$R^2$='+f'{regr.rvalue**2:.3f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    plt.legend(fontsize=12)
    plt.tight_layout()

    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])

    plt.show()
    
    sns.reset_defaults()