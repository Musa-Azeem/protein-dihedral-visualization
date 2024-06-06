import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from scipy.stats import linregress
from lib.utils import find_kdepeak, calc_da, calc_da_for_one, get_phi_psi_dist

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def plot_one_dist(ins, seq, pred_id, pred_name, axlims, bw_method, fn):
    pred_name = pred_name or pred_id[5:]
    bw_method = bw_method if bw_method != -1 else ins.bw_method
    phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
    phi_psi_pred = ins.phi_psi_predictions[
        (ins.phi_psi_predictions.protein_id == pred_id) & 
        (ins.phi_psi_predictions.seq_ctxt == seq)
    ]
    phi_psi_alphafold = ins.phi_psi_predictions[
        (ins.phi_psi_predictions.protein_id == ins.alphafold_id) & 
        (ins.phi_psi_predictions.seq_ctxt == seq)
    ]
    xray_phi_psi_seq = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]
    preds = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq]

    for i in info:
        print(f'Win {i[0]}: {i[1]} - {i[2]} samples')

    kdepeak = find_kdepeak(phi_psi_dist, bw_method)

    fig, ax = plt.subplots(figsize=(7,5))
    sns.kdeplot(data=phi_psi_dist, x='phi', y='psi', weights='weight', ax=ax, fill=True, color=colors[0], bw_method=bw_method)
    ax.scatter(xray_phi_psi_seq.phi, xray_phi_psi_seq.psi, marker='o', color=colors[1], label='X-ray', zorder=10)
    ax.scatter(phi_psi_pred.phi, phi_psi_pred.psi, marker='o', color=colors[2], label=f'{pred_name} Prediction', zorder=10)
    ax.scatter(phi_psi_alphafold.phi, phi_psi_alphafold.psi, marker='o', color=colors[4], label='AlphaFold', zorder=10)
    ax.scatter(kdepeak.phi, kdepeak.psi, color='red', marker='x', label='KDE Peak', zorder=20)
    sns.scatterplot(data=preds, x='phi', y='psi', ax=ax, color='black', zorder=5, alpha=0.2, marker='.')
    ax.legend(loc='lower left')

    ax.set_title(f'PDBMine Distribution of Dihedral Angles for Residue {xray_phi_psi_seq.res.values[0]} of Window {seq}', fontsize=14)
    ax.set_xlabel('Phi', fontsize=12)
    ax.set_ylabel('Psi', fontsize=12)

    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])

    plt.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

def plot_one_dist_3d(ins, seq, bw_method, fn):
    bw_method = bw_method if bw_method != -1 else ins.bw_method
    phi_psi_dist,_ = get_phi_psi_dist(ins.queries, seq)

    x = phi_psi_dist[['phi','psi']].values.T
    weights = phi_psi_dist['weight'].values
    kde = gaussian_kde(x, weights=weights, bw_method=bw_method)

    x_grid, y_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-180, 180, 360))
    grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid).reshape(x_grid.shape)
    print(f'Max: P({grid[0,z.argmax()]:02f}, {grid[1,z.argmax()]:02f})={z.max():02f}')

    cm = plt.get_cmap('turbo')
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z, cmap=cm)
    ax.set_xlabel('Phi', fontsize=12, labelpad=10)
    ax.set_ylabel('Psi', fontsize=12, labelpad=10)
    ax.set_zlabel('Density', fontsize=12, labelpad=10)
    ax.set_title(f'PDBMine Distribution of Dihedral Angles\nfor Residue {seq[ins.winsize_ctxt//2]} of Window {seq}', y=0.99, fontsize=14)
    ax.dist = 12

    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.zaxis.set_tick_params(labelsize=10)

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

def plot_da_for_seq(ins, seq, pred_id, pred_name, bw_method, axlims, fn, fill):
    pred_name = pred_name or pred_id[5:]
    bw_method = bw_method or ins.bw_method
    phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
    xray = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]
    pred = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == pred_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    preds = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq]
    alphafold = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == ins.alphafold_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]

    if xray.shape[0] == 0:
        print('No xray data for this window')
        return
    if pred.shape[0] == 0:
        print(f'No {pred_id} data for this window')
        return
    if preds.shape[0] == 0:
        print('No predictions for this window')
        return
    pos = xray['pos'].values[0]
    res = xray.res.values[0]

    print(f'Residue {res} of Window {seq} centered at {pos} of {seq}')
    for i in info:
        print(f'\tWin {i[0]}: {i[1]} - {i[2]} samples')

    kdepeak = find_kdepeak(phi_psi_dist, bw_method)

    # Mahalanobis distance to most common cluster
    da_xray = calc_da_for_one(kdepeak[['phi', 'psi']].values, xray[['phi','psi']].values[0])
    da_pred = calc_da_for_one(kdepeak[['phi', 'psi']].values, pred[['phi','psi']].values[0])
    da_alphafold = calc_da_for_one(kdepeak[['phi', 'psi']].values, alphafold[['phi','psi']].values[0])
    da_preds = calc_da(kdepeak[['phi', 'psi']].values, preds[['phi','psi']].values)

    print(f'KDEpeak:\t ({kdepeak.phi:.02f}, {kdepeak.psi:.02f})')
    print(f'X-ray[{pos}]:\t ({xray.phi.values[0]:.02f}, {xray.psi.values[0]:.02f}), DA={da_xray:.02f}')
    print(f'{pred_name}[{pred.pos.values[0]}]:\t ({pred.phi.values[0]:.02f}, {pred.psi.values[0]:.02f}), DA={da_pred:.02f}')
    print(f'AlphaFold[{alphafold.pos.values[0]}]:\t ({alphafold.phi.values[0]:.02f}, {alphafold.psi.values[0]:.02f}), DA={da_alphafold:.02f}')
    print('Other Predictions DA:\n', pd.DataFrame(da_preds).describe())

    fig, ax = plt.subplots(figsize=(9,7))
    sns.kdeplot(
        data=phi_psi_dist, 
        x='phi', y='psi', weights='weight',
        ax=ax, levels=8, zorder=0, 
        fill=fill, color='black'
    )
    ax.scatter(preds.phi, preds.psi, color='black', marker='o', s=5, alpha=0.2, label='All Other CASP-14 Predictions', zorder=1)
    ax.scatter(xray.iloc[0].phi, xray.iloc[0].psi, color=colors[1], marker='o', label='X-ray', zorder=10, s=100)
    ax.scatter(pred.phi, pred.psi,  color=colors[2], marker='o', label=pred_name, zorder=10, s=100)
    ax.scatter(alphafold.phi, alphafold.psi, color=colors[4], marker='o', label='AlphaFold', zorder=10, s=100)
    ax.scatter(kdepeak.phi, kdepeak.psi, color='red', marker='x', label='KDE Peak')

    # dotted line from each point to mean
    ax.plot([xray.phi.values[0], kdepeak.phi], [xray.psi.values[0], kdepeak.psi], linestyle='dashed', color=colors[1], zorder=1, linewidth=1)
    ax.plot([pred.phi.values[0], kdepeak.phi], [pred.psi.values[0], kdepeak.psi], linestyle='dashed', color=colors[2], zorder=1, linewidth=1)
    ax.plot([alphafold.phi.values[0], kdepeak.phi], [alphafold.psi.values[0], kdepeak.psi], linestyle='dashed', color=colors[4], zorder=1, linewidth=1)

    ax.set_xlabel('Phi', fontsize=12)
    ax.set_ylabel('Psi', fontsize=12)
    ax.set_title(r'Chosen Distribution of Dihedral Angles $D^{(i)}$ for Residue'+f' {res} of Window {seq}', fontsize=14)

    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])

    ax.legend(loc='lower left')
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

def plot_res_vs_da(ins, pred_id, pred_name, highlight_res, limit_quantile, legend_loc, fn):
    # Plot xray vs prediction da for each residue of one prediction
    pred_name = pred_name or pred_id
    pred = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.protein_id == pred_id]
    pred = pred.drop_duplicates(subset=['seq_ctxt']) # for plotting only
    xray = ins.xray_phi_psi[['pos', 'seq_ctxt', 'da']]
    xray = xray.drop_duplicates(subset=['seq_ctxt']) # for plotting only

    both = pd.merge(pred, xray, how='inner', on=['seq_ctxt','seq_ctxt'], suffixes=('_pred','_xray'))
    both['da_diff'] = both['da_pred'] - both['da_xray']
    both = both.rename(columns={'pos_pred':'pos'})
    # Add na rows for missing residues
    pos = np.arange(both.pos.min(), both.pos.max(), 1)
    both = both.set_index('pos').reindex(pos).reset_index()

    # Print highest values
    print('Highest DA Differences:\n')
    print(both.sort_values('da_diff', ascending=False).head(10)[
        ['pos', 'pos_xray', 'seq_ctxt','da_pred','da_xray','da_diff']
    ].to_markdown(index=False))

    if limit_quantile:
        both[both.da_pred > both.da_pred.quantile(limit_quantile)] = np.nan
        both[both.da_xray > both.da_xray.quantile(limit_quantile)] = np.nan
        both[both.da_diff > both.da_diff.quantile(limit_quantile)] = np.nan
    
    fig, axes = plt.subplots(2, figsize=(10, 5), sharex=True)
    # sns.lineplot(data=both, x='pos', y='da_pred', ax=axes[0], label=pred_name)
    # sns.lineplot(data=both, x='pos', y='da_xray', ax=axes[0], label='X-Ray')
    axes[0].plot(both.pos, both.da_pred, label=pred_name)
    axes[0].plot(both.pos, both.da_xray, label='X-Ray')
    axes[0].set_ylabel('')
    axes[0].legend(loc=legend_loc)

    # sns.lineplot(data=both, x='pos', y='da_diff', ax=axes[1], label=f'Difference:\n{pred_name} - Xray')
    axes[1].plot(both.pos, both.da_diff, label=f'Difference:\n{pred_name} - Xray')
    axes[1].fill_between(
        x=both.pos, 
        y1=both['da_diff'].mean() + both['da_diff'].std(), 
        y2=both['da_diff'].mean() - both['da_diff'].std(), 
        color='tan', 
        alpha=0.4
    )
    axes[1].hlines(both['da_diff'].mean(), xmin=both.pos.min(), xmax=both.pos.max(), color='tan', label='Mean Difference', linewidth=0.75)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('Residue Position in Chain', fontsize=12)
    axes[1].legend(loc=legend_loc)

    fig.text(0.845, 1.70, f'Pred RMSD={ins.results.loc[ins.results.Model == pred_id, "RMS_CA"].values[0]:.02f}', 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    fig.text(-0.02, 0.5, 'Dihedral Adherence of Residue', va='center', rotation='vertical', fontsize=12)
    fig.suptitle('Dihedral Adherence for each Residue of the Protein 7W6B: Prediction vs X-Ray', fontsize=16)
    plt.tight_layout()

    for highlight in highlight_res:
        for ax in axes:
            ax.axvspan(highlight[0], highlight[1], color='red', alpha=0.2)
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

    return both

def plot_da_vs_rmsd(ins, axlims, fn):
    regr = linregress(ins.grouped_preds.rms_pred, ins.grouped_preds.RMS_CA)

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=ins.grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, marker='o', s=25, edgecolor='b', legend=True)
    ax.plot(
        np.linspace(0, ins.grouped_preds.rms_pred.max() + 5, 100), 
        regr.intercept + regr.slope * np.linspace(0, ins.grouped_preds.rms_pred.max() + 5, 100), 
        color='red', lw=2, label='Regression Line'
    )
    # sns.regplot(data=ins.grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, scatter=False, 
    #             color='red', ci=False, label='Regression Line', line_kws={'lw':2})

    ax.set_xlabel('Regression-Aggregated Dihedral Adherence Score', fontsize=14, labelpad=15)
    ax.set_ylabel('Prediction Backbone RMSD', fontsize=14, labelpad=15)
    ax.set_title(r'Aggregated Dihedral Adherence vs RMSD ($C_{\alpha}$) for each prediction', fontsize=16, pad=20)
    ax.text(0.85, 0.10, r'$R^2$='+f'{ins.model.rsquared:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    
    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])

    plt.legend(fontsize=12)
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

    sns.reset_defaults()

def plot_heatmap(ins, fillna, fn):
    cmap = sns.color_palette("rocket", as_cmap=True)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ins.grouped_preds = ins.grouped_preds.sort_values('protein_id')
    ins.grouped_preds_da = ins.grouped_preds_da.sort_index()
    df = ins.grouped_preds_da.copy()
    df['rmsd'] = ins.grouped_preds['RMS_CA'].values
    df = df.sort_values('rmsd')
    af_idx = df.index.get_loc(ins.alphafold_id)
    # print(ins.grouped_preds[ins.grouped_preds.protein_id == ins.alphafold_id])
    X = df.iloc[:, :-1].values
    X = np.where(np.isnan(X), np.nanmean(X,axis=0), X)
    if fillna:
        X[np.isnan(X)] = 0 # for entire column nan
    sns.heatmap(X, ax=ax, cmap=cmap)

    ax.set_xlabel('Residue Position', fontsize=10)
    ax.set_yticks([af_idx + 0.5])
    ax.set_yticklabels([f'Alpha\nFold'], fontsize=7)
    ax.set_ylabel('Prediction', fontsize=10)
    ax.set_xticks([])
    ax.set_xticklabels([])

    cbar = ax.collections[0].colorbar
    cbar.set_label('Dihedral Adherence Magnitude', fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(f'Dihedral Adherence for each Residue of\nPredictions for the Protein {ins.casp_protein_id}', fontsize=12, pad=20)

    plt.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()


def plot_da_vs_rmsd_simple(ins, axlims, fn):
    grouped_preds = ins.grouped_preds.dropna()
    regr = linregress(grouped_preds.da, grouped_preds.RMS_CA)

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    fig, ax = plt.subplots(figsize=(8, 8))
    print(regr.intercept, regr.slope)
    sns.scatterplot(data=grouped_preds, x='da', y='RMS_CA', ax=ax, marker='o', s=25, edgecolor='b', legend=True)
    ax.plot(
        np.linspace(0, grouped_preds.da.max() + 5, 100), 
        regr.intercept + regr.slope * np.linspace(0, grouped_preds.da.max() + 5, 100), 
        color='red', lw=2, label='Regression Line'
    )

    ax.set_xlabel('Mean Dihedral Adherence Score', fontsize=14, labelpad=15)
    ax.set_ylabel('Prediction Backbone RMSD', fontsize=14, labelpad=15)
    ax.set_title(r'Mean Dihedral Adherence vs RMSD ($C_{\alpha}$) for each prediction', fontsize=16, pad=20)
    ax.text(0.85, 0.10, r'$R^2$='+f'{regr.rvalue**2:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    
    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])

    plt.legend(fontsize=12)
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

    sns.reset_defaults()