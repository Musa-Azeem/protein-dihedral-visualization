import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from scipy.stats import linregress
from lib.utils import find_kdepeak, calc_da, calc_da_for_one

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def plot_one_dist(ins, seq, pred_id, pred_name, axlims, bw_method, fn):
    pred_name = pred_name or pred_id
    bw_method = bw_method if bw_method != -1 else ins.bw_method
    inner_seq = ins.get_subseq(seq)
    phi_psi_dist = ins.phi_psi_mined[ins.phi_psi_mined.seq == inner_seq].copy()
    phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt[ins.phi_psi_mined_ctxt.seq == seq].copy()
    phi_psi_alpha = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == pred_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    xray_phi_psi_seq = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]

    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])
    counts = phi_psi_dist.groupby('weight').count().iloc[:,0].values
    print(f' Winsize={ins.winsize}: {counts[0]} samples, Winsize={ins.winsize_ctxt}: {counts[1]} samples')

    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=phi_psi_ctxt_dist, x='phi', y='psi', ax=ax, color=colors[1],zorder=5, alpha=0.5, marker='.')
    sns.kdeplot(data=phi_psi_dist, x='phi', y='psi', weights='weight', ax=ax, fill=True, color=colors[0], bw_method=bw_method)

    kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T, weights=phi_psi_dist['weight'], bw_method=bw_method)
    most_likely = phi_psi_dist.iloc[kernel(phi_psi_dist[['phi', 'psi']].values.T).argmax()]
    
    ax.plot(phi_psi_alpha.phi, phi_psi_alpha.psi, 'o', color=colors[2], label='AlphaFold Prediction', zorder=10)
    ax.plot(xray_phi_psi_seq.phi, xray_phi_psi_seq.psi, 'o', color=colors[3], label='X-ray', zorder=10)
    ax.scatter(most_likely.phi, most_likely.psi, color=colors[4], marker='x', label='KDE Peak', zorder=20)
    sns.scatterplot(data=ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq], x='phi', y='psi', ax=ax, color='black', zorder=5, alpha=0.2, marker='.')

    ax.legend(handles=[
        mpatches.Patch(color=colors[0], label=f'Query for {inner_seq}'),
        mpatches.Patch(color=colors[1], label=f'Query for {seq}'),
        mpatches.Patch(color=colors[2], label=f'{pred_name} Prediction'),
        mpatches.Patch(color=colors[3], label='X-ray'),
        mpatches.Patch(color=colors[4], label='KDE Peak'),
        mpatches.Patch(color='black', label='Other Predictions')
    ])
    ax.set_title(f'PDBMine Distribution of Dihedral Angles for Resiude {seq[ins.winsize_ctxt//2]} of Window {seq}')

    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')

    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])

    plt.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

def plot_one_dist_3d(ins, seq, bw_method, fn):
    bw_method = bw_method if bw_method != -1 else ins.bw_method
    inner_seq = ins.get_subseq(seq)
    phi_psi_dist = ins.phi_psi_mined[ins.phi_psi_mined.seq == inner_seq].copy()
    phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt[ins.phi_psi_mined_ctxt.seq == seq].copy()

    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])

    x = phi_psi_dist[['phi','psi']].values.T
    weights = phi_psi_dist['weight'].values
    kde = gaussian_kde(x, weights=weights, bw_method=bw_method)

    x_grid, y_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-180, 180, 360))
    grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid).reshape(x_grid.shape)
    print(f'Max: P({grid[0,z.argmax()]}, {grid[1,z.argmax()]})={z.max()}')

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

def plot_md_for_seq(ins, seq, pred_id, pred_name, bw_method, axlims, fn, fill):
    pred_name = pred_name or pred_id[5:]
    bw_method = bw_method or ins.bw_method
    inner_seq = ins.get_subseq(seq)
    phi_psi_dist = ins.phi_psi_mined.loc[ins.phi_psi_mined.seq == inner_seq][['phi','psi', 'weight']].copy()
    phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt.loc[ins.phi_psi_mined_ctxt.seq == seq][['phi','psi', 'weight']].copy()
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
    res = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq==inner_seq, 'res'].values[0]

    print(f'Window {ins.winsize_ctxt} centered at {pos} of {seq}')
    print(f'Window {ins.winsize}: {phi_psi_dist.shape[0]} samples, Window {ins.winsize_ctxt}: {phi_psi_ctxt_dist.shape[0]} samples')

    kdepeak = find_kdepeak(phi_psi_dist, phi_psi_ctxt_dist, bw_method)

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
        data=pd.concat([phi_psi_dist, phi_psi_ctxt_dist]), 
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

def plot_res_vs_md(ins, pred_id, pred_name, highlight_res, limit_quantile, legend_loc, fn):
    # Plot xray vs prediction md for each residue of one prediction
    pred_name = pred_name or pred_id
    pred = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.protein_id == pred_id]
    both = pd.merge(pred, ins.xray_phi_psi[['seq', 'md']].copy(), how='inner', on=['seq','seq'], suffixes=('_pred','_xray'))
    both['md_diff'] = both['md_pred'] - both['md_xray']
    if limit_quantile:
        both[both.md_pred > both.md_pred.quantile(limit_quantile)] = np.nan
        both[both.md_xray > both.md_xray.quantile(limit_quantile)] = np.nan
        both[both.md_diff > both.md_diff.quantile(limit_quantile)] = np.nan
    
    fig, axes = plt.subplots(2, figsize=(10, 5), sharex=True)
    sns.lineplot(data=both, x='pos', y='md_pred', ax=axes[0], label=pred_name)
    sns.lineplot(data=both, x='pos', y='md_xray', ax=axes[0], label='X-Ray')
    axes[0].set_ylabel('')
    axes[0].legend(loc=legend_loc)

    sns.lineplot(data=both, x='pos', y='md_diff', ax=axes[1], label=f'Difference:\n{pred_name} - Xray')
    axes[1].fill_between(
        x=both.pos, 
        y1=both['md_diff'].mean() + both['md_diff'].std(), 
        y2=both['md_diff'].mean() - both['md_diff'].std(), 
        color='tan', 
        alpha=0.4
    )
    axes[1].hlines(both['md_diff'].mean(), xmin=both.pos.min(), xmax=both.pos.max(), color='tan', label='Mean Difference', linewidth=0.75)
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

def plot_md_vs_rmsd(ins, axlims, fn):
    regr = linregress(ins.grouped_preds.rms_pred, ins.grouped_preds.RMS_CA)

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=ins.grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, marker='o', s=25, edgecolor='b', legend=True)
    ax.plot(
        regr.intercept + regr.slope * np.linspace(0, ins.grouped_preds.rms_pred.max() + 5, 100), 
        np.linspace(0, ins.grouped_preds.rms_pred.max() + 5, 100), 
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
    ins.grouped_preds_md = ins.grouped_preds_md.sort_index()
    df = ins.grouped_preds_md.copy()
    df['rmsd'] = ins.grouped_preds['RMS_CA'].values
    df = df.sort_values('rmsd')
    af_idx = df.index.get_loc(ins.alphafold_id)
    # print(ins.grouped_preds[ins.grouped_preds.protein_id == ins.alphafold_id])
    X = df.iloc[:, :-1].values
    print(X.shape)
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