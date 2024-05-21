import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from lib.utils import find_phi_psi_c, calc_maha_for_one, calc_maha

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
def plot_one_dist(ins, seq, pred_id, pred_name, axlims, bw_method):
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
    plt.show()

def plot_one_dist_3d(ins, seq, bw_method):
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

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')
    ax.set_zlabel('Density')
    plt.tight_layout()
    plt.show()

def plot_clusters_for_seq(ins, seq, bw_method):
    bw_method = bw_method or ins.bw_method
    inner_seq = ins.get_subseq(seq)
    phi_psi_dist = ins.phi_psi_mined.loc[ins.phi_psi_mined.seq == inner_seq][['phi','psi', 'weight']].copy()
    phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt.loc[ins.phi_psi_mined_ctxt.seq == seq][['phi','psi', 'weight']].copy()
    phi_psi_dist, phi_psi_dist_c, most_likely = find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, bw_method)
    value_counts = phi_psi_dist.cluster.value_counts().sort_values(ascending=False)
    print('Cluster Counts:', value_counts.to_dict())
    fig, ax = plt.subplots(figsize=(7,7))
    sns.scatterplot(data=phi_psi_dist, x='phi', y='psi', hue='cluster', palette='tab10')
    ax.scatter(most_likely.phi, most_likely.psi, marker='x', color='r')

def plot_md_for_seq(ins, seq, pred_id, pred_name, bw_method, axlims):
    pred_name = pred_name or pred_id
    bw_method = bw_method or ins.bw_method
    inner_seq = ins.get_subseq(seq)

    phi_psi_dist = ins.phi_psi_mined.loc[ins.phi_psi_mined.seq == inner_seq][['phi','psi', 'weight']].copy()
    phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt.loc[ins.phi_psi_mined_ctxt.seq == seq][['phi','psi', 'weight']].copy()
    xray = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]
    alpha = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == pred_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    preds = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq]

    if xray.shape[0] == 0:
        print('No xray data for this window')
        return
    if alpha.shape[0] == 0:
        print('No alpha data for this window')
        return
    if preds.shape[0] == 0:
        print('No predictions for this window')
        return
    pos = xray['pos'].values[0]
    res = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq==inner_seq, 'res'].values[0]

    print(f'Window {ins.winsize_ctxt} centered at {pos} of {seq}')
    print(f'Window {ins.winsize}: {phi_psi_dist.shape[0]} samples, Window {ins.winsize_ctxt}: {phi_psi_ctxt_dist.shape[0]} samples')

    phi_psi_dist, phi_psi_dist_c, most_likely = find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, bw_method)

    # Mahalanobis distance to most common cluster
    md_xray = calc_maha_for_one(xray[['phi','psi']].values[0], phi_psi_dist_c[['phi','psi']].values, most_likely[['phi', 'psi']].values)
    md_alpha = calc_maha_for_one(alpha[['phi','psi']].values[0], phi_psi_dist_c[['phi','psi']].values, most_likely[['phi', 'psi']].values)
    md_preds = calc_maha(preds[['phi','psi']].values, phi_psi_dist_c[['phi','psi']].values, most_likely[['phi', 'psi']].values)

    value_counts = phi_psi_dist.cluster.value_counts().sort_values(ascending=False)
    print('Clusters:', value_counts.to_dict())
    print('xray:', md_xray)
    print('alpha:', md_alpha)
    print('preds:\n', pd.DataFrame(md_preds).describe())

    fig, ax = plt.subplots(figsize=(9,7))
    sns.kdeplot(data=phi_psi_dist_c, x='phi', y='psi', ax=ax, legend=False, fill=True)
    ax.scatter(preds.phi, preds.psi, color='black', marker='.', alpha=0.2, label='All CASP-14 Predictions')
    ax.scatter(xray.phi, xray.psi, color='green', marker='x', label='X-ray')
    ax.scatter(alpha.phi, alpha.psi, color='blue', marker='x', label=pred_name)
    ax.scatter(most_likely.phi, most_likely.psi, color='red', marker='x', label='KDE Peak')

    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')
    ax.set_title(f'PDBMine Dihedral Distribution for Residue {res} of Window {seq} (Position {pos})')
    ax.legend()
    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])
    plt.tight_layout()
    plt.show()

def plot_res_vs_md(ins, pred_id, pred_name, highlight_res, limit_quantile):
    # Plot xray vs prediction md for each residue of one prediction
    pred_name = pred_name or pred_id
    pred = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.protein_id == pred_id]
    both = pd.merge(pred, ins.xray_phi_psi[['seq', 'md']].copy(), how='inner', on=['seq','seq'], suffixes=('_pred','_xray'))
    both['md_diff'] = both['md_pred'] - both['md_xray']
    fig, axes = plt.subplots(2, figsize=(15,10), sharex=True)
    if limit_quantile:
        both[both.md_pred> both.md_pred.quantile(limit_quantile)] = np.nan
        both[both.md_xray > both.md_xray.quantile(limit_quantile)] = np.nan
        both[both.md_diff > both.md_diff.quantile(limit_quantile)] = np.nan

    sns.lineplot(data=both, x=both.index, y='md_pred', ax=axes[0], label=pred_name)
    sns.lineplot(data=both, x=both.index, y='md_xray', ax=axes[0], label='X-Ray')
    axes[0].set_ylabel('Mahalanobis Distance')
    axes[1].hlines(both['md_diff'].mean(), xmin=0, xmax=len(both), color='tan', label='Average')
    axes[1].fill_between(
        x=both.index, 
        y1=both['md_diff'].mean() + both['md_diff'].std(), 
        y2=both['md_diff'].mean() - both['md_diff'].std(), 
        color='tan', alpha=0.2
    )
    sns.lineplot(data=both.reset_index(), x='index', y='md_diff', ax=axes[1], label=f'{pred_name} minus X-Ray')
    axes[1].set_ylabel('Mahalanobis Distance')
    axes[1].set_xlabel('Reisdue Position in Chain')
    fig.suptitle('Mahalanobis Distance from phi-psi of each residue to the distribution for its window')
    plt.tight_layout()

    if highlight_res:
        for ax in axes:
            ax.axvspan(highlight_res[0], highlight_res[1], color='red', alpha=0.2)
    
    return both

def plot_md_vs_rmsd(ins, axlims):
    fig, ax = plt.subplots()
    sns.scatterplot(data=ins.grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, marker='.', legend=False)
    sns.regplot(data=ins.grouped_preds, x='rms_pred', y='RMS_CA', ax=ax, ci=False, scatter=False, color='red', line_kws={'lw':1}, label='Regression Line')
    ax.set_xlabel('Average Mahalanobis Distance')
    ax.set_ylabel('Prediction RMSD')
    ax.set_title(r'Average Mahalanobis Distance vs RMSD ($C_{\alpha}$) for each prediction')
    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])
    plt.tight_layout()
    plt.show()

def plot_heatmap(ins):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    X = ins.grouped_preds_md.values
    X = np.where(np.isnan(X), np.nanmean(X,axis=0), X)
    sns.heatmap(X, ax=ax)