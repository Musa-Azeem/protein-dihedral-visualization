import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress
from lib.utils import calc_da, calc_da_for_one, get_phi_psi_dist
from lib.across_window_utils import (
    get_combined_phi_psi_dist, get_xrays_window, get_afs_window, 
    get_preds_window, precompute_dists, find_clusters, 
    filter_precomputed_dists, get_cluster_medoid
)
from matplotlib.patches import ConnectionPatch


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def plot_one_dist_scatter(ins, seq, fn):
    phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
    xray_phi_psi = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]
    sns.scatterplot(data=phi_psi_dist, x='phi', y='psi', hue='winsize', alpha=0.8, palette='Dark2')
    plt.scatter(xray_phi_psi.phi, xray_phi_psi.psi, color='red', marker='x', label='X-ray')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title(f'PDBMine Distribution of Dihedral Angles for Residue {xray_phi_psi.res.values[0]} of Window {seq}')
    plt.xlabel('Phi')
    plt.ylabel('Psi')
    plt.legend()
    plt.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

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

    target = ins.find_target(phi_psi_dist, bw_method=bw_method)

    fig, ax = plt.subplots(figsize=(7,5))
    sns.kdeplot(data=phi_psi_dist, x='phi', y='psi', weights='weight', ax=ax, fill=True, color=colors[0], bw_method=bw_method)
    ax.scatter(xray_phi_psi_seq.phi, xray_phi_psi_seq.psi, marker='o', color=colors[1], label='X-ray', zorder=10)
    ax.scatter(phi_psi_pred.phi, phi_psi_pred.psi, marker='o', color=colors[2], label=f'{pred_name} Prediction', zorder=10)
    ax.scatter(phi_psi_alphafold.phi, phi_psi_alphafold.psi, marker='o', color=colors[4], label='AlphaFold', zorder=10)
    ax.scatter(target.phi, target.psi, color='red', marker='x', label='KDE Peak', zorder=20)
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
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z, cmap=cm)
    ax.zaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x * 10e3:.1f}'))
    
    ax.set_title(f'PDBMine Distribution of Dihedral Angles\nfor Residue {seq[ins.winsize_ctxt//2]} of Window {seq}', y=0.99, fontsize=12)
    ax.set_xlabel('Phi', fontsize=10, labelpad=10)
    ax.set_ylabel('Psi', fontsize=10, labelpad=10)
    ax.set_zlabel(r'Density $(10^{-3})$', fontsize=10, labelpad=10)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_box_aspect(aspect=None, zoom=0.8)
    # ax.dist = 5
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

def plot_da_for_seq(ins, seq, pred_id, pred_name, bw_method, axlims, fn, fill, scatter):
    pred_name = pred_name or pred_id[5:]
    bw_method = bw_method or ins.bw_method
    phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
    print(phi_psi_dist.describe())
    xray = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]
    pred = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == pred_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    preds = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq]
    alphafold = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == ins.alphafold_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    has_af = True
    if alphafold.shape[0] == 0:
        print('No AlphaFold data for this window')
        has_af = False

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

    target = ins.find_target(phi_psi_dist, bw_method=bw_method)

    # Distance to most common cluster
    da_xray = calc_da_for_one(target[['phi', 'psi']].values, xray[['phi','psi']].values[0])
    da_pred = calc_da_for_one(target[['phi', 'psi']].values, pred[['phi','psi']].values[0])
    da_preds = calc_da(target[['phi', 'psi']].values, preds[['phi','psi']].values)
    if has_af:
        da_alphafold = calc_da_for_one(target[['phi', 'psi']].values, alphafold[['phi','psi']].values[0])

    print(f'Ideal:\t ({target.phi:.02f}, {target.psi:.02f})')
    print(f'X-ray[{pos}]:\t ({xray.phi.values[0]:.02f}, {xray.psi.values[0]:.02f}), DA={da_xray:.02f}')
    print(f'{pred_name}[{pred.pos.values[0]}]:\t ({pred.phi.values[0]:.02f}, {pred.psi.values[0]:.02f}), DA={da_pred:.02f}')
    if has_af:
        print(f'AlphaFold[{alphafold.pos.values[0]}]:\t ({alphafold.phi.values[0]:.02f}, {alphafold.psi.values[0]:.02f}), DA={da_alphafold:.02f}')
    print('Other Predictions DA:\n', pd.DataFrame(da_preds).describe())

    fig, ax = plt.subplots(figsize=(9,7))
    sns.kdeplot(
        data=phi_psi_dist, 
        x='phi', y='psi', weights='weight',
        ax=ax, zorder=0, 
        fill=fill, color='black'
    )
    if scatter:
        sns.scatterplot(data=phi_psi_dist, x='phi', y='psi', ax=ax, hue='winsize', alpha=0.8, zorder=-1)
    ax.scatter(preds.phi, preds.psi, color='black', marker='o', s=5, alpha=0.2, label='All Other CASP-14 Predictions', zorder=1)
    ax.scatter(xray.iloc[0].phi, xray.iloc[0].psi, color=colors[1], marker='o', label='X-ray', zorder=10, s=100)
    ax.scatter(pred.phi, pred.psi,  color=colors[2], marker='o', label=pred_name, zorder=10, s=100)
    if has_af:
        ax.scatter(alphafold.phi, alphafold.psi, color=colors[4], marker='o', label='AlphaFold', zorder=10, s=100)
    ax.scatter(target.phi, target.psi, color='red', marker='X', label='PDBMine Target', s=200, linewidths=0.1)

    # dotted line from each point to mean
    ax.plot([xray.phi.values[0], target.phi], [xray.psi.values[0], target.psi], linestyle='dashed', color=colors[1], zorder=1, linewidth=1)
    ax.plot([pred.phi.values[0], target.phi], [pred.psi.values[0], target.psi], linestyle='dashed', color=colors[2], zorder=1, linewidth=1)
    if has_af:
        ax.plot([alphafold.phi.values[0], target.phi], [alphafold.psi.values[0], target.psi], linestyle='dashed', color=colors[4], zorder=1, linewidth=1)

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

def plot_res_vs_da(ins, pred_id, pred_name, highlight_res, limit_quantile, legend_loc, fn, text_loc):
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

    xtext = 0.845 if text_loc == 'right' else 0.017
    fig.text(xtext, 1.70, f'Pred RMSD={ins.results.loc[ins.results.Model == pred_id, "RMS_CA"].values[0]:.02f}', 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    fig.text(-0.02, 0.5, 'Dihedral Adherence of Residue', va='center', rotation='vertical', fontsize=12)
    fig.suptitle(f'Dihedral Adherence for each Residue of the Protein {ins.pdb_code}: Prediction vs X-Ray', fontsize=16)

    for highlight in highlight_res:
        for ax in axes:
            ax.axvspan(highlight[0], highlight[1], color='red', alpha=0.2)

    plt.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

    return both

def plot_da_vs_gdt(ins, axlims, fn):
    regr = linregress(ins.grouped_preds.gdt_pred, ins.grouped_preds.GDT_TS)

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=ins.grouped_preds, x='gdt_pred', y='GDT_TS', ax=ax, marker='o', s=25, edgecolor='b', legend=True)
    ax.plot(
        np.linspace(0, ins.grouped_preds.gdt_pred.max() + 5, 100), 
        regr.intercept + regr.slope * np.linspace(0, ins.grouped_preds.gdt_pred.max() + 5, 100), 
        color='red', lw=2, label='Regression Line'
    )

    ax.set_xlabel('Regression-Aggregated Dihedral Adherence Score', fontsize=14, labelpad=15)
    ax.set_ylabel('Prediction GDT', fontsize=14, labelpad=15)
    ax.set_title(r'Aggregated Dihedral Adherence vs GDT ($C_{\alpha}$) for each prediction', fontsize=16, pad=20)
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

def plot_heatmap(ins, fillna, fillna_row, fn):
    cmap = sns.color_palette("rocket", as_cmap=True)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ins.grouped_preds = ins.grouped_preds.sort_values('protein_id')
    ins.grouped_preds_da = ins.grouped_preds_da.sort_index()
    df = ins.grouped_preds_da.copy()
    df['gdt'] = ins.grouped_preds['GDT_TS'].values
    df = df.sort_values('gdt', ascending=False)
    # print(ins.grouped_preds[ins.grouped_preds.protein_id == ins.alphafold_id])
    X = df.iloc[:, :-1].values
    if fillna_row:
        X = np.where(np.isnan(X), np.nanmean(X,axis=0), X)
    if fillna:
        X[np.isnan(X)] = 0 # for entire column nan
    sns.heatmap(X, ax=ax, cmap=cmap)

    # af_idx = df.index.get_loc(ins.alphafold_id)
    # ax.set_yticks([af_idx + 0.5])
    # ax.set_yticklabels([f'Alpha\nFold'], fontsize=7)
    
    ax.set_ylabel('Prediction', fontsize=10)
    ax.set_xlabel('Residue Position', fontsize=10)
    ax.set_xticks([])
    ax.set_xticklabels([])

    cbar = ax.collections[0].colorbar
    cbar.set_label('Dihedral Adherence Magnitude', fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(f'Dihedral Adherence for each Residue of\nPredictions for the Protein {ins.pdb_code}', fontsize=12, pad=20)

    plt.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()


def plot_da_vs_gdt_simple(ins, axlims, fn):
    grouped_preds = ins.grouped_preds.dropna(subset=['log_da','GDT_TS'])
    # grouped_preds = grouped_preds[grouped_preds.GDT_TS < 40]
    # grouped_preds = grouped_preds[grouped_preds.GDT_TS < 30]
    # regr = linregress(grouped_preds.da, grouped_preds.GDT_TS)
    regr = linregress(grouped_preds.log_da, grouped_preds.GDT_TS)
    print(f'Slope: {regr.slope}, Intercept: {regr.intercept}', 'R-squared:', regr.rvalue**2)

    af = grouped_preds[grouped_preds.protein_id == ins.alphafold_id]
    xray_da = np.log10(ins.xray_phi_psi.da.mean())

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    # fig, ax = plt.subplots(figsize=(8, 6.5))
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(data=grouped_preds, x='log_da', y='GDT_TS', ax=ax, marker='o', s=25, edgecolor='b', legend=True)
    ax.scatter(af.log_da, af.GDT_TS, color='red', marker='x', label='AlphaFold', zorder=10)
    ax.scatter(xray_da, 100, color='green', marker='x', label='X-ray', zorder=10)
    ax.plot(
        np.linspace(0, grouped_preds.log_da.max() + 5, 100), 
        regr.intercept + regr.slope * np.linspace(0, grouped_preds.log_da.max() + 5, 100), 
        color='red', lw=2, label='Regression Line'
    )

    ax.set_xlabel(r'Log$_{10}$ Mean Dihedral Adherence', fontsize=12, labelpad=15)
    ax.set_ylabel('Prediction GDT', fontsize=12, labelpad=15)
    ax.set_title(r'GDT ($C_{\alpha}$) vs Total Dihedral Adherence for Each Prediction of '+ins.pdb_code, fontsize=14, pad=20)
    # ax.text(0.83, 0.10, r'$R^2$='+f'{regr.rvalue**2:.3f}', transform=ax.transAxes, fontsize=14,
    ax.text(0.98, 0.58, r'$R$='+f'{regr.rvalue:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    if regr.intercept > 0:
        s = f'y = {regr.slope:.1E}x + {regr.intercept:.1f}'
    else:
        s = f'y = {regr.slope:.1E}x - {-regr.intercept:.1f}'
    ax.text(.98,.69, s, transform=ax.transAxes, fontsize=12, color='red',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor='red', facecolor='white'))
    if axlims:
        ax.set_xlim(axlims[0][0], axlims[0][1])
        ax.set_ylim(axlims[1][0], axlims[1][1])
    else:
        ax.set_xlim(grouped_preds.log_da.min() - 0.1, grouped_preds.log_da.max() + 0.1)
        ax.set_ylim(-0.5, grouped_preds.GDT_TS.max() + 5)
        ax.set_ylim(-0.5, grouped_preds.GDT_TS.max() + 5)
        # ax.set_ylim(0, 105)

    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

    sns.reset_defaults()


def plot_res_vs_da_1plot(ins, pred_id, pred_name, highlight_res, limit_quantile, legend_loc, fn, text_loc, rmsds):
    # Plot xray vs prediction da for each residue of one prediction
    pred_name = pred_name or pred_id
    pred = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.protein_id == pred_id]
    pred = pred.drop_duplicates(subset=['seq_ctxt']) # for plotting only
    xray = ins.xray_phi_psi[['pos', 'seq_ctxt', 'da']]
    xray = xray.drop_duplicates(subset=['seq_ctxt']) # for plotting only

    both = pd.merge(pred, xray, how='inner', on=['seq_ctxt','seq_ctxt'], suffixes=('_pred','_xray'))
    both['da_diff'] = both['da_pred'] - both['da_xray']
    both['da_diff'] = np.abs(both['da_diff'])
    # both.loc[both.da_diff < 0, 'da_diff'] = 0
    both = both.rename(columns={'pos_pred':'pos'})
    # Add na rows for missing residues
    pos = np.arange(both.pos.min(), both.pos.max(), 1)
    both = both.set_index('pos').reindex(pos).reset_index()
    both['da_diff'] = both['da_diff'].fillna(0)

    # Print highest values
    print('Highest DA Differences:\n')
    print(both.sort_values('da_diff', ascending=False).head(10)[
        ['pos', 'pos_xray', 'seq_ctxt','da_pred','da_xray','da_diff']
    ].to_markdown(index=False))

    if limit_quantile:
        both[both.da_pred > both.da_pred.quantile(limit_quantile)] = np.nan
        both[both.da_xray > both.da_xray.quantile(limit_quantile)] = np.nan
        both[both.da_diff > both.da_diff.quantile(limit_quantile)] = np.nan
    
    fig, ax = plt.subplots(1, figsize=(10, 5), sharex=True)
    ax.plot(both.pos, both.da_diff, label=f'Difference:\n|{pred_name} - Xray|')
    ax.fill_between(
        x=both.pos, 
        y1=both['da_diff'].mean() + both['da_diff'].std(), 
        y2=both['da_diff'].mean() - both['da_diff'].std(), 
        color='tan', 
        alpha=0.4
    )
    ax.hlines(both['da_diff'].mean(), xmin=both.pos.min(), xmax=both.pos.max(), color='tan', label='Mean Difference', linewidth=0.75)
    ax.set_ylabel('Dihedral Adherence of Residue', fontsize=12)
    ax.set_xlabel('Residue Position in Chain', fontsize=12)
    ax.legend(loc=legend_loc)

    xtext = 0.825 if text_loc == 'right' else 0.017
    fig.text(xtext, .75, f'Total RMSD={ins.results.loc[ins.results.Model == pred_id, "RMS_CA"].values[0]:.02f}', 
             transform=ax.transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    if rmsds:
        for i,r in enumerate(rmsds):
            fig.text(r[0], r[1], r'RMSD$_' + f'{i+1}' + r'$='+f'{r[2]:.02f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    ax.set_title(f'Dihedral Adherence for each Residue of the Protein {ins.pdb_code}:\n Prediction - Xray', fontsize=16)

    ax.set_xlim(215, 390)
    ax.set_ylim(-60, 150)

    for highlight in highlight_res:
        ax.axvspan(highlight[0], highlight[1], color='red', alpha=0.2)
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.show()

    return both

def plot_dist_kde(ins, pred_id, percentile, fn):

    results = ins.grouped_preds.set_index('protein_id')
    xray_phi_psi = ins.xray_phi_psi.dropna().copy()
    xray_phi_psi['rmsd'] = 0
    # af_phi_psi = ins.phi_psi_predictions[ins.phi_psi_predictions.protein_id == ins.alphafold_id].dropna().copy()
    # af_phi_psi['rmsd'] = results.loc[ins.alphafold_id].RMS_CA

    other_id = ins.protein_ids[0]
    other_phi_psi = ins.phi_psi_predictions[ins.phi_psi_predictions.protein_id == other_id].dropna().copy()
    other_phi_psi['rmsd'] = results.loc[other_id].RMS_CA

    print(f'DA: Xray {xray_phi_psi.da.mean():.2f}, {pred_id} {other_phi_psi.da.mean():.2f}')
    # print(f'DA: Xray {xray_phi_psi.da.mean():.2f}, AF {af_phi_psi.da.mean():.2f}, {pred_id} {other_phi_psi.da.mean():.2f}')
    print(f'RMSD: Xray {xray_phi_psi.rmsd.mean():.2f}, {pred_id} {other_phi_psi.rmsd.mean():.2f}')
    # print(f'RMSD: Xray {xray_phi_psi.rmsd.mean():.2f}, AF {af_phi_psi.rmsd.mean():.2f}, {pred_id} {other_phi_psi.rmsd.mean():.2f}')

    df = pd.concat([
        xray_phi_psi, 
        # af_phi_psi.drop('da_na', axis=1),
        other_phi_psi.drop('da_na', axis=1)
    ])

    def get_probs(x, da):
        kde = gaussian_kde(da)
        p = kde(x)
        c = np.cumsum(p) / np.sum(p)
        peak = x[np.argmax(p)]
        return p, c, peak

    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 5))
    x = np.linspace(0, df.da.max(), 1000)

    def plot(df, label, color):
        p, c, peak = get_probs(x, df.da)
        axes[0].plot(x, p, color=color, label=f'{label} [RMSD={df.rmsd.iloc[0]}], peak at {peak:.2f}')
        axes[0].fill_between(x, 0, p, alpha=0.2, color=color)
        axes[0].vlines(peak, 0, p.max(), color=color)
        perc = x[np.argmax(c > percentile)]
        suffix = 'st' if ((percentile*100) % 10 == 1 and percentile != 0.11) else 'nd' if ((percentile*100) % 10 == 2 and percentile != 0.12) else 'th'
        axes[1].plot(x, c, color=color, label=f'{label}, {int(percentile*100)}{suffix} percentile at {perc:.2f}')
        axes[1].vlines(perc, 0, 1, color=color)
        axes[1].fill_between(x, 0, c, alpha=0.2)

    colors = sns.color_palette("tab10")
    plot(xray_phi_psi, 'Xray', colors[0])
    # plot(af_phi_psi, 'AF', colors[1])
    plot(other_phi_psi, other_id[7:10], colors[2])
    axes[0].legend()
    axes[0].set_ylabel('Density')
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('Dihedral Adherence')
    axes[1].set_ylabel('Cumulative density')
    # axes[0].set_xlim(0, 5000)
    # axes[1].set_xlim(0, 5000)
    fig.suptitle('Dihedral adherence distribution')
    plt.show()


def plot_across_window_clusters(ins, seq_ctxt, plot_xrays, plot_afs, n_cluster_lines=50):
    _, info = get_phi_psi_dist(ins.queries, seq_ctxt)
    for j in info:
        print(f'\tWin {j[0]}: {j[1]} - {j[2]} samples')
    phi_psi_dist, phi_psi_dist_v = get_combined_phi_psi_dist(ins, seq_ctxt)

    q = ins.queries[0]
    xrays = get_xrays_window(ins, q, seq_ctxt)
    preds = get_preds_window(ins, q, seq_ctxt)
    afs = get_afs_window(ins, q, seq_ctxt)

    precomputed_dists = precompute_dists(phi_psi_dist_v)
    n_clusters, clusters = find_clusters(precomputed_dists, 20)
    precomputed_dists, phi_psi_dist_v, clusters = filter_precomputed_dists(precomputed_dists, phi_psi_dist_v, clusters)

    def plot(q, seq_ctxt, xrays, afs, clusters, phi_psi_dist, precomputed_dists):
        n_cluster_plot = 10
        n_clusters = len(np.unique(clusters))
        xrays = xrays.reshape(2, -1)
        afs = afs.reshape(2, -1)
        print(pd.Series(clusters).value_counts())

        cluster_points = phi_psi_dist.groupby(clusters).count().sort_values('phi_0', ascending=False).index.values
        clusters_plot = cluster_points[:n_cluster_plot]
        medoids = []
        for cluster in cluster_points:
            medoid = get_cluster_medoid(phi_psi_dist, precomputed_dists, clusters, cluster)
            medoids.append(medoid)
        medoids = np.array(medoids)

        colors = sns.color_palette('Dark2', n_clusters)
        fig, axes = plt.subplots(len(clusters_plot), q.winsize, figsize=(16, min(n_cluster_plot, len(clusters_plot))*4), sharey=True, sharex=True)
        for i,axrow in enumerate(axes):
            for j, ax in enumerate(axrow):
                cluster_dist = phi_psi_dist[clusters == clusters_plot[i]]

                sns.scatterplot(data=phi_psi_dist[clusters != clusters_plot[i]], x=f'phi_{j}', y=f'psi_{j}', ax=ax, label='Other Clusters', color='tab:blue', alpha=0.5)
                sns.scatterplot(data=cluster_dist, x=f'phi_{j}', y=f'psi_{j}', ax=ax, label=f'Cluster {clusters_plot[i]}', color=colors[i])
                if plot_xrays:
                    ax.scatter(xrays[0,j], xrays[1,j], color='tab:red', marker='X', label='X-ray', zorder=1000)
                if plot_afs:
                    ax.scatter(afs[0,j], afs[1,j], color='tab:orange', marker='X', label='AF', zorder=1000)
                # ax.scatter(pred[0,j], pred[1,j], color='tab:orange', marker='X', label=pred_id, zorder=1000)
                ax.scatter(medoids[i].reshape(2,-1)[0,j], medoids[i].reshape(2,-1)[1,j], color='black', marker='X', label='Cluster Centroid', zorder=1000)

                def add_conn(xyA, xyB, color, lw, **kwargs):
                    con = ConnectionPatch(
                        xyA=xyA, 
                        xyB=xyB, 
                        coordsA="data", coordsB="data", 
                        axesA=axrow[j], axesB=axrow[j+1], 
                        color=color, lw=lw, linestyle='--', alpha=0.5, **kwargs
                    )
                    fig.add_artist(con)
                if j < q.winsize - 1:
                    # TODO draw lines for 50 points closest to centroid
                    for k, row in cluster_dist.sample(min(cluster_dist.shape[0], n_cluster_lines)).iterrows():
                        add_conn((row[f'phi_{j}'], row[f'psi_{j}']), (row[f'phi_{j+1}'], row[f'psi_{j+1}']), colors[i], 1)
                    if plot_xrays:
                        add_conn((xrays[0,j], xrays[1,j]), (xrays[0,j+1], xrays[1,j+1]), 'tab:red', 5, zorder=100)
                    if plot_afs:
                        add_conn((afs[0,j], afs[1,j]), (afs[0,j+1], afs[1,j+1]), 'tab:orange', 5, zorder=100)
                    # add_conn((pred[0,j], pred[1,j]), (pred[0,j+1], pred[1,j+1]), 'tab:orange', 5, zorder=100)
                    add_conn((medoids[i].reshape(2,-1)[0,j], medoids[i].reshape(2,-1)[1,j]), (medoids[i].reshape(2,-1)[0,j+1], medoids[i].reshape(2,-1)[1,j+1]), 'black', 5, zorder=100)

                ax.set_xlim(-180, 180)
                ax.set_ylim(-180, 180)
                ax.set_xlabel('')
                if j == q.winsize - 1:
                    ax.legend()
                else:
                    ax.legend().remove()
                if j == 0:
                    ax.set_ylabel(f'Cluster {clusters_plot[i]} [{cluster_dist.shape[0]}]')
        fig.supxlabel('Phi')
        fig.supylabel('Psi')
        # fig.suptitle(
        #     # f'Clustered Phi/Psi Distributions for {seq_ctxt} in protein {da.casp_protein_id}: N={n_points} Silhouette Score: {sil_score:.2f}, X-ray Score [Cluster {nearest_cluster}]: {xray_sil:.2f}, Prediction Score [Cluster {nearest_cluster_pred}]: {pred_sil:.2f}', 
        #     f'Clustered Phi/Psi Distributions for {seq_ctxt} in protein {da.casp_protein_id}: N={n_points} ({n_unassigned} unassigned) Silhouette Score: {sil_score:.2f}, X-ray Score [Cluster {nearest_cluster}]: {xray_maha:.2f}', 
        #     y=1.01
        # )
        plt.tight_layout()
        plt.show()
    
    plot(q, seq_ctxt, xrays, afs, clusters, phi_psi_dist_v, precomputed_dists)

def plot_across_window_cluster_medoids(ins, seq_ctxt, plot_xrays=False, plot_afs=False, verbose=False, mode_scatter=False, cse=30, fn=None):
    _, info = get_phi_psi_dist(ins.queries, seq_ctxt)
    for j in info:
        print(f'\tWin {j[0]}: {j[1]} - {j[2]} samples')
    phi_psi_dist, phi_psi_dist_v = get_combined_phi_psi_dist(ins, seq_ctxt)

    q = ins.queries[0]
    xrays = get_xrays_window(ins, q, seq_ctxt)
    xrays = xrays.reshape(2, -1)
    # preds = get_preds_window(ins, q, seq_ctxt)
    # afs = get_afs_window(ins, q, seq_ctxt)

    precomputed_dists = precompute_dists(phi_psi_dist_v)
    n_clusters, clusters = find_clusters(precomputed_dists, 20, cse)
    if verbose:
        print(f'Number of clusters: {n_clusters}')
    if n_clusters == 0:
        print('No clusters found')
        return
    precomputed_dists, phi_psi_dist_v, clusters = filter_precomputed_dists(precomputed_dists, phi_psi_dist_v, clusters)
    print(phi_psi_dist_v.shape)

    def plot(q, phi_psi_dist, precomputed_dists, clusters, seq_ctxt):
        seq = q.get_subseq(seq_ctxt)
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        medoids = []
        for cluster in unique_clusters:
            medoid = get_cluster_medoid(phi_psi_dist, precomputed_dists, clusters, cluster)
            medoids.append(medoid)
            if verbose:
                print(f'Cluster {cluster} has {cluster_counts[cluster]} members and medoid {medoid}')
        medoids = np.array(medoids).reshape(unique_clusters.shape[0], 2, -1)

        fig, axes = plt.subplots(1, q.winsize, figsize=(q.winsize*3,3.5), sharey=True)
        colors = sns.color_palette('Dark2', len(unique_clusters))
        # if xrays is not None:
            # xrays = xrays.reshape(2, -1)
        legend_handles = []
        legend_labels = []
        for i in range(q.winsize):
            def add_conn(xyA, xyB, color, lw, **kwargs):
                con = ConnectionPatch(
                    xyA=xyA, 
                    xyB=xyB, 
                    coordsA="data", coordsB="data", 
                    axesA=axes[i], axesB=axes[i+1], 
                    color=color, lw=lw, linestyle='--', alpha=0.5, **kwargs
                )
                fig.add_artist(con)
            # KDE of all points together
            if not mode_scatter:
                sns.kdeplot(data=phi_psi_dist, x=f'phi_{i}', y=f'psi_{i}', ax=axes[i], color='black', fill=True)
            for j,cluster in enumerate(unique_clusters):
                if mode_scatter:
                    sns.scatterplot(
                        data=phi_psi_dist[clusters==cluster], x=f'phi_{i}', y=f'psi_{i}', ax=axes[i], 
                        color=colors[cluster], label=f'Cluster {cluster} [{cluster_counts[j]} members]', alpha=0.3, zorder=j
                    )
                else:
                    # individual KDEs
                    # sns.kdeplot(
                    #     data=phi_psi_dist[clusters==cluster], x=f'phi_{i}', y=f'psi_{i}', ax=axes[i], 
                    #     color=colors[cluster], zorder=j, levels=5
                    # )
                    axes[i].scatter(
                        x=medoids[j][0,i], y=medoids[j][1,i], color=colors[cluster], marker='X', 
                        label=f'Cluster {cluster} [{cluster_counts[j]} members]', zorder=100
                    )
                if i < q.winsize - 1:
                    add_conn((medoids[j][0,i], medoids[j][1,i]), (medoids[j][0,i+1], medoids[j][1,i+1]), colors[cluster], 2.5, zorder=100)
            if plot_xrays:
                axes[i].scatter(xrays[0,i], xrays[1,i], color='red', marker='X', label='X-ray', s=150, zorder=150)
                if i < q.winsize - 1:
                    add_conn((xrays[0,i], xrays[1,i]), (xrays[0,i+1], xrays[1,i+1]), 'red', 2.5, zorder=100)
            if i == 0:
                legend_handles, legend_labels = axes[i].get_legend_handles_labels()
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].set_xlim(-180,180)
            axes[i].set_ylim(-180,180)
            axes[i].set_title(f'Residue {seq[i]}')
            axes[i].legend().remove()
            axes[i].set_xticks(np.arange(-180, 181, 45), minor=True)
            axes[i].set_yticks(np.arange(-180, 181, 45), minor=True)
            axes[i].grid(True, linestyle="--", alpha=0.6)
            axes[i].set_xticks(np.arange(-180, 181, 90))
            axes[i].set_yticks(np.arange(-180, 181, 90))
            axes[i].axhline(0, color='black', linewidth=0.75, zorder=0)
            axes[i].axvline(0, color='black', linewidth=0.75, zorder=0)
        
        if plot_xrays:
            fig.legend(
                handles=legend_handles, 
                labels=legend_labels, 
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.1), 
                ncol=min(len(unique_clusters)+1, 6)
            )
        else:
            fig.legend(
                handles=legend_handles, 
                labels=legend_labels, 
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.1), 
                ncol=min(len(unique_clusters), 5)
            )
        fig.supxlabel('Phi')
        fig.supylabel('Psi')
        fig.suptitle(f'Clustered Phi/Psi Distributions Queried from PDBMine for Sequence {seq}', y=1.01)
        plt.tight_layout()
        if fn:
            plt.savefig(fn, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
    plot(q, phi_psi_dist_v, precomputed_dists, clusters, seq_ctxt)