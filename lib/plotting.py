import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
def plot_one_dist(ins, seq, pred_id, pred_name=None, axlims=None, bw_method=None):
    pred_name = pred_name or pred_id
    inner_seq = ins.get_subseq(seq)
    phi_psi_dist = ins.phi_psi_mined[ins.phi_psi_mined.seq == inner_seq].copy()
    phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt[ins.phi_psi_mined_ctxt.seq == seq].copy()
    phi_psi_alpha = ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == pred_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    xray_phi_psi_seq = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq]

    kdews = [1,128]
    phi_psi_dist['weight'] = kdews[0]
    phi_psi_ctxt_dist['weight'] = kdews[1]
    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])
    print(phi_psi_dist.groupby('weight').count().iloc[:,0].values)

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
