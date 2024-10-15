

def get_phi_psi_dist_window(q, seq_ctxt):
    seq = q.get_subseq(seq_ctxt)
    phi_psi_dist = q.results_window[q.results_window.seq == seq]
    phi_psi_dist = phi_psi_dist[['match_id', 'window_pos', 'phi', 'psi']].pivot(index='match_id', columns='window_pos', values=['phi', 'psi'])
    phi_psi_dist.columns = [f'{c[0]}_{c[1]}' for c in phi_psi_dist.columns.to_flat_index()]
    phi_psi_dist = phi_psi_dist.dropna(axis=0)
    return phi_psi_dist

def get_xrays_window(ins, q, seq_ctxt, return_df=False):
    center_idx = q.get_center_idx_pos()
    xray_pos = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq_ctxt].pos.iloc[0]
    xrays = ins.xray_phi_psi[(ins.xray_phi_psi.pos >= xray_pos-center_idx) & (ins.xray_phi_psi.pos < xray_pos-center_idx+q.winsize)].copy()
    xray_point = np.concatenate([xrays['phi'].values, xrays['psi'].values])
    if return_df:
        return xray_point, xrays
    return xray_point

def get_afs_window(ins, q, seq_ctxt, return_df=False):
    center_idx = q.get_center_idx_pos()
    af_pos = ins.af_phi_psi[ins.af_phi_psi.seq_ctxt == seq_ctxt].pos.iloc[0]
    afs = ins.af_phi_psi[(ins.af_phi_psi.pos >= af_pos-center_idx) & (ins.af_phi_psi.pos < af_pos-center_idx+q.winsize)].copy()
    af_point = np.concatenate([afs['phi'].values, afs['psi'].values])
    if return_df:
        return af_point, afs
    return af_point


def get_preds_window(ins, q, seq_ctxt):
    center_idx = q.get_center_idx_pos()
    pred_pos = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].pos.unique()
    if len(pred_pos) == 0:
        print(f"No predictions for {seq_ctxt}")
    if len(pred_pos) > 1:
        print(f"Multiple predictions for {seq_ctxt}")
        raise ValueError
    pred_pos = pred_pos[0]
    preds = ins.phi_psi_predictions[(ins.phi_psi_predictions.pos >= pred_pos-center_idx) & (ins.phi_psi_predictions.pos < pred_pos-center_idx+q.winsize)].copy()
    preds = preds[['protein_id', 'pos', 'phi', 'psi']].pivot(index='protein_id', columns='pos', values=['phi', 'psi'])
    preds.columns = [f'{c[0]}_{c[1]-pred_pos+center_idx}' for c in preds.columns.to_flat_index()]
    preds = preds.dropna(axis=0)
    return preds

def precompute_dists(phi_psi_dist):
    def diff(x1, x2):
            d = np.abs(x1 - x2)
            return np.minimum(d, 360-d)
    precomputed_dists = np.linalg.norm(diff(phi_psi_dist.values[:,np.newaxis], phi_psi_dist.values), axis=2)
    return precomputed_dists

def get_cluster_medoid(phi_psi_dist, precomputed_dists, c, q):
    d = precomputed_dists[phi_psi_dist.cluster == c][:,phi_psi_dist.cluster == c]
    return phi_psi_dist[phi_psi_dist.cluster == c].iloc[d.sum(axis=1).argmin(), :q.winsize*2].values

def estimate_icov(q, phi_psi_dist_c, cluster_medoid):
    # estimate covariance matrix
    cluster_points = phi_psi_dist_c.iloc[:,:q.winsize*2].values
    diffs = diff(cluster_points, cluster_medoid)

    cov = (diffs[...,np.newaxis] @ diffs[:,np.newaxis]).sum(axis=0) / (diffs.shape[0] - 1)
    cov = cov + np.eye(cov.shape[0]) * 1e-6 # add small value to diagonal to avoid singular matrix
    if np.any(cov <= 0):
        print("Non-positive covariance matrix")
        return None
    if np.any(cov.diagonal() < 1):
        print("Covariance matrix less than 1")
        return None
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    if np.any(eigenvalues < 0):
        print("Negative eigenvalues - non-positive semi-definite covariance matrix")
        return None
    icov = inv(cov)
    return icov

def get_target_cluster(q, phi_psi_dist, point):
    d = np.linalg.norm(diff(point[np.newaxis,:], phi_psi_dist.iloc[:,:q.winsize*2].values), axis=1)
    d = pd.DataFrame({'d': d, 'c': phi_psi_dist.cluster})
    nearest_cluster = d.groupby('c').d.mean().idxmin()
    return nearest_cluster