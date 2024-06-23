def plot_kdepeaks(ins, seq):
    # Find probability of each point
    if af.shape[0] == 0:
        print('\tNo AlphaFold prediction - Using ordinary KDE')
        return find_kdepeak(phi_psi_dist, bw_method)

    af = af[['phi', 'psi']].values[0]

    phi_psi_dist = phi_psi_dist.loc[~phi_psi_dist[['phi', 'psi']].isna().any(axis=1)]

    # Find two clusters
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(phi_psi_dist[['phi','psi']])
    phi_psi_dist['cluster'] = kmeans.labels_

    if (phi_psi_dist.groupby('cluster').size() < 2).any():
        print('\tOne cluster has less than 2 points - Using ordinary KDE')
        return find_kdepeak(phi_psi_dist, bw_method)
    # find kdepeak for each cluster and entire dist
    kdepeak = find_kdepeak(phi_psi_dist, bw_method)
    kdepeak_c1 = find_kdepeak(phi_psi_dist[phi_psi_dist.cluster == 0], bw_method)
    kdepeak_c2 = find_kdepeak(phi_psi_dist[phi_psi_dist.cluster == 1], bw_method)

    # Choose peak that is closest to AlphaFold prediction
    targets = np.array([kdepeak.values, kdepeak_c1.values, kdepeak_c2.values])
    dists = calc_da(af, targets)
    if dists.argmin() == 0:
        print('\tKDEPEAK: Using kdepeak of entire distribution')
    else:
        print(f'\tKDEPEAK: Using kdepeak of cluster')
    target = targets[dists.argmin()]
    target = pd.Series({'phi': target[0], 'psi': target[1]})

    return target