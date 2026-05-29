import itertools

import numpy as np
from typing import Callable, List, Optional, Tuple

def get_pairwise_from_lca(C, A):
    # compute pairwise distance matrix D from LCA distance matrix C and asymmetric distance matrix A
    # take the maximum of the two asymmetric distances to the root as the root distance for each cell
    n = A.shape[0]
    D = np.zeros((n, n))
    root_dist = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i < j:
                root_dist[i] = max(root_dist[i], C[i, j] + A[i, j])
                root_dist[j] = max(root_dist[j], C[i, j] + A[j, i])
            D[i, j] = A[i, j] + A[j, i]
            D[j, i] = D[i, j]
    return D, root_dist

class DistanceProvider:
    """
    Base interface for distance providers.

    Subclasses must implement `dist(i, j)` returning (lca, adm)
    and `get_dms(taxa)` returning (lca_dm, adm_dm) for the requested taxa list.
    """

    def __init__(self, taxa: Optional[List] = None, track_distinct_calls: bool = False):
        self._taxa = taxa
        # whether to count only distinct unordered taxon-pair requests
        self._track_distinct_calls = track_distinct_calls
        # track unordered taxon pairs that have been requested (used only if flag set)
        self._seen_pairs = set()
        # Cache label->index for efficient lookup
        if taxa is not None:
            self._taxon_to_index = {label: idx for idx, label in enumerate(taxa)}
        else:
            self._taxon_to_index = {}

    @property
    def taxa(self) -> List:
        """Taxon identifiers (usually integer indices). If None, callers should
        pass explicit taxa lists to get_dms.
        """
        return self._taxa

    @property
    def num_calls(self) -> int:
        return len(self._seen_pairs)

    def reset_call_count(self):
        self._seen_pairs.clear()

    def _get_index(self, label):
        if self._taxa is None:
            raise ValueError("Taxa list is not set for this DistanceProvider.")
        try:
            return self._taxon_to_index[label]
        except KeyError:
            raise ValueError(f"Taxon label {label} not found in taxa list.")

    def _record_call(self, taxon1, taxon2):
        """Record a call for the unordered pair (taxon1, taxon2).
        If tracking distinct calls is enabled, increment only the first time this pair is seen.
        Otherwise increment for every invocation.
        """
        if self._track_distinct_calls:
            key = frozenset((taxon1, taxon2))
            assert len(key) == 2, f"Taxon pair ({taxon1}, {taxon2}) must be distinct for tracking distinct calls."
            if key not in self._seen_pairs:
                self._seen_pairs.add(key)

    def dist(self, taxon1, taxon2) -> Tuple[float, float]:
        """Return (lca_distance, adm_distance) for taxa taxon1 and taxon2 (labels).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_dms(self, taxa: List):
        """Return (lca_dm, adm_dm) for the provided list of taxa (labels).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_dist_matrix(self, taxa: List):
        """
        Return the single standard symmetric distance matrix for the provided taxa list (labels).
        If taxa contains one that is not in self.taxa, treat it as the root and compute distances accordingly.
        The distance matrix will always have shape (len(taxa), len(taxa)) and be ordered according to the input taxa list.
        """
        # if one taxa is not in self.taxa, we treat it as the root
        extra = len(set(taxa).difference(set(self.taxa)))
        if extra > 1:
            raise ValueError(f"Some taxa (n={extra}) do not match provided taxa")
        taxa_no_root = taxa
        root_idx = None
        if extra == 1:
            taxa_no_root = []
            for i, t in enumerate(taxa):
                if t in self.taxa:
                    taxa_no_root.append(t)
                else:
                    root_idx = i

        dist_matrix, root_dist = self.get_dist_matrix_with_root(taxa_no_root)
        D = np.zeros((len(taxa), len(taxa))) # initialize full distance matrix
        D[:len(taxa_no_root), :len(taxa_no_root)] = dist_matrix
        if extra == 1:
            D[len(taxa_no_root), :len(taxa_no_root)] = root_dist
            D[:len(taxa_no_root), len(taxa_no_root)] = root_dist
            # build index order: insert root_idx into the sequence
            order = list(range(len(taxa_no_root)))
            order.insert(root_idx, len(taxa_no_root))  # insert root's current index at desired position
            D = D[np.ix_(order, order)]
        return D

    def get_dist_matrix_with_root(self, taxa: List) -> tuple[np.ndarray, np.ndarray]:
        # sum adm + adm.T and set diagonal to zero. add root distance for last
        return get_pairwise_from_lca(*self.get_dms(taxa))


class LazyDistanceProvider(DistanceProvider):
    """
    Lazily computes distances on demand and caches results.

    Constructor arguments:
    - data: any object that the compute function can use (commonly a 3D numpy array
      where data[i, j, 0] is lca and data[i, j, 1] is adm).
    - taxa: optional list of taxon ids
    - compute_distance_fn: optional function f(data, i, j) -> (lca, adm, adm_rev)
      If not provided, a default that reads the 3D-array layout above is used.
    - track_distinct_calls: if True, count only distinct unordered taxon pairs
    """

    def __init__(
        self,
        data,
        taxa: Optional[List] = None,
        compute_distance_fn: Optional[Callable] = None,
        track_distinct_calls: bool = False,
    ):
        if taxa is None:
            taxa = list(range(data.shape[0]))
        super().__init__(taxa, track_distinct_calls)
        self.data = data
        self._cache_lca = {}
        self._cache_adm = {}
        # compute_distance_fn(data, i, j) -> (lca, adm, adm_rev)
        if compute_distance_fn is None:
            self._compute_distance_fn = self._default_compute_distance
        else:
            self._compute_distance_fn = compute_distance_fn

    def _default_compute_distance(self, data, i: int, j: int):
        # increment call counter and assume 3D array layout as before
        return data[i, j, 0], data[i, j, 1], data[j, i, 1]

    def dist(self, taxon1, taxon2) -> Tuple[float, float]:
        # taxon1, taxon2 are labels; map to indices
        # record the call (distinct or not is handled by base)
        self._record_call(taxon1, taxon2)
        idx_i = self._get_index(taxon1)
        idx_j = self._get_index(taxon2)
        key = frozenset((taxon1, taxon2))
        if key not in self._cache_lca:
            dlca, dadm, dadm_rev = self._compute_distance_fn(self.data, idx_i, idx_j)
            self._cache_lca[key] = dlca
            self._cache_adm[(taxon1, taxon2)] = dadm
            self._cache_adm[(taxon2, taxon1)] = dadm_rev
        return self._cache_lca[key], self._cache_adm[(taxon1, taxon2)]

    def get_dms(self, taxa: List):
        """Build LCA and ADM distance matrices for the provided taxa list (labels).

        This method uses the cached `dist` method so repeated calls reuse cached values.
        """
        n = len(taxa)
        lca_dm = np.zeros((n, n))
        adm = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                lca_d, ad = self.dist(taxa[a], taxa[b])
                lca_dm[a, b] = lca_d
                lca_dm[b, a] = lca_d
                adm[a, b] = ad
                adm[b, a] = self._cache_adm[(taxa[b], taxa[a])]
        return lca_dm, adm


class FixedDistanceProvider(DistanceProvider):
    """
    Fixed (fully materialized) distance provider. The full LCA and ADM matrices
    are provided at construction time and `get_dms(taxa)` returns slices of them.
    """

    def __init__(
        self,
        lca_dm: Optional[np.ndarray] = None,
        adm_dm: Optional[np.ndarray] = None,
        taxa: Optional[List] = None,
        track_distinct_calls: bool = False,
    ):
        # If taxa is None, assume the matrices are indexed 0..n-1
        if taxa is None:
            taxa = list(range(lca_dm.shape[0]))
        super().__init__(taxa, track_distinct_calls)
        self._full_lca = lca_dm
        self._full_adm = adm_dm

    def dist(self, taxon1, taxon2) -> Tuple[float, float]:
        """Return distances using the full matrices. `taxon1` and `taxon2` are labels.
        Counting is delegated to the base class helper.
        """
        # record the call (distinct or not is handled by base)
        self._record_call(taxon1, taxon2)
        i = self._get_index(taxon1)
        j = self._get_index(taxon2)
        return float(self._full_lca[i, j]), float(self._full_adm[i, j])

    def get_dms(self, taxa: List):
        """Return slices of the stored full matrices for the requested taxa list (labels).

        The returned matrices are in the same order as `taxa`.
        """
        for a, b in itertools.combinations(taxa, 2):
            self._record_call(a, b)
        taxa_idx = [self._get_index(t) for t in taxa]
        idx = np.ix_(taxa_idx, taxa_idx)
        return self._full_lca[idx].copy(), self._full_adm[idx].copy()


def get_lca_from_pairwise(med2_D, med2_rootdist):
    # compute LCA distance matrix C and asymmetric distance matrix A from pairwise distance matrix D and root distances
    n = med2_D.shape[0]
    C = np.zeros((n, n))
    A = np.zeros((n, n))

    for i in range(n):
        C[i, i] = med2_rootdist[i]
    for i in range(n):
        for j in range(i+1, n):
            Cij = (med2_rootdist[i] + med2_rootdist[j] - med2_D[i, j]) / 2.
            C[j, i] = C[i, j] = Cij
            A[i, j] = med2_rootdist[i] - Cij
            A[j, i] = med2_rootdist[j] - Cij
    return C, A
