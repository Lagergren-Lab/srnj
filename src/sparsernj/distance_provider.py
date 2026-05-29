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

    To integrate a custom distance estimator, subclass DistanceProvider, set ``taxa``
    in ``__init__``, and implement ``_compute_triplet``.  All caching, call counting,
    ``dist()``, and ``get_dms()`` are handled automatically by the base class.

    Example (see README for the full cellmates integration example)::

        class MyProvider(DistanceProvider):
            def __init__(self, cells, estimator):
                super().__init__(taxa=list(range(len(cells))))
                self.cells = cells
                self.estimator = estimator

            def _compute_triplet(self, taxon1, taxon2):
                lca, adm_fwd, adm_rev = self.estimator.triplet_distance(
                    self.cells[taxon1], self.cells[taxon2]
                )
                return lca, adm_fwd, adm_rev
    """

    def __init__(self, taxa: Optional[List] = None, track_distinct_calls: bool = False):
        self._taxa = taxa
        self._track_distinct_calls = track_distinct_calls
        self._seen_pairs: set = set()
        # per-pair distance cache shared by dist() / get_dms()
        self._cache_lca: dict = {}
        self._cache_adm: dict = {}
        if taxa is not None:
            self._taxon_to_index = {label: idx for idx, label in enumerate(taxa)}
        else:
            self._taxon_to_index = {}

    @property
    def taxa(self) -> List:
        """Taxon identifiers (usually integer indices)."""
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
        if self._track_distinct_calls:
            key = frozenset((taxon1, taxon2))
            assert len(key) == 2, f"Taxon pair ({taxon1}, {taxon2}) must be distinct for tracking distinct calls."
            if key not in self._seen_pairs:
                self._seen_pairs.add(key)

    def _compute_triplet(self, taxon1, taxon2) -> Tuple[float, float, float]:
        """Return ``(lca, adm_fwd, adm_rev)`` for the given taxon labels.

        Override this in a subclass to plug in a custom distance estimator.
        ``dist()`` and ``get_dms()`` call this method and cache the result, so only
        the raw per-pair computation needs to be provided here.

        Returns:
            lca     : LCA distance C[taxon1, taxon2] (symmetric)
            adm_fwd : asymmetric distance A[taxon1, taxon2] (taxon1 → their LCA)
            adm_rev : asymmetric distance A[taxon2, taxon1] (taxon2 → their LCA)
        """
        raise NotImplementedError

    def dist(self, taxon1, taxon2) -> Tuple[float, float]:
        """Return ``(lca_distance, adm_fwd)`` for taxa taxon1 and taxon2 (labels)."""
        self._record_call(taxon1, taxon2)
        key = frozenset((taxon1, taxon2))
        if key not in self._cache_lca:
            dlca, dadm, dadm_rev = self._compute_triplet(taxon1, taxon2)
            self._cache_lca[key] = dlca
            self._cache_adm[(taxon1, taxon2)] = dadm
            self._cache_adm[(taxon2, taxon1)] = dadm_rev
        return self._cache_lca[key], self._cache_adm[(taxon1, taxon2)]

    def get_dms(self, taxa: List):
        """Build LCA and ADM distance matrices for the provided taxa list (labels).

        Uses cached ``dist()`` calls so repeated queries reuse cached values.
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

    def get_dist_matrix(self, taxa: List):
        """Return the symmetric pairwise distance matrix for ``taxa``.

        If one element of ``taxa`` is not in ``self.taxa``, it is treated as the root
        and its row/column is filled with per-taxon root distances.
        """
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
        D = np.zeros((len(taxa), len(taxa)))
        D[:len(taxa_no_root), :len(taxa_no_root)] = dist_matrix
        if extra == 1:
            D[len(taxa_no_root), :len(taxa_no_root)] = root_dist
            D[:len(taxa_no_root), len(taxa_no_root)] = root_dist
            order = list(range(len(taxa_no_root)))
            order.insert(root_idx, len(taxa_no_root))
            D = D[np.ix_(order, order)]
        return D

    def get_dist_matrix_with_root(self, taxa: List) -> tuple:
        return get_pairwise_from_lca(*self.get_dms(taxa))


class LazyDistanceProvider(DistanceProvider):
    """
    Lazily computes distances on demand and caches results.

    Constructor arguments:
    - data: any object passed to ``compute_distance_fn`` (commonly a 3D numpy array
      where ``data[i, j, 0]`` is lca and ``data[i, j, 1]`` is adm).
    - taxa: optional list of taxon ids (defaults to ``range(data.shape[0])``).
    - compute_distance_fn: optional ``f(data, i, j) -> (lca, adm, adm_rev)``.
      If not provided, a default reading the 3D-array layout above is used.
    - track_distinct_calls: if True, count only distinct unordered taxon pairs.
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
        if compute_distance_fn is None:
            self._compute_distance_fn = self._default_compute_distance
        else:
            self._compute_distance_fn = compute_distance_fn

    def _default_compute_distance(self, data, i: int, j: int):
        return data[i, j, 0], data[i, j, 1], data[j, i, 1]

    def _compute_triplet(self, taxon1, taxon2) -> Tuple[float, float, float]:
        idx_i = self._get_index(taxon1)
        idx_j = self._get_index(taxon2)
        return self._compute_distance_fn(self.data, idx_i, idx_j)


class FixedDistanceProvider(DistanceProvider):
    """
    Fixed (fully materialized) distance provider.  The full LCA and ADM matrices
    are provided at construction time and ``get_dms(taxa)`` returns slices of them.
    """

    def __init__(
        self,
        lca_dm: Optional[np.ndarray] = None,
        adm_dm: Optional[np.ndarray] = None,
        taxa: Optional[List] = None,
        track_distinct_calls: bool = False,
    ):
        if taxa is None:
            taxa = list(range(lca_dm.shape[0]))
        super().__init__(taxa, track_distinct_calls)
        self._full_lca = lca_dm
        self._full_adm = adm_dm

    def _compute_triplet(self, taxon1, taxon2) -> Tuple[float, float, float]:
        i = self._get_index(taxon1)
        j = self._get_index(taxon2)
        return float(self._full_lca[i, j]), float(self._full_adm[i, j]), float(self._full_adm[j, i])

    def get_dms(self, taxa: List):
        """Vectorized slice — faster than the per-pair loop in the base class."""
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
