import numpy as np
from itertools import groupby
from collections import OrderedDict, defaultdict
from ase.io import read

class Events:
    """
    Hold and manage KMC jump events derived from site geometry.
    """

    def __init__(self, structure_file="./sites.vasp", path=None):
        """
        Parameters
        ----------
        structure_file : str
            POSCAR / VASP structure file
        path : dict or OrderedDict
            {symbol: SiteGeometry(distances=array([...]))}
        """
        # ---- structure ----
        self.sites = read(structure_file)
        self.n_sites = len(self.sites)
        self.site_symbols = self.sites.get_chemical_symbols()

        # ---- site equivalence ----
        self.site_symbol_ranges = OrderedDict()
        self._build_site_ranges()

        self.site_symbol_of = np.empty(self.n_sites, dtype=object)
        for sym, (i0, i1) in self.site_symbol_ranges.items():
            self.site_symbol_of[i0:i1 + 1] = sym

        # ---- geometry / configuration ----
        self.path = path if path is not None else {}

        # ---- dynamic KMC data ----
        self.targets = None
        self.equal_events = None
        self.n_events = None

        # ---- cache ----
        self._jump_vectors = None

    # ======================================================================
    # internal helpers
    # ======================================================================

    def _build_site_ranges(self):
        """Build contiguous index ranges for each inequivalent site symbol."""
        idx = 0
        for sym, group in groupby(self.site_symbols):
            n = len(list(group))
            self.site_symbol_ranges[sym] = (idx, idx + n - 1)
            idx += n

    # ======================================================================
    # event construction
    # ======================================================================

    def build_events_from_site_geometry(self, tol=1e-3):
        """
        Build KMC events with flattened targets.

        Returns
        -------
        targets : list
            targets[i] = list of reachable site indices from site i
        n_events : list
            n_events[i] = total number of jumps from site i
        equal_events : list
            equal_events[i][s] = number of equivalent jumps
            for the s-th distance shell (path definition)
        """

        if not self.path:
            raise ValueError("path is empty; cannot build events")

        sites = self.sites
        n_sites = self.n_sites
        path_keys = list(self.path.keys())

        # ---------- allocate (flattened) ----------
        targets = [[] for _ in range(n_sites)]
        equal_events = [[] for _ in range(n_sites)]
        n_events = np.zeros(n_sites, dtype=int)

        # ---------- main loop ----------
        for i in range(n_sites):

            sym = self.site_symbol_of[i]
            if sym not in self.path:
                continue

            geom = self.path[sym]

            site_targets = []
            site_equal = []

            for d_ref in geom.distances:
                hit = []

                for k in range(n_sites):
                    if k == i:
                        continue

                    d = sites.get_distance(i, k, mic=True)
                    if abs(d - d_ref) < tol:
                        hit.append(k)
                        
                site_targets.extend(hit)
                site_equal.append(len(hit))

            targets[i] = site_targets
            equal_events[i] = site_equal
            n_events[i] = len(site_targets)

        self.targets = targets
        self.equal_events = equal_events
        self.n_events = n_events.tolist()

        return self.targets, self.n_events, self.equal_events

    # ======================================================================
    # jump vectors & displacement
    # ======================================================================

    def precompute_jump_vectors(self):
        """Precompute and cache jump vectors."""
        if self._jump_vectors is not None:
            return self._jump_vectors

        self._jump_vectors = np.zeros((self.n_sites, self.n_sites, 3))
        for i in range(self.n_sites):
            for j in range(self.n_sites):
                if i != j:
                    self._jump_vectors[i, j] = \
                        self.sites.get_distance(i, j, mic=True, vector=True)

        return self._jump_vectors

    def cal_displacement(self, events_count):
        """
        Calculate total displacement from events_count.
        """
        jump_vectors = self.precompute_jump_vectors()
        disp = np.zeros(3)

        rows, cols = np.nonzero(events_count)
        for i, j in zip(rows, cols):
            disp += jump_vectors[i, j] * events_count[i, j]

        return tuple(disp) 