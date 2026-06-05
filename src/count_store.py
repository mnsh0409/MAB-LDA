"""
count_store.py
==============
Abstract storage backend for the three count matrices used in MAB-LDA Gibbs sampling:

    cntTW  [TOPICS, VOCABS]  — word-topic counts
    cntDT  [DOCS,   TOPICS]  — document-topic counts
    cntT   [TOPICS]          — total topic counts  (always dense, small)
    cntW   [VOCABS]          — total word counts   (always dense)

    topicAssignments [DOCS, max_doc_size]  — current topic per token position

API contract (all backends must honour)
----------------------------------------
    # scalar reads used in gibbs_update
    store.get_tw(z, w)          -> float    cntTW[z, w]
    store.get_dt(d, z)          -> float    cntDT[d, z]
    store.get_t(z)              -> float    cntT[z]
    store.get_w(w)              -> float    cntW[w]
    store.get_z(d, pos)         -> int      topicAssignments[d, pos]

    # vector reads used in prL / prR computation
    store.get_tw_col(w)         -> np.ndarray [K]   cntTW[:, w]
    store.get_dt_row(d)         -> np.ndarray [K]   cntDT[d, :]
    store.get_t_all()           -> np.ndarray [K]   cntT[:]

    # updates (called after every gibbs_update decision)
    store.dec(z, w, d, pos)     decrement cntTW[z,w], cntDT[d,z], cntT[z]
    store.inc(z, w, d, pos)     increment cntTW[z,w], cntDT[d,z], cntT[z]
    store.inc_w(w)              increment cntW[w]
    store.set_z(d, pos, z)      topicAssignments[d, pos] = z

    # bulk operations used in run() bookkeeping
    store.snapshot()            -> opaque object    deep copy of all state
    store.restore(snap)         restore from snapshot
    store.to_dense()            -> (cntTW, cntDT, cntT, cntW, topicAssignments)
                                   all as numpy arrays (for Loss, update_phi_theta, etc.)

    # initialisation
    store.init_from_corpus(documents, initial_topics, pad_int, START_id)
"""

import abc
import numpy as np


# ===========================================================================
# Abstract base
# ===========================================================================
class CountStore(abc.ABC):
    """Abstract storage backend. See module docstring for the full API."""

    def __init__(self, TOPICS: int, VOCABS: int, DOCS: int, max_doc_size: int):
        self.TOPICS       = TOPICS
        self.VOCABS       = VOCABS
        self.DOCS         = DOCS
        self.max_doc_size = max_doc_size

    # ------------------------------------------------------------------
    # Scalar reads
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_tw(self, z: int, w: int) -> float: ...

    @abc.abstractmethod
    def get_dt(self, d: int, z: int) -> float: ...

    @abc.abstractmethod
    def get_t(self, z: int) -> float: ...

    @abc.abstractmethod
    def get_w(self, w: int) -> float: ...

    @abc.abstractmethod
    def get_z(self, d: int, pos: int) -> int: ...

    # ------------------------------------------------------------------
    # Vector reads
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_tw_col(self, w: int) -> np.ndarray: ...

    @abc.abstractmethod
    def get_dt_row(self, d: int) -> np.ndarray: ...

    @abc.abstractmethod
    def get_t_all(self) -> np.ndarray: ...

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def dec(self, z: int, w: int, d: int, pos: int) -> None: ...

    @abc.abstractmethod
    def inc(self, z: int, w: int, d: int, pos: int) -> None: ...

    @abc.abstractmethod
    def inc_w(self, w: int) -> None: ...

    @abc.abstractmethod
    def set_z(self, d: int, pos: int, z: int) -> None: ...

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def snapshot(self): ...

    @abc.abstractmethod
    def restore(self, snap) -> None: ...

    @abc.abstractmethod
    def to_dense(self) -> tuple: ...

    @abc.abstractmethod
    def init_from_corpus(self, documents, initial_topics: np.ndarray,
                         pad_int: int, start_id: int) -> np.ndarray:
        """
        Populate all count matrices from the corpus and initial random topics.
        Returns lenD: np.ndarray [DOCS] — non-PAD, non-START token count per doc.
        """
        ...


# ===========================================================================
# ICDM backend — dense numpy arrays (current behaviour, zero behaviour change)
# ===========================================================================
class DenseCountStore(CountStore):
    """
    Dense numpy array backend.
    Identical in behaviour to the original direct-array accesses in model_base.py.
    Memory: O(K*V + D*K) floats.
    Used for: all ICDM experiments (K≤20, V≤50K, D≤5K).
    """

    def __init__(self, TOPICS: int, VOCABS: int, DOCS: int, max_doc_size: int):
        super().__init__(TOPICS, VOCABS, DOCS, max_doc_size)
        # Primary count matrices
        self._cntTW = np.zeros((TOPICS, VOCABS),   dtype=np.float64)
        self._cntDT = np.zeros((DOCS,   TOPICS),   dtype=np.float64)
        self._cntT  = np.zeros(TOPICS,             dtype=np.float64)
        self._cntW  = np.zeros(VOCABS,             dtype=np.float64)
        self._z     = np.zeros((DOCS, max_doc_size), dtype=np.int32)

    # --- scalar reads ---
    def get_tw(self, z, w):       return self._cntTW[z, w]
    def get_dt(self, d, z):       return self._cntDT[d, z]
    def get_t(self, z):           return self._cntT[z]
    def get_w(self, w):           return self._cntW[w]
    def get_z(self, d, pos):      return int(self._z[d, pos])

    # --- vector reads ---
    def get_tw_col(self, w):      return self._cntTW[:, w]      # [K] view
    def get_dt_row(self, d):      return self._cntDT[d, :]      # [K] view
    def get_t_all(self):          return self._cntT             # [K] view

    # --- updates ---
    def dec(self, z, w, d, pos):
        self._cntTW[z, w] -= 1
        self._cntDT[d, z] -= 1
        self._cntT[z]     -= 1

    def inc(self, z, w, d, pos):
        self._cntTW[z, w] += 1
        self._cntDT[d, z] += 1
        self._cntT[z]     += 1

    def inc_w(self, w):           self._cntW[w] += 1
    def set_z(self, d, pos, z):   self._z[d, pos] = z

    # --- bulk ---
    def snapshot(self):
        return (self._cntTW.copy(), self._cntDT.copy(),
                self._cntT.copy(),  self._cntW.copy(),
                self._z.copy())

    def restore(self, snap):
        self._cntTW[:] = snap[0]
        self._cntDT[:] = snap[1]
        self._cntT[:]  = snap[2]
        self._cntW[:]  = snap[3]
        self._z[:]     = snap[4]

    def to_dense(self):
        return (self._cntTW, self._cntDT, self._cntT,
                self._cntW,  self._z)

    def init_from_corpus(self, documents, initial_topics, pad_int, start_id):
        lenD = np.zeros(self.DOCS, dtype=np.float64)
        for doc_num in range(self.DOCS):
            np.random.seed(doc_num)
            tmp = initial_topics[doc_num]
            self._z[doc_num] = tmp
            for i, word in enumerate(documents[doc_num]):
                w_int = int(word)
                if w_int == pad_int:
                    break
                z = int(tmp[i])
                self._cntTW[z, w_int] += 1
                self._cntDT[doc_num, z] += 1
                self._cntT[z]           += 1
                if w_int != start_id:
                    lenD[doc_num] += 1
                self._cntW[w_int] += 1
        return lenD

    # ------------------------------------------------------------------
    # Dense-only helpers used by Loss(), update_alpha_beta(), bincount
    # (not in the abstract interface — only DenseCountStore exposes these)
    # ------------------------------------------------------------------
    @property
    def cntTW(self) -> np.ndarray:
        """Direct access to the [K, V] matrix for vectorized ops in run()."""
        return self._cntTW

    @property
    def cntDT(self) -> np.ndarray:
        """Direct access to the [D, K] matrix for vectorized ops in run()."""
        return self._cntDT

    @property
    def cntT(self) -> np.ndarray:
        return self._cntT

    @property
    def cntW(self) -> np.ndarray:
        return self._cntW

    @property
    def z(self) -> np.ndarray:
        """Direct access to topicAssignments for bincount and attention masks."""
        return self._z


# ===========================================================================
# Correctness-bridge backend — scipy CSR sparse matrices
# ===========================================================================
class SparseCountStore(CountStore):
    """
    Sparse CSR backend using scipy.
    Memory: O(nnz) where nnz << K*V for large K.
    Used for:
      - Correctness validation of CUDACountStore before GPU development
      - Large-K CPU experiments (K=100+) where dense [K,V] exceeds RAM
      - NOT used for ICDM submission experiments (DenseCountStore is faster
        for the small K,V sizes in ICDM datasets)

    Correctness guarantee:
        For any corpus, DenseCountStore and SparseCountStore must produce
        identical numerical results. Run test_count_store_equivalence() to verify.
    """

    def __init__(self, TOPICS: int, VOCABS: int, DOCS: int, max_doc_size: int):
        try:
            import scipy.sparse as sp
            self._sp = sp
        except ImportError:
            raise ImportError("scipy is required for SparseCountStore: pip install scipy")
        super().__init__(TOPICS, VOCABS, DOCS, max_doc_size)

        # lil_matrix allows efficient incremental updates; convert to CSR for reads
        self._cntTW_lil = self._sp.lil_matrix((TOPICS, VOCABS), dtype=np.float64)
        self._cntDT_lil = self._sp.lil_matrix((DOCS,   TOPICS), dtype=np.float64)
        self._cntT      = np.zeros(TOPICS, dtype=np.float64)
        self._cntW      = np.zeros(VOCABS, dtype=np.float64)
        self._z         = np.zeros((DOCS, max_doc_size), dtype=np.int32)
        self._dirty     = True    # True = lil is current; False = csr is current
        self._csr_tw    = None
        self._csr_dt    = None
        # Let's assume it stores data in self._cntTW (a dict of dicts)
        self._cntTW = {} 
        self._cntDT = {}
        self.K = K
        self.V = len(tokens_tensor)

    # --- The API Bridge ---
    @property
    def cntTW(self):
        """
        Temporarily reconstructs the dense matrix specifically for the 
        model_base.py loss calculation, then destroys it to save RAM.
        """
        dense_matrix = np.zeros((self.K, self.V), dtype=np.int32)
        for k, vocab_dict in self._cntTW.items():
            for v, count in vocab_dict.items():
                dense_matrix[k, v] = count
        return dense_matrix

    @property
    def cntDT(self):
        # Similarly reconstruct the D x K matrix if needed
        pass
        
    def _ensure_csr(self):
        if self._dirty:
            self._csr_tw = self._cntTW_lil.tocsr()
            self._csr_dt = self._cntDT_lil.tocsr()
            self._dirty  = False

    # --- scalar reads ---
    def get_tw(self, z, w):
        return float(self._cntTW_lil[z, w])

    def get_dt(self, d, z):
        return float(self._cntDT_lil[d, z])

    def get_t(self, z):    return self._cntT[z]
    def get_w(self, w):    return self._cntW[w]
    def get_z(self, d, pos): return int(self._z[d, pos])

    # --- vector reads (convert slice to dense array) ---
    def get_tw_col(self, w):
        return np.asarray(self._cntTW_lil.getcol(w).todense()).ravel()

    def get_dt_row(self, d):
        return np.asarray(self._cntDT_lil.getrow(d).todense()).ravel()

    def get_t_all(self): return self._cntT

    # --- updates ---
    def dec(self, z, w, d, pos):
        self._cntTW_lil[z, w] -= 1
        self._cntDT_lil[d, z] -= 1
        self._cntT[z]         -= 1
        self._dirty = True

    def inc(self, z, w, d, pos):
        self._cntTW_lil[z, w] += 1
        self._cntDT_lil[d, z] += 1
        self._cntT[z]         += 1
        self._dirty = True

    def inc_w(self, w):         self._cntW[w] += 1
    def set_z(self, d, pos, z): self._z[d, pos] = z

    # --- bulk ---
    def snapshot(self):
        return (self._cntTW_lil.copy(), self._cntDT_lil.copy(),
                self._cntT.copy(),      self._cntW.copy(),
                self._z.copy())

    def restore(self, snap):
        self._cntTW_lil = snap[0].copy()
        self._cntDT_lil = snap[1].copy()
        self._cntT[:]   = snap[2]
        self._cntW[:]   = snap[3]
        self._z[:]      = snap[4]
        self._dirty     = True

    def to_dense(self):
        return (self._cntTW_lil.toarray(),
                self._cntDT_lil.toarray(),
                self._cntT,
                self._cntW,
                self._z)

    def init_from_corpus(self, documents, initial_topics, pad_int, start_id):
        lenD = np.zeros(self.DOCS, dtype=np.float64)
        for doc_num in range(self.DOCS):
            np.random.seed(doc_num)
            tmp = initial_topics[doc_num]
            self._z[doc_num] = tmp
            for i, word in enumerate(documents[doc_num]):
                w_int = int(word)
                if w_int == pad_int:
                    break
                z = int(tmp[i])
                self._cntTW_lil[z, w_int] += 1
                self._cntDT_lil[doc_num, z] += 1
                self._cntT[z]               += 1
                if w_int != start_id:
                    lenD[doc_num] += 1
                self._cntW[w_int] += 1
        self._dirty = True
        return lenD


# ===========================================================================
# Equivalence test (run before any ICDE CUDACountStore development)
# ===========================================================================
def test_count_store_equivalence(store_a: CountStore, store_b: CountStore,
                                 documents, initial_topics,
                                 pad_int: int, start_id: int,
                                 n_update_rounds: int = 100,
                                 tol: float = 1e-10) -> bool:
    """
    Verify that two CountStore backends produce identical results for the
    same sequence of init + update operations.

    Usage (run before ICDE CUDACountStore replaces DenseCountStore):
        dense  = DenseCountStore(K, V, D, L)
        sparse = SparseCountStore(K, V, D, L)
        assert test_count_store_equivalence(dense, sparse, documents, ...)
    """
    import numpy as np

    # Initialise both from same corpus
    lenD_a = store_a.init_from_corpus(documents, initial_topics, pad_int, start_id)
    lenD_b = store_b.init_from_corpus(documents, initial_topics, pad_int, start_id)
    assert np.allclose(lenD_a, lenD_b, atol=tol), "lenD mismatch after init"

    rng = np.random.default_rng(0)
    K, V, D, L = store_a.TOPICS, store_a.VOCABS, store_a.DOCS, store_a.max_doc_size

    for _ in range(n_update_rounds):
        d   = int(rng.integers(0, D))
        pos = int(rng.integers(0, L))
        w   = int(rng.integers(0, V))
        z   = store_a.get_z(d, pos)

        # Apply same dec / inc / set_z to both
        store_a.dec(z, w, d, pos)
        store_b.dec(z, w, d, pos)
        new_z = int(rng.integers(0, K))
        store_a.inc(new_z, w, d, pos)
        store_b.inc(new_z, w, d, pos)
        store_a.set_z(d, pos, new_z)
        store_b.set_z(d, pos, new_z)

    # Compare to_dense() outputs
    da = store_a.to_dense()
    db = store_b.to_dense()
    for i, name in enumerate(['cntTW', 'cntDT', 'cntT', 'cntW', 'z']):
        if not np.allclose(np.array(da[i], dtype=float),
                           np.array(db[i], dtype=float), atol=tol):
            print(f"MISMATCH in {name}: max diff = "
                  f"{np.max(np.abs(np.array(da[i]) - np.array(db[i]))):.2e}")
            return False

    print(f"CountStore equivalence verified over {n_update_rounds} random updates.")
    return True


# ===========================================================================
# Placeholder stub for the ICDE CUDA backend
# ===========================================================================
class CUDACountStore(CountStore):
    """
    GPU-resident sparse count matrix backend.
    ICDE paper contribution — implemented in a separate C++/CUDA repository.

    This stub exists to:
      1. Document the expected interface for the CUDA implementation
      2. Raise a clear error if accidentally instantiated before ICDE work begins
      3. Serve as the import target in model_base.py so no import changes
         are needed when the real implementation is dropped in

    Implementation notes for ICDE paper:
      cntTW  → cuSPARSE CSR matrix on GPU global memory
      cntDT  → cuSPARSE CSR matrix on GPU global memory
      cntT   → dense float32 array in GPU global memory (K elements, small)
      z      → flat (wordId, docId, topicId) triplet list for warp-parallel access

      Core kernel: warp_gibbs_kernel<<<grid, block>>>
        - Each warp owns one document slice of the token list
        - Atomic increment/decrement on sparse CSR entries
        - Thompson sampling via cuRAND standard_gamma approximation
        - Exclusivity reward computed from GPU-resident word-topic distribution

      Memory layout:
        [wordId | docId | topicId] packed as int32 triplets, sorted by docId
        Enables coalesced memory access when iterating by document
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CUDACountStore (GPU backend) is not included in this release. "
            "Please use DenseCountStore (default) and SparseCountStore for CPU execution  "
        )

    # Stub implementations to satisfy the abstract interface checker
    def get_tw(self, z, w): raise NotImplementedError
    def get_dt(self, d, z): raise NotImplementedError
    def get_t(self, z):     raise NotImplementedError
    def get_w(self, w):     raise NotImplementedError
    def get_z(self, d, pos): raise NotImplementedError
    def get_tw_col(self, w): raise NotImplementedError
    def get_dt_row(self, d): raise NotImplementedError
    def get_t_all(self):    raise NotImplementedError
    def dec(self, z, w, d, pos): raise NotImplementedError
    def inc(self, z, w, d, pos): raise NotImplementedError
    def inc_w(self, w):     raise NotImplementedError
    def set_z(self, d, pos, z): raise NotImplementedError
    def snapshot(self):     raise NotImplementedError
    def restore(self, snap): raise NotImplementedError
    def to_dense(self):     raise NotImplementedError
    def init_from_corpus(self, *a, **kw): raise NotImplementedError
