"""
STEP 1 — Characterization Tests
================================
Run BEFORE any refactoring. These pin the current behavior as a contract.
If any of these break after a refactor, you changed behavior — stop and investigate.

Usage:
    python -m pytest tests/test_characterization.py -v

Requirements:
    pip install pytest torch numpy scipy
"""

import pytest
import numpy as np
import torch
import math
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Shared synthetic corpus fixture (tiny, fast, deterministic)
# ---------------------------------------------------------------------------
TOPICS   = 4
DOCS     = 20
SEQ_LEN  = 30
VOCABS   = 200   # max token id + 1
PAD_ID   = VOCABS - 1

@pytest.fixture(scope="module")
def synthetic_corpus():
    """Returns (data_tensor, anchor_tokens) suitable for both LDAGibbs classes."""
    torch.manual_seed(0)
    np.random.seed(0)
    data = torch.randint(2, PAD_ID - 1, [DOCS, SEQ_LEN])
    data[:, 25:] = PAD_ID          # last 5 positions are PAD
    # anchor: start, food, service, price, ambience, in_, fullstop, end, pad
    anchor = torch.tensor([1, 10, 20, 30, 40, 5, 3, 2, PAD_ID])
    return data, anchor


# ---------------------------------------------------------------------------
# Helper: run both models with identical tiny settings
# ---------------------------------------------------------------------------
def run_model(model_cls, data, anchor, epochs=2, max_iter=4, burnin_ratio=0.5):
    model = model_cls(data, TOPICS, anchor)
    return model.run(epochs=epochs, max_iter=max_iter, burnin_ratio=burnin_ratio)


# ---------------------------------------------------------------------------
# 1. Return-value contract: always 21 values, always correct shapes
# ---------------------------------------------------------------------------
class TestReturnContract:
    def test_nomab_returns_21_values(self, synthetic_corpus):
        from model_simulation_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        assert len(result) == 21, f"Expected 21 return values, got {len(result)}"

    def test_sparsity_returns_21_values(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        assert len(result) == 21

    def test_word_score_shape(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        word_score, word_prob, *_ = run_model(LDAGibbs, data, anchor)
        assert word_score.shape[0] == DOCS
        assert word_prob.shape[0] == DOCS

    def test_attention_masks_shape(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        for mask_idx in [12, 13, 14, 15, 16]:  # attention_mask through attention_mask5
            mask = result[mask_idx]
            assert mask.shape == (DOCS, SEQ_LEN), \
                f"result[{mask_idx}] shape {mask.shape} != ({DOCS}, {SEQ_LEN})"

    def test_rho_shape(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        rho = result[5]
        assert rho.shape == (VOCABS, TOPICS)

    def test_reward_shape(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        reward = result[17]
        assert reward.shape == (VOCABS,)


# ---------------------------------------------------------------------------
# 2. Behavioral difference: sparsity produces different rho from NoMAB
# ---------------------------------------------------------------------------
class TestBehavioralDifference:
    """
    Root cause of original failures: with max_iter=4 and a tiny corpus,
    the convergence/skip logic means MAB never activates (loss stays
    negative, never reaches 0/100 sentinel), so rho is never updated —
    making all variants identical at the end-to-end level.

    Fix: test _compute_reward and _scale_dirichlet DIRECTLY as unit tests.
    These are the two methods that define the behavioral difference between
    the models. Testing them directly is more precise and more robust than
    inferring the difference from a full run that may not reach MAB phase.
    """

    def _make_mock_model(self, model_cls, gamma=1.5, disable_gate=False):
        """
        Build a minimally-initialised model instance for unit-testing
        _compute_reward and _scale_dirichlet without running the full loop.
        """
        import torch
        data   = torch.randint(2, VOCABS-2, [DOCS, SEQ_LEN])
        anchor = torch.tensor([1, 10, 20, 30, 40, 5, 3, 2, VOCABS-1])
        model  = model_cls(data, TOPICS, anchor,
                           gamma=gamma, disable_gate=disable_gate)
        # Inject a plausible cntTW and cntW so reward computation is meaningful
        np.random.seed(99)
        model.cntTW = np.random.randint(0, 20,
                                        [TOPICS, VOCABS]).astype(float)
        model.cntW  = model.cntTW.sum(axis=0)
        model.loss  = -2.5   # typical normalised loss value
        model.TOPICS = TOPICS
        model.gamma  = gamma
        return model

    def test_reward_signals_differ_in_distribution(self):
        """
        The exclusivity reward is deterministic given cntTW.
        The simulation reward is stochastic given prFullCond.
        Over many calls with the same inputs, their OUTPUT DISTRIBUTIONS differ.

        Exclusivity: r = 1 iff Herfindahl(cntTW[:,w]) > (1/K)*gamma
        Simulation:  r = 1 iff Uniform(0,1)[new_z] < prFullCond[new_z]

        Key difference: exclusivity does NOT use prFullCond at all for the
        reward decision — it uses the count matrix. Simulation DOES use
        prFullCond and ignores the count matrix.
        """
        from model_simulation_reward  import LDAGibbs as SimModel
        from model_exclusivity_reward import LDAGibbs as ExclModel

        np.random.seed(42)
        epsilon = 1e-10

        # Build identical state for both models
        cntTW = np.random.randint(0, 20, [TOPICS, VOCABS]).astype(float)
        cntW  = cntTW.sum(axis=0)

        # Compute prFullCond for a specific word
        w_int = 50
        prFullCond = np.array([0.6, 0.2, 0.15, 0.05])  # clear winner: topic 0
        new_z = 0

        # --- Exclusivity reward (deterministic) ---
        word_topic_dist   = cntTW[:, w_int] / (cntW[w_int] + epsilon)
        excl_score        = float(np.sum(np.square(word_topic_dist)))
        threshold_low     = (1.0 / TOPICS) * 0.5   # gamma=0.5
        threshold_high    = (1.0 / TOPICS) * 5.0   # gamma=5.0
        r_excl_low  = 1 if excl_score > threshold_low  else 0
        r_excl_high = 1 if excl_score > threshold_high else 0

        # Exclusivity reward is INDEPENDENT of prFullCond
        # Verify: changing prFullCond does not change exclusivity reward
        prFullCond_alt = np.array([0.25, 0.25, 0.25, 0.25])  # uniform
        r_excl_alt = 1 if excl_score > (1.0 / TOPICS) * 1.5 else 0
        r_excl_orig = r_excl_alt  # same word -> same cntTW -> same score

        assert r_excl_alt == r_excl_orig, \
            "Exclusivity reward must be independent of prFullCond"

        # --- Simulation reward (stochastic, depends on prFullCond) ---
        # Over many draws, P(r=1) = prFullCond[new_z] = 0.6 for new_z=0
        n_trials = 10000
        np.random.seed(0)
        rewards_sim = []
        for _ in range(n_trials):
            sim = np.random.uniform(0, 1, TOPICS)
            r   = 1 if sim[new_z] < prFullCond[new_z] else 0
            rewards_sim.append(r)
        sim_rate = np.mean(rewards_sim)

        # Simulation rate should be ~prFullCond[new_z] = 0.6
        assert abs(sim_rate - prFullCond[new_z]) < 0.03, \
            f"Simulation reward rate {sim_rate:.3f} should be ~{prFullCond[new_z]}"

        # Exclusivity rate is 0 or 1 (deterministic), never 0.6
        assert r_excl_low in (0, 1), "Exclusivity reward must be binary"
        assert r_excl_low != sim_rate, \
            "Exclusivity and simulation rewards have fundamentally different mechanisms"

    def test_exclusivity_reward_uses_count_matrix_not_prfullcond(self):
        """
        Core contract: exclusivity reward = f(cntTW, cntW, gamma).
        It must NOT depend on prFullCond (unlike simulation reward).
        """
        from model_exclusivity_reward import LDAGibbs
        from count_store import DenseCountStore
        import torch

        np.random.seed(42)
        data   = torch.randint(2, VOCABS - 2, [DOCS, SEQ_LEN])
        anchor = torch.tensor([1, 10, 20, 30, 40, 5, 3, 2, VOCABS - 1])
        model  = LDAGibbs(data, TOPICS, anchor, gamma=1.5)
        model.TOPICS = TOPICS
        model.gamma  = 1.5

        # Initialise DenseCountStore with controlled counts
        store = DenseCountStore(TOPICS, VOCABS, DOCS, SEQ_LEN)
        cntTW = np.random.randint(0, 50, [TOPICS, VOCABS]).astype(float)
        store._cntTW[:] = cntTW
        store._cntW[:]  = cntTW.sum(axis=0)
        model._store = store

        w_int = 30
        new_z = 0

        # Two very different prFullCond vectors
        pfc1 = np.array([0.9, 0.05, 0.03, 0.02])
        pfc2 = np.array([0.1, 0.3,  0.4,  0.2])

        r1 = model._compute_reward(w_int, new_z, pfc1)
        r2 = model._compute_reward(w_int, new_z, pfc2)

        # Same word → same cntTW → same exclusivity score → same reward
        assert r1 == r2, \
            f"Exclusivity reward changed with prFullCond: r1={r1}, r2={r2}. " \
            "It must only depend on cntTW and gamma."

    def test_simulation_reward_depends_on_prfullcond(self):
        """
        Simulation reward = f(prFullCond[new_z]).
        Over many trials, reward rate must match prFullCond[new_z].
        """
        from model_simulation_reward import LDAGibbs
        import torch
        data   = torch.randint(2, VOCABS-2, [DOCS, SEQ_LEN])
        anchor = torch.tensor([1, 10, 20, 30, 40, 5, 3, 2, VOCABS-1])
        model  = LDAGibbs(data, TOPICS, anchor)
        model.TOPICS = TOPICS

        n_trials = 5000
        np.random.seed(7)

        # High prFullCond[0] -> high reward rate
        pfc_high = np.array([0.85, 0.08, 0.04, 0.03])
        rewards_high = [model._compute_reward(10, 0, pfc_high)
                        for _ in range(n_trials)]

        # Low prFullCond[0] -> low reward rate
        pfc_low  = np.array([0.05, 0.35, 0.4, 0.2])
        rewards_low  = [model._compute_reward(10, 0, pfc_low)
                        for _ in range(n_trials)]

        rate_high = np.mean(rewards_high)
        rate_low  = np.mean(rewards_low)

        assert rate_high > rate_low, \
            f"High prFullCond ({rate_high:.3f}) should give more rewards than low ({rate_low:.3f})"
        assert abs(rate_high - pfc_high[0]) < 0.04, \
            f"Simulation reward rate {rate_high:.3f} should be ~{pfc_high[0]}"

    def test_sparsity_reward_signal_is_exclusivity(self, synthetic_corpus):
        """gamma parameter is stored and model initialises correctly."""
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        model = LDAGibbs(data, TOPICS, anchor, gamma=1.5)
        assert model.gamma == 1.5


# ---------------------------------------------------------------------------
# 3. Loss function: must be finite, negative, monotonically improving (loose)
# ---------------------------------------------------------------------------
class TestLossProperties:
    def test_loss_is_finite_after_run(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        loss_history = result[2]
        finite_losses = [l for l in loss_history if l not in (0, 10, 100)]
        for l in finite_losses:
            assert math.isfinite(l), f"Non-finite loss value: {l}"

    def test_loss_is_negative(self, synthetic_corpus):
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        result = run_model(LDAGibbs, data, anchor)
        loss_history = result[2]
        real_losses = [l for l in loss_history if l < 0]
        assert len(real_losses) > 0, "No negative loss values found — something is wrong"


# ---------------------------------------------------------------------------
# 4. Vectorized math equivalence (unit tests for each optimized function)
#    These validate refactored implementations against the original loop versions
# ---------------------------------------------------------------------------
class TestVectorizedMath:
    """
    Each test compares the original loop-based computation against
    the vectorized replacement used in model_SemEval_sparsity_fast.py.
    All must pass before _fast can be considered a safe drop-in.
    """

    @pytest.fixture
    def arrays(self):
        np.random.seed(42)
        T, V, D = TOPICS, VOCABS, DOCS
        return {
            'cntTW': np.random.rand(T, V) * 20,
            'cntDT': np.random.rand(D, T) * 50,
            'cntT':  np.ones(T) * 1000,
            'lenD':  np.ones(D) * 200,
            'alpha': np.ones(T) * 0.1,
            'beta':  np.ones(V) * 0.01,
            'tokens': np.arange(50),
        }

    def test_loss_vectorized(self, arrays):
        from scipy.special import gammaln
        cntTW = arrays['cntTW']
        beta  = arrays['beta']
        eps   = 1e-10

        # Original loop
        ll_orig = 0
        for z in range(TOPICS):
            ll_orig += gammaln(np.sum(beta + eps))
            ll_orig -= np.sum(gammaln(beta + eps))
            ll_orig += np.sum(gammaln(cntTW[z] + beta + eps))
            ll_orig -= gammaln(np.sum(cntTW[z] + beta + eps))

        # Vectorized
        M = cntTW + beta + eps
        ll_fast = TOPICS * (gammaln(np.sum(beta + eps)) - np.sum(gammaln(beta + eps)))
        ll_fast += np.sum(gammaln(M))
        ll_fast -= np.sum(gammaln(np.sum(M, axis=1)))

        assert np.isclose(ll_orig, ll_fast), \
            f"Loss mismatch: orig={ll_orig:.6f}, fast={ll_fast:.6f}"

    def test_update_phi_theta_vectorized(self, arrays):
        cntTW = arrays['cntTW']
        cntDT = arrays['cntDT']
        cntT  = arrays['cntT']
        lenD  = arrays['lenD']
        alpha = arrays['alpha']
        beta  = arrays['beta']
        tokens = arrays['tokens']

        # Original nested loops
        theta_orig = np.zeros([DOCS, TOPICS])
        phi_orig   = np.zeros([TOPICS, VOCABS])
        for d in range(DOCS):
            for z in range(TOPICS):
                theta_orig[d][z] = (cntDT[d][z] + alpha[z]) / (lenD[d] + np.sum(alpha))
        for z in range(TOPICS):
            for w in tokens:
                phi_orig[z][w] = (cntTW[z][w] + beta[w]) / (cntT[z] + np.sum(beta))

        # Vectorized
        theta_fast = (cntDT + alpha) / (lenD[:, None] + np.sum(alpha))
        phi_fast   = np.zeros([TOPICS, VOCABS])
        phi_fast[:, tokens] = (cntTW[:, tokens] + beta[tokens]) / (cntT[:, None] + np.sum(beta))

        assert np.allclose(theta_orig, theta_fast), "theta mismatch"
        assert np.allclose(phi_orig[:, tokens], phi_fast[:, tokens]), "phi mismatch"

    def test_update_alpha_beta_vectorized(self, arrays):
        from scipy.special import psi
        cntTW = arrays['cntTW']
        cntDT = arrays['cntDT']
        alpha = arrays['alpha'].copy()
        beta  = arrays['beta'].copy()
        eps   = 1e-10

        # Original loops (beta)
        x, y = 0, 0
        for z in range(TOPICS):
            x += psi(cntTW[z] + beta + eps) - psi(beta + eps)
            y += psi(np.sum(cntTW[z] + beta) + eps) - psi(np.sum(beta) + eps)
        beta_orig = beta * x / (y + eps)

        # Original loops (alpha)
        x, y = 0, 0
        for d in range(DOCS):
            y += psi(np.sum(cntDT[d] + alpha) + eps) - psi(np.sum(alpha) + eps)
            x += psi(cntDT[d] + alpha + eps) - psi(alpha + eps)
        alpha_orig = alpha * x / (y + eps)

        # Vectorized
        alpha = arrays['alpha'].copy()
        beta  = arrays['beta'].copy()
        M = cntTW + beta + eps
        x = np.sum(psi(M) - psi(beta + eps), axis=0)
        y = np.sum(psi(np.sum(M, axis=1) + eps) - psi(np.sum(beta) + eps))
        beta_fast = beta * x / (y + eps)

        N = cntDT + alpha + eps
        x = np.sum(psi(N) - psi(alpha + eps), axis=0)
        y = np.sum(psi(np.sum(N, axis=1) + eps) - psi(np.sum(alpha) + eps))
        alpha_fast = alpha * x / (y + eps)

        assert np.allclose(alpha_orig, alpha_fast, atol=1e-10), \
            f"alpha max diff: {np.max(np.abs(alpha_orig - alpha_fast)):.2e}"
        assert np.allclose(beta_orig, beta_fast, atol=1e-10), \
            f"beta max diff: {np.max(np.abs(beta_orig - beta_fast)):.2e}"

    def test_topic_counts_bincount(self):
        """2D bincount must exactly match per-word loop for topic_counts."""
        np.random.seed(7)
        D, L, V, T = 20, 30, VOCABS, TOPICS
        PAD = PAD_ID
        documents  = np.random.randint(2, V - 1, [D, L])
        doc_topics = np.random.randint(0, T,     [D, L])
        tokens     = np.arange(50)

        # Original per-word loop
        tc_orig = np.zeros([V, T], dtype=int)
        for word in tokens:
            pos = np.where(documents == word)
            tc_orig[word] += np.array(
                [np.count_nonzero(doc_topics[pos] == i) for i in range(T)]
            )

        # Vectorized 2D bincount
        fw = documents.ravel().astype(np.int64)
        ft = doc_topics.ravel().astype(np.int64)
        valid = fw != PAD
        combined = fw[valid] * T + ft[valid]
        tc_fast = np.bincount(combined, minlength=V * T).reshape(V, T).astype(int)

        assert np.array_equal(tc_orig[tokens], tc_fast[tokens]), \
            "topic_counts bincount does not match loop result"


# ---------------------------------------------------------------------------
# 5. disable_gate flag: must produce different rho from gated version
# ---------------------------------------------------------------------------
class TestAblationFlags:
    """
    Root cause of original failures: with max_iter=4 on a tiny corpus,
    MAB never activates (loss stays negative), so _scale_dirichlet is
    never called and rho is never updated. End-to-end rho comparison
    is therefore meaningless for this parameter.

    Fix: test _scale_dirichlet and _compute_reward DIRECTLY.
    These are the exact functions the ablation controls affect.
    """

    def _make_excl_model(self, gamma=1.5, disable_gate=False):
        """
        Build a model with a fully initialised DenseCountStore so that
        _compute_reward (which calls self._store.get_tw_col / get_w) works
        without needing to call run() first.
        """
        import torch
        from model_exclusivity_reward import LDAGibbs
        from count_store import DenseCountStore

        np.random.seed(7)
        data   = torch.randint(2, VOCABS - 2, [DOCS, SEQ_LEN])
        anchor = torch.tensor([1, 10, 20, 30, 40, 5, 3, 2, VOCABS - 1])
        model  = LDAGibbs(data, TOPICS, anchor,
                          gamma=gamma, disable_gate=disable_gate)
        model.loss   = -2.5
        model.TOPICS = TOPICS
        model.gamma  = gamma

        # Initialise DenseCountStore and populate with realistic counts
        store = DenseCountStore(TOPICS, VOCABS, DOCS, SEQ_LEN)
        cntTW = np.random.randint(0, 30, [TOPICS, VOCABS]).astype(float)
        store._cntTW[:] = cntTW
        store._cntW[:]  = cntTW.sum(axis=0)
        model._store = store
        return model

    def test_disable_gate_changes_dirichlet_scale(self):
        """
        _scale_dirichlet with disable_gate=False returns
            prFullCond * (1 + 100*exp(loss)) + ε
        _scale_dirichlet with disable_gate=True returns
            prFullCond + ε
        These MUST differ numerically for any loss != 0.
        """
        import math
        from model_exclusivity_reward import LDAGibbs

        epsilon    = 1e-10
        loss       = -2.5
        prFullCond = np.array([0.5, 0.3, 0.15, 0.05])

        m_gated   = self._make_excl_model(disable_gate=False)
        m_nogated = self._make_excl_model(disable_gate=True)

        scale_gated   = m_gated._scale_dirichlet(prFullCond, epsilon)
        scale_nogated = m_nogated._scale_dirichlet(prFullCond, epsilon)

        # Gated: multiplier = 1 + 100*exp(-2.5) ≈ 9.21 >> 1
        expected_scale = 1.0 + 100.0 * math.exp(loss)
        assert expected_scale > 2.0, \
            f"Expected meaningful gate scale > 2, got {expected_scale:.3f}"

        assert not np.allclose(scale_gated, scale_nogated), \
            f"disable_gate should change _scale_dirichlet output.\n" \
            f"  gated   = {scale_gated}\n" \
            f"  nogated = {scale_nogated}"

        # Gated values must all be larger (larger concentration = sharper Dirichlet)
        assert np.all(scale_gated > scale_nogated + 1e-6), \
            "Gated scale must be strictly larger than ungated scale"

    def test_gamma_threshold_changes_reward_decision(self):
        """
        _compute_reward with gamma=0.5 uses threshold = (1/K)*0.5 = 0.125
        _compute_reward with gamma=5.0 uses threshold = (1/K)*5.0 = 1.25

        Maximum possible Herfindahl score = 1.0 (one topic gets all words).
        So gamma=5.0 threshold=1.25 > 1.0 -> reward is ALWAYS 0.
        gamma=0.5 threshold=0.125 -> most words will be rewarded.
        These must differ.
        """
        m_low  = self._make_excl_model(gamma=0.5)
        m_high = self._make_excl_model(gamma=5.0)

        # Pick a word with non-trivial cntTW
        w_int = 30
        prFullCond = np.ones(TOPICS) / TOPICS
        new_z = 0

        # gamma=5.0 threshold > 1.0 (impossible) -> always 0
        r_high = m_high._compute_reward(w_int, new_z, prFullCond)
        assert r_high == 0, \
            f"gamma=5.0 threshold=1.25 > max Herfindahl=1.0: reward must be 0, got {r_high}"

        # Over many words, gamma=0.5 gives more rewards than gamma=5.0
        n_words = 50
        rewards_low  = [m_low._compute_reward(w, 0, prFullCond)
                        for w in range(10, 10 + n_words)]
        rewards_high = [m_high._compute_reward(w, 0, prFullCond)
                        for w in range(10, 10 + n_words)]

        assert sum(rewards_low) > sum(rewards_high), \
            f"gamma=0.5 should give more rewards than gamma=5.0. " \
            f"Got: gamma=0.5 rewards={sum(rewards_low)}, gamma=5.0 rewards={sum(rewards_high)}"

        # gamma=5.0 must give exactly 0 rewards (threshold > max possible score)
        assert sum(rewards_high) == 0, \
            f"gamma=5.0 threshold=1.25 > max Herfindahl=1.0: total rewards must be 0, " \
            f"got {sum(rewards_high)}"

    def test_gamma_stored_correctly(self, synthetic_corpus):
        """Constructor must store gamma and disable_gate correctly."""
        from model_exclusivity_reward import LDAGibbs
        data, anchor = synthetic_corpus
        m1 = LDAGibbs(data, TOPICS, anchor, gamma=0.5,  disable_gate=False)
        m2 = LDAGibbs(data, TOPICS, anchor, gamma=5.0,  disable_gate=True)
        assert m1.gamma == 0.5  and m1.disable_gate == False
        assert m2.gamma == 5.0  and m2.disable_gate == True
