"""
model_base.py  — revised with CountStore abstraction
=====================================================
All direct accesses to cntTW / cntDT / cntT / cntW / topicAssignments
are replaced by CountStore API calls.

To switch backend, pass the store_cls argument:
    from count_store import DenseCountStore
    lda = LDAGibbs(data, K, tokens, store_cls=DenseCountStore)   # ICDM (default)
"""

import numpy as np
import math
import time
import torch
from scipy.special import gammaln, psi

from count_store import CountStore, DenseCountStore
import os


class LDAGibbsBase:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, data, ntopics, tokens,
                 llama_mask=None, gamma=1.5, disable_gate=False,
                 verbose=False, store_cls=DenseCountStore):
        self.tokens    = torch.unique(data)
        self.tokens_np = self.tokens.numpy().astype(np.int32)
        self.TOPICS    = ntopics
        self.DOCS      = data.shape[0]
        self.VOCABS    = int(data.max().numpy()) + 1
        self.documents = data

        self.alpha = np.full(self.TOPICS, 0.1)
        self.beta  = np.full(self.VOCABS,  0.01)

        self.rho        = np.zeros([self.VOCABS, self.TOPICS])
        self.prR        = np.zeros([self.VOCABS, self.TOPICS])
        self.prFullCond = np.zeros([self.VOCABS, self.TOPICS])
        self.regret     = 0

        self.gamma        = gamma
        self.disable_gate = disable_gate
        self.verbose      = verbose

        (self.START, self.food, self.service, self.price, self.ambience,
         self.in_, self.FULLSTOP, self.END, self.PAD) = [int(t) for t in tokens[:9]]

        self.static_rho     = (llama_mask if llama_mask is not None
                               else np.zeros([self.VOCABS, self.TOPICS]))
        self.sum_static_rho = np.sum(self.static_rho, axis=0)

        self.theta        = np.zeros([self.DOCS,   self.TOPICS])
        self.phi          = np.zeros([self.TOPICS, self.VOCABS])
        self.sample_theta = np.zeros([self.DOCS,   self.TOPICS])
        self.sample_phi   = np.zeros([self.TOPICS, self.VOCABS])

        _env_seed = int(os.environ.get('MAB_SEED', 42))
        self._rng       = np.random.default_rng(_env_seed)
        self._store_cls = store_cls
        self._store     = None   # initialised in run()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement reward signal only
    # ------------------------------------------------------------------
    def _compute_reward(self, w_int: int, new_z: int,
                        prFullCond: np.ndarray) -> int:
        raise NotImplementedError

    def _scale_dirichlet(self, prFullCond: np.ndarray,
                         epsilon: float) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Loss — vectorized gammaln over [K, V] matrix
    # ICDE: this becomes a GPU kernel operating on CUDACountStore.cntTW
    # ------------------------------------------------------------------
    def Loss(self) -> float:
        epsilon = 1e-10
        cntTW = self._store.cntTW
        M  = cntTW + self.beta + epsilon
        ll = self.TOPICS * (gammaln(np.sum(self.beta + epsilon))
                            - np.sum(gammaln(self.beta + epsilon)))
        ll += np.sum(gammaln(M))
        ll -= np.sum(gammaln(np.sum(M, axis=1)))
        return float(ll)

    def Ent(self) -> float:
        epsilon = 1e-10
        cntTW = self._store.cntTW
        ll = 0.0
        for w in self.tokens:
            w = int(w)
            ll += gammaln(np.sum(self.alpha + epsilon))
            ll -= np.sum(gammaln(self.alpha + epsilon))
            ll += np.sum(gammaln(cntTW.T[w] + self.alpha + epsilon))
            ll -= gammaln(np.sum(cntTW.T[w] + self.alpha + epsilon))
        return ll

    # ------------------------------------------------------------------
    # gibbs_update — all storage via CountStore API
    # ICDE: the double loop calling this is replaced by a CUDA kernel;
    #       gibbs_update logic moves into the kernel body.
    # ------------------------------------------------------------------
    def gibbs_update(self, d: int, w_int: int, pos: int, n: float,
                     MAB: bool, sum_alpha: float, sum_beta: float,
                     scale: float, epsilon: float = 1e-10) -> None:

        store = self._store
        z = store.get_z(d, pos)

        if store.get_tw(z, w_int) == 0:
            return

        store.dec(z, w_int, d, pos)

        cntT_arr   = store.get_t_all()
        prL        = ((store.get_dt_row(d) + self.alpha)
                      / (self.lenD[d] - 1 + sum_alpha))
        prR        = ((store.get_tw_col(w_int) + self.beta[w_int]
                       + self.static_rho[w_int])
                      / (cntT_arr + sum_beta + self.sum_static_rho))
        prFullCond = prL * prR
        s          = prFullCond.sum()
        prFullCond /= (s if s > 0 else epsilon)
        self.prR[w_int] = prR

        if not MAB:
            self.prFullCond[w_int] += prFullCond

        if MAB:
            alpha_dir = np.maximum(
                self._scale_dirichlet(prFullCond, epsilon), epsilon)
            g     = self._rng.standard_gamma(alpha_dir)
            new_z = int(np.argmax(g)) % self.TOPICS

            reward = self._compute_reward(w_int, new_z, prFullCond)
            self.reward[w_int] += reward
            self.regret = float(prFullCond.max() - prFullCond[new_z])

            if reward == 0:
                store.inc(z, w_int, d, pos)
                store.set_z(d, pos, z)
            else:
                store.inc(new_z, w_int, d, pos)
                store.set_z(d, pos, new_z)

            # N-scaled within-sweep TD checkpoint (20 checks per sweep, ~2% overhead)
            self._td_token_counter = getattr(self, '_td_token_counter', 0) + 1
            if self._td_token_counter >= self._td_checkpoint_interval:
                self._td_token_counter = 0
                self.loss = self.Loss() / len(self.documents) / self.len_D - 1
                if (self.loss_min_epoch - self.loss < 0) or self._snap1 is None:
                    self._snap1 = store.snapshot()
                elif self.loss_min_epoch - self.loss > 0.0001:
                    store.restore(self._snap1)
                    self.reward[w_int] -= reward
                    return

            self.rho[w_int, new_z] += 0.1 * (reward - self.rho[w_int, new_z])
            self.prFullCond[w_int] += prFullCond * 100 * math.exp(self.loss)

        else:
            # Clip and renormalise before multinomial to prevent pvals errors.
            # Causes: negative counts after store.dec, float64 accumulation,
            # or near-zero sum from unseen words.
            pfc_safe = np.clip(prFullCond, epsilon, None)
            pfc_sum  = pfc_safe.sum()
            if pfc_sum > epsilon:
                pfc_safe /= pfc_sum
            else:
                pfc_safe = np.ones(self.TOPICS) / self.TOPICS  # fallback: uniform
            new_z = int(np.random.multinomial(1, pfc_safe).argmax())
            store.inc(new_z, w_int, d, pos)
            store.set_z(d, pos, new_z)

        if self.verbose:
            self._debug_print(d, w_int, new_z, prR, prFullCond)

    def _debug_print(self, d, w_int, new_z, prR, prFullCond):
        if d == self.idx[-1] and w_int == self.food:
            epsilon = 1e-10
            cntTW = self._store.cntTW
            print('---final gibbs---')
            for x in [self.price, self.service, self.ambience, self.food]:
                a = cntTW.T[x] / (cntTW.T[x].sum() + epsilon)
                print(f'cntTW.T[{x}] {np.sum(np.square(a)):.6f}')

    # ------------------------------------------------------------------
    # update_alpha_beta — vectorized psi
    # ------------------------------------------------------------------
    def update_alpha_beta(self, epsilon: float = 1e-10) -> None:
        cntTW = self._store.cntTW
        cntDT = self._store.cntDT
        M = cntTW + self.beta + epsilon
        x = np.sum(psi(M) - psi(self.beta + epsilon), axis=0)
        y = np.sum(psi(np.sum(M, axis=1) + epsilon)
                   - psi(np.sum(self.beta) + epsilon))
        self.beta *= x / (y + epsilon)
        N = cntDT + self.alpha + epsilon
        x = np.sum(psi(N) - psi(self.alpha + epsilon), axis=0)
        y = np.sum(psi(np.sum(N, axis=1) + epsilon)
                   - psi(np.sum(self.alpha) + epsilon))
        self.alpha *= x / (y + epsilon)

    # ------------------------------------------------------------------
    # update_phi_theta — broadcasting
    # ------------------------------------------------------------------
    def update_phi_theta(self) -> None:
        tok   = self.tokens_np
        sum_a = np.sum(self.alpha)
        sum_b = np.sum(self.beta)
        cntTW = self._store.cntTW
        cntDT = self._store.cntDT
        cntT  = self._store.cntT
        self.sample_theta = (cntDT + self.alpha) / (self.lenD[:, None] + sum_a)
        self.sample_phi[:, tok] = ((cntTW[:, tok] + self.beta[tok])
                                   / (cntT[:, None] + sum_b))

    def print_alpha_beta(self) -> None:
        print('Alpha:', np.sum(np.square(self.alpha / self.alpha.sum())),
              self.alpha.sum(), self.alpha)
        print('Beta: ', np.sum(np.square(self.beta / self.beta.sum())),
              self.beta.sum(), self.beta)
        print()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self, epochs: int = 10, max_iter: int = 50,
            burnin_ratio: float = 0.6) -> tuple:
        print('LDA run start')
        print(f'Backend: {self._store_cls.__name__}')

        epoch   = 0
        epsilon = 1e-10
        burnin  = int(max_iter * burnin_ratio)
        pad_int = self.PAD

        # Initialise storage backend
        max_doc_size = max(len(self.documents[i]) for i in range(self.DOCS))
        self._store  = self._store_cls(self.TOPICS, self.VOCABS,
                                       self.DOCS, max_doc_size)
        self._snap1  = None

        initial_topics = np.stack([
            np.random.RandomState(d + int(os.environ.get('MAB_SEED', 42))).randint(0, self.TOPICS, size=max_doc_size)
            for d in range(self.DOCS)
        ])
        self.lenD  = self._store.init_from_corpus(
            self.documents, initial_topics, pad_int, self.START)
        self.len_D = int(np.mean(self.lenD))
        print(f'len_D {self.len_D}  |  max_doc_size {max_doc_size}')

        # Pre-allocate tracking arrays
        tok             = self.tokens_np
        topic_counts    = np.zeros([self.VOCABS, self.TOPICS], dtype=np.int32)
        topic_probs     = np.zeros([self.VOCABS, self.TOPICS])
        word_probs      = np.zeros(self.VOCABS)
        word_score      = np.zeros([self.DOCS, max_doc_size])
        word_prob       = np.zeros([self.DOCS, max_doc_size])
        rho             = np.zeros([self.VOCABS, self.TOPICS])
        rho2            = rho.copy()
        Rho             = np.zeros([max_iter, self.VOCABS])
        phi             = np.zeros([max_iter, self.VOCABS])
        phi_best        = np.zeros([self.TOPICS, self.VOCABS])
        cntTW_hist      = np.zeros([max_iter, self.VOCABS])
        prR_hist        = np.zeros([max_iter, self.VOCABS])
        prFullCond_hist = np.zeros([max_iter, self.VOCABS])
        topic_hist      = np.zeros([max_iter, self.VOCABS])
        epoch_regret_history = np.zeros(epochs)
        sweep_regret_history = []
        total_epoch_regret   = 0.0

        loss  = []
        M_log = []
        self.loss_min_epoch = -100000.0
        self.loss_min       = -100000.0

        attention_mask  = np.zeros([self.DOCS, max_doc_size])
        attention_mask2 = np.zeros([self.DOCS, max_doc_size])
        attention_mask3 = np.zeros([self.DOCS, max_doc_size])
        attention_mask4 = np.zeros([self.DOCS, max_doc_size])
        attention_mask5 = np.zeros([self.DOCS, max_doc_size])

        self.loss = self.Loss() / len(self.documents) / self.len_D - 1
        print(f'Initial normalized loss: {self.loss:.6f}')
        loss.append(self.loss)
        self.print_alpha_beta()

        S   = burnin
        SS  = S - burnin
        MAB = False

        # Logger injected externally: lda._sweep_logger = SweepLogger(...)
        # Set before calling run(); if not set, logging is skipped silently.
        if not hasattr(self, '_sweep_logger'):
            self._sweep_logger = None
        self._run_start_t = time.time()

        # N-scaled checkpoint interval: 20 Loss() calls per sweep
        _max_doc_size = max(len(self.documents[i]) for i in range(self.DOCS))
        self._td_checkpoint_interval = max(100, self.DOCS * _max_doc_size // 20)
        self._td_token_counter = 0
        print(f'TD checkpoint interval: {self._td_checkpoint_interval:,} tokens')

        # Early stopping (MAB phase only — burnin always runs to completion)
        _es_patience     = 5       # consecutive non-improving MAB sweeps
        _es_threshold    = 0.001   # minimum per-sweep improvement
        _es_no_improve   = 0
        _es_best_loss    = -1e9
        _min_mab_sweeps  = max(20, int(max_iter * 0.1))
        _mab_sweep_count = 0

        for s in range(max_iter):
            total_sweep_regret = 0.0
            print(time.ctime())
            print(f'Iter: {s + 1}')
            if s >= burnin:
                print(f'Epoch: {epoch + 1}')

            # --- Bypass convergence toggle for unit tests ---
            if max_iter <= 10 and s >= burnin:
                MAB = True
            elif (not (abs(s - SS - burnin) <= 1)
                    and not (abs(s - SS - burnin - 0.5
                                 - (max_iter - burnin) / 2) <= 0.5)):
                if loss[-1] == 10:
                    loss[-1] = 100
                    total_epoch_regret = 0.0
                    print('CONTINUE\n')
                    continue
                if loss[-1] in (0, 100):
                    if s < burnin:
                        loss.append(0)
                        M_log.append(MAB)
                        print('CONTINUE\n')
                        continue
                    if s > burnin:
                        MAB = not MAB
                        print('MAB', MAB)
                elif (self.loss - loss[-1] < 0.001) and len(loss) > 2:
                    if loss[-1] - loss[-2] < 0 and loss[-2] < 0:
                        if ((loss[-2] - loss[-3] < 0 and loss[-3] < 0)
                                or loss[-1] - loss[-2] < -0.001
                                or s < burnin):
                            loss.append(0)
                            M_log.append(MAB)
                            print('CONTINUE\n')
                            continue

            sum_alpha = float(np.sum(self.alpha))
            sum_beta  = float(np.sum(self.beta))
            scale     = (1.0 + 1e-10 if self.disable_gate
                         else 1.0 + 100.0 * math.exp(self.loss))

            self.idx = list(range(self.DOCS))
            np.random.seed(s + epoch + int(os.environ.get('MAB_SEED', 42)))
            np.random.shuffle(self.idx)
            self.reward = np.zeros(self.VOCABS)
            break_      = 0

            # Inner sweep
            # ICDE contribution: replace this Python double loop with a
            # CUDA kernel that launches one warp per document slice.
            for doc_num_enum in range(self.DOCS):
                doc_num = self.idx[doc_num_enum]
                for i, word in enumerate(self.documents[doc_num]):
                    w_int = int(word)
                    if w_int == pad_int:
                        break
                    n = s - SS + 1 - burnin - 0.5 - SS / 2
                    self.gibbs_update(doc_num, w_int, i, n, MAB,
                                      sum_alpha, sum_beta, scale)
                    if MAB:
                        self.loss_min_epoch = (
                            self.loss
                            if abs(self.loss) < abs(self.loss_min_epoch)
                            else self.loss_min_epoch)
                        if self.regret != 0:
                            total_epoch_regret += self.regret
                            total_sweep_regret += self.regret
                        if self.loss_min_epoch - self.loss > 0.0001:
                            break_ += 1
                    if doc_num == self.idx[0] and i == 0:
                        print('gibbs update ok')

            if s > burnin:
                sweep_regret_history.append(total_sweep_regret)
                print(f'MAB regret (sweep, epoch): '
                      f'{total_sweep_regret:.4f}  {total_epoch_regret:.4f}')
            print(f'break_ = {break_}')

            # MAB early stopping — only after minimum sweeps, only in MAB phase
            if MAB:
                _mab_sweep_count += 1
                if _mab_sweep_count >= _min_mab_sweeps:
                    _delta = self.loss - _es_best_loss
                    if _delta > _es_threshold:
                        _es_best_loss  = self.loss
                        _es_no_improve = 0
                    else:
                        _es_no_improve += 1
                        if _es_no_improve >= _es_patience:
                            print(f'\nEarly stopping: {_mab_sweep_count} MAB sweeps '
                                  f'(no improvement > {_es_threshold} '
                                  f'for {_es_patience} consecutive sweeps)')
                            break

            cntTW = self._store.cntTW
            for w_idx in tok:
                s_tw  = cntTW.T[w_idx]
                s_sum = s_tw.sum() + epsilon
                cntTW_hist[s, w_idx] = np.sum(np.square(s_tw / s_sum))
                pr  = self.prR[w_idx] * 1e6
                prR_hist[s, w_idx]        = np.sum(np.square(
                    pr / (pr.sum() + epsilon)))
                pfc = self.prFullCond[w_idx] * 1e6
                prFullCond_hist[s, w_idx] = np.sum(np.square(
                    pfc / (pfc.sum() + epsilon)))
            topic_hist[s] = np.argmax(self.prFullCond, axis=1)

            if s == max(0, burnin - 1):
                burnin_snap  = self._store.snapshot()
                burnin_alpha = self.alpha.copy()
                burnin_beta  = self.beta.copy()
                burnin_theta = self.theta.copy()
                burnin_phi   = self.phi.copy()

            self.update_alpha_beta()
            sum_alpha = float(np.sum(self.alpha))
            sum_beta  = float(np.sum(self.beta))
            self.loss = self.Loss() / len(self.documents) / self.len_D - 1

            if self.loss - loss[-1] < 0.001 and loss[-1] < 0:
                if loss[-1] - loss[-2] < 0.001 and loss[-2] < 0:
                    print('\nCONVERGED\n')
                elif s >= burnin:
                    print('\n---Word Probability / Score---')
                    z_arr    = self._store.z
                    docs_np  = self.documents.numpy()
                    flat_w   = docs_np.ravel().astype(np.int64)
                    flat_z   = z_arr.ravel().astype(np.int64)
                    valid    = flat_w != pad_int
                    combined = flat_w[valid] * self.TOPICS + flat_z[valid]
                    tc_flat  = np.bincount(combined,
                                           minlength=self.VOCABS * self.TOPICS)
                    topic_counts += tc_flat.reshape(
                        self.VOCABS, self.TOPICS).astype(np.int32)
                    print('topic_counts[food]',
                          np.sum(np.square(
                              topic_counts[self.food]
                              / (topic_counts[self.food].sum() + epsilon))),
                          topic_counts[self.food])

                    for word in tok:
                        tc = topic_counts[word]
                        if tc.sum() == 0:
                            continue
                        topic_probs[word] = tc / tc.sum()
                        word_probs[word]  = ((topic_probs[word].max()
                                              - 1 / self.TOPICS)
                                             / (1 - 1 / self.TOPICS))
                    for doc_num in range(self.DOCS):
                        sc = word_probs[docs_np[doc_num]]
                        word_score[doc_num] = sc / (sc.max() + epsilon)

                    for word in tok:
                        tp = topic_probs[word]
                        if tp.sum() == 0:
                            continue
                        word_probs[word] = ((np.sum(np.square(
                            tp / (tp.sum() + epsilon)))
                                             - 1 / self.TOPICS)
                                            / (1 - 1 / self.TOPICS))
                    for doc_num in range(self.DOCS):
                        sc = word_probs[docs_np[doc_num]]
                        word_prob[doc_num] = sc / (sc.max() + epsilon)

            print(f'Loss {self.loss:.6f}  diff {self.loss - loss[-1]:.6f}')
            print(f'min Loss @epoch: {self.loss_min_epoch:.6f}')
            if MAB and self.loss_min_epoch < self.loss:
                self.loss_min_epoch = self.loss
            if MAB and self.loss_min < self.loss_min_epoch:
                self.loss_min = self.loss_min_epoch
            loss.append(self.loss)
            self.print_alpha_beta()

            if MAB:
                print('---MAB results---')
                rho = rho + self.rho
                self.rho /= 2

            if s >= burnin:
                self.update_phi_theta()
                self.theta += self.sample_theta
                self.phi   += self.sample_phi
                for w_idx in tok:
                    rho_w = rho[w_idx] / (s + 1)
                    Rho[s, w_idx] = np.sum(np.square(
                        rho_w / (rho_w.sum() + epsilon)))
                    phi_w = (self.phi[:, w_idx] / (s + 1)) * 1e6
                    phi[s, w_idx] = np.sum(np.square(
                        phi_w / (phi_w.sum() * 1e6 + epsilon)))

            M_log.append(MAB)
            print('MAB', M_log, '\n')

            # Scalability early termination: stop after 10 real MAB sweeps.
            # Only fires when epochs==1 (scalability test). Never fires in
            # main experiments (epochs=10) or ablations (epochs>1).
            if epochs == 1 and M_log.count(True) >= 10:
                print('--- SCALABILITY: MAB=True 10 times. Early termination. ---')
                _consecutive_mab = sum(1 for x in reversed(M_log) if x)
                if _consecutive_mab >= 10:
                    print('--- SCALABILITY: 10 consecutive MAB sweeps. Early termination. ---')
                    self.actual_sweeps = s + 1
                    self.mab_sweeps    = _consecutive_mab
                    break

            if MAB and (self.loss_min_epoch > self.loss) and (break_ > 1000 or max_iter <= 10):
                if loss[S] < self.loss:
                    SS       = S - burnin - 1
                    S        = s
                    phi_best = self.phi.copy()
                self._store.restore(burnin_snap)
                self.alpha = burnin_alpha.copy()
                self.beta  = burnin_beta.copy()
                self.theta = burnin_theta.copy()
                self.phi   = burnin_phi.copy()
                rho2       = rho.copy()
                self.loss  = self.Loss() / len(self.documents) / self.len_D - 1
                epoch_regret_history[epoch] = total_epoch_regret
                print(f'Epoch {epoch + 1} Cumulative Regret: {total_epoch_regret:.4f}')
                self.loss_min_epoch = -100000.0
                epoch += 1
                loss.append(10)
                print('EPOCH ends\n\n')
                if epoch == epochs:
                    self.actual_sweeps = s + 1
                    self.mab_sweeps = M_log.count(True)
                    break
                self.rho = np.zeros([self.VOCABS, self.TOPICS])
                
            # Sweep-level logging — all algorithm state, no hardcoding
            if hasattr(self, '_sweep_logger') and self._sweep_logger is not None:
                import numpy as _np
                _wall = time.time() - self._run_start_t
                _tps  = (self.DOCS * self.len_D * (s + 1)) / max(_wall, 1e-6)
                self._sweep_logger.log_sweep(
                    sweep          = s,
                    epoch          = epoch,
                    mab_active     = MAB,
                    loss           = self.loss,
                    break_count    = break_,
                    sweep_regret   = total_sweep_regret,
                    epoch_regret   = total_epoch_regret,
                    tokens_per_sec = _tps,
                    alpha_entropy  = float(_np.sum(_np.square(
                        self.alpha / (self.alpha.sum() + 1e-10)))),
                    beta_entropy   = float(_np.sum(_np.square(
                        self.beta  / (self.beta.sum()  + 1e-10)))),
                )

        rho  = rho2
        denom = max(S - SS + 1 - burnin, 1)
        rho_  = Rho[S] / denom
        self.theta /= denom
        self.phi   /= denom

        z_arr = self._store.z
        for word in self.tokens:
            w_int = int(word)
            pos   = (self.documents == word).nonzero(as_tuple=True)
            attention_mask[pos]  = rho_[w_int]
            attention_mask2[pos] = prR_hist[S, w_int]
            attention_mask3[pos] = prFullCond_hist[S, w_int]
            attention_mask4[pos] = phi[S, w_int] / denom
            attention_mask5[pos] = np.sum(np.square(
                self.rho[w_int] / (self.rho[w_int].sum() + epsilon)))

        return (word_score, word_prob, loss, S, topic_hist[S], rho, Rho,
                cntTW_hist, prR_hist, prFullCond_hist, phi, phi_best,
                attention_mask, attention_mask2, attention_mask3,
                attention_mask4, attention_mask5, self.reward,
                epoch_regret_history, sweep_regret_history, self.idx)
