// Lightweight Reinforcement Learning Refiner for k-way Hypergraph/Graph Partitioning.
//
// Formulates post-placement partition refinement as an episodic Markov Decision Process (MDP):
//   - State s_t: Current feasible k-way partition, vertex-to-partition incidence tables (cut weight,
//     cut edge count, route-through edge count, route-through weight), and FPGA resource slack.
//   - Action a_t: Selection of a capacity-feasible single-vertex move (u -> q) or pairwise swap (u <-> v)
//     from a priority-heap shortlist of top candidate moves.
//   - Policy pi_theta(a_t | s_t): Softmax policy over 12 normalized state-action features (gains on
//     cut weight, cut edges, route-through edges, route-through weight, move type, vertex degree
//     affinity, FPGA resource slack, and rollout phase).
//   - Training: Policy Gradient (REINFORCE with sliding-window baseline + Evolution Strategies)
//     trained on "/home/rajkumar/tr2/anpart/spectral_solver/random_graphs" and "examples".
//   - Inference: Deterministic argmax rollout with tabu tenure (allowing non-myopic hill-climbing
//     out of greedy 1-opt local minima) and exact prefix rollback to the best Pareto-valid state
//     (guaranteed non-worsening on total cut edges, total cut weight, and route-through edges).
#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace RL {

constexpr int kFeatDim = 12;
using FeatureVec = std::array<double, kFeatDim>;
using PolicyWeights = std::array<double, kFeatDim>;

// Default trained policy weights over the 12 state-action features:
//   f[0]: bias
//   f[1]: normalized delta in total cut weight (dCutW)
//   f[2]: normalized delta in total cut edges (dCutM)
//   f[3]: normalized delta in route-through cut edges (dRtM)
//   f[4]: normalized delta in route-through cut weight (dRtW)
//   f[5]: swap indicator (1 if pairwise swap, 0 if single move)
//   f[6]: target partition connectivity affinity
//   f[7]: source partition internal affinity (negative = loosely bound to current FPGA)
//   f[8]: target FPGA capacity slack ratio
//   f[9]: early-episode exploration interaction (gain * (1 - t/T))
//   f[10]: Pareto-safe indicator (1 if move immediately satisfies all global reference bounds)
//   f[11]: normalized vertex resource weight
inline PolicyWeights defaultPolicyWeights()
{
    return PolicyWeights{{
        0.00000000, // f[0] bias
        3.62592231, // f[1] dCutW (learned weight on total cut weight reduction)
        2.48727909, // f[2] dCutM (learned weight on total cut edge reduction)
        2.26525046, // f[3] dRtM  (learned weight on route-through edge reduction)
        1.46505002, // f[4] dRtW  (learned weight on route-through weight reduction)
        0.44912216, // f[5] swap indicator bonus
        0.61804877, // f[6] target partition connectivity affinity
       -0.16278768, // f[7] source partition internal affinity
       -0.01994723, // f[8] capacity slack ratio
        0.83110640, // f[9] early-episode gain emphasis
        1.25055585, // f[10] Pareto-safe state bonus
       -0.37849931  // f[11] relative vertex size penalty
    }};
}

inline std::string& policyFilePath()
{
    static std::string path = "rl_policy.weights";
    return path;
}

inline bool loadPolicyWeights(const std::string& path, PolicyWeights& w)
{
    std::ifstream in(path);
    if (!in.good())
        return false;
    std::string header;
    int dim = 0;
    if (!(in >> header >> dim) || header != "RL_MDP_POLICY_V3" || dim != kFeatDim)
        return false;
    for (int i = 0; i < kFeatDim; ++i) {
        if (!(in >> w[i]))
            return false;
    }
    return true;
}

inline bool savePolicyWeights(const std::string& path, const PolicyWeights& w)
{
    std::ofstream out(path);
    if (!out.good())
        return false;
    out << std::fixed << std::setprecision(8);
    out << "RL_MDP_POLICY_V3 " << kFeatDim << "\n";
    for (int i = 0; i < kFeatDim; ++i)
        out << w[i] << "\n";
    return true;
}

inline PolicyWeights activePolicy()
{
    PolicyWeights w = defaultPolicyWeights();
    loadPolicyWeights(policyFilePath(), w);
    return w;
}

struct Problem {
    unsigned n = 0;
    unsigned A = 0;
    std::vector<size_t> off;
    std::vector<unsigned> nbr;
    std::vector<double> wgt;
    std::vector<double> cnt;
    std::vector<unsigned> wl, wb, wd;
    std::vector<unsigned> capL, capB, capD;
    std::vector<char> linked; // A x A: 1 if connected or same FPGA, 0 if route-through
    double totW = 0.0;
    double totM = 0.0;
};

struct Metrics {
    double cutM = 0.0; // total cut edges (nrtCutEdges + rtCutEdges)
    double cutW = 0.0; // total cut weight (cutWt + rtCutWt)
    double rtM  = 0.0; // route-through cut edges
    double rtW  = 0.0; // route-through cut weight
};

inline Metrics evaluateMetrics(const Problem& P, const std::vector<unsigned>& pa)
{
    Metrics m;
    for (unsigned u = 0; u < P.n; ++u) {
        const unsigned pu = pa[u];
        for (size_t idx = P.off[u]; idx < P.off[u + 1]; ++idx) {
            const unsigned v = P.nbr[idx];
            if (v <= u)
                continue;
            const unsigned pv = pa[v];
            if (pu == pv)
                continue;
            const double c = P.cnt[idx];
            const double w = P.wgt[idx];
            m.cutM += c;
            m.cutW += w;
            if (!P.linked[(size_t)pu * P.A + pv]) {
                m.rtM += c;
                m.rtW += w;
            }
        }
    }
    return m;
}

struct TrajectoryStep {
    FeatureVec chosenFeat;
    FeatureVec expectedFeat;
    double reward = 0.0;
};

// Episodic RL Rollout Environment with Priority-Heap Shortlisting & Exact Prefix Rollback
class RolloutEnv {
public:
    RolloutEnv(const Problem& prob, const std::vector<unsigned>& initPa)
        : P(prob), n(prob.n), A(prob.A), pa(initPa)
    {
        nl.assign((size_t)A * A, 0.0);
        for (size_t i = 0; i < nl.size(); ++i)
            nl[i] = P.linked[i] ? 0.0 : 1.0;

        ldL.assign(A, 0);
        ldB.assign(A, 0);
        ldD.assign(A, 0);
        members.assign(A, {});
        pos.assign(n, 0);
        for (unsigned u = 0; u < n; ++u) {
            const unsigned a = pa[u];
            ldL[a] += P.wl[u];
            ldB[a] += P.wb[u];
            ldD[a] += P.wd[u];
            pos[u] = members[a].size();
            members[a].push_back(u);
        }

        connW.assign((size_t)n * A, 0.0);
        connM.assign((size_t)n * A, 0.0);
        degW.assign(n, 0.0);
        degM.assign(n, 0.0);

        curCutW = 0.0;
        curCutM = 0.0;
        curRtW  = 0.0;
        curRtM  = 0.0;

        for (unsigned u = 0; u < n; ++u) {
            const unsigned pu = pa[u];
            double* cw = &connW[(size_t)u * A];
            double* cm = &connM[(size_t)u * A];
            for (size_t idx = P.off[u]; idx < P.off[u + 1]; ++idx) {
                const unsigned v = P.nbr[idx];
                const unsigned pv = pa[v];
                const double w = P.wgt[idx];
                const double c = P.cnt[idx];
                cw[pv] += w;
                cm[pv] += c;
                degW[u] += w;
                degM[u] += c;
                if (v > u && pu != pv) {
                    curCutW += w;
                    curCutM += c;
                    if (nl[(size_t)pu * A + pv] > 0.5) {
                        curRtW += w;
                        curRtM += c;
                    }
                }
            }
        }

        rtMTable.assign((size_t)n * A, 0.0);
        rtWTable.assign((size_t)n * A, 0.0);
        for (unsigned u = 0; u < n; ++u) {
            const double* cm = &connM[(size_t)u * A];
            const double* cw = &connW[(size_t)u * A];
            double* rtm = &rtMTable[(size_t)u * A];
            double* rtw = &rtWTable[(size_t)u * A];
            for (unsigned t = 0; t < A; ++t) {
                const double* nlt = &nl[(size_t)t * A];
                double sm = 0.0, sw = 0.0;
                for (unsigned s = 0; s < A; ++s) {
                    if (nlt[s] > 0.5) {
                        sm += cm[s];
                        sw += cw[s];
                    }
                }
                rtm[t] = sm;
                rtw[t] = sw;
            }
        }

        scaleW  = 1.0 / std::max(1.0, curCutW / (double)n);
        scaleM  = 1.0 / std::max(1.0, curCutM / (double)n);
        scaleRM = 1.0 / std::max(1.0, std::max(curRtM, curCutM * 0.25) / (double)n);
        scaleRW = 1.0 / std::max(1.0, std::max(curRtW, curCutW * 0.25) / (double)n);

        relSize.assign(n, 0.0);
        double sumSz = 0.0;
        for (unsigned u = 0; u < n; ++u) {
            relSize[u] = std::max({
                P.wl[u] / (double)std::max(1u, P.capL[0]),
                P.wb[u] / (double)std::max(1u, P.capB[0]),
                P.wd[u] / (double)std::max(1u, P.capD[0])
            });
            sumSz += relSize[u];
        }
        avgRelSize = std::max(1e-9, sumSz / (double)n);

        lastMoved.assign(n, -1000000);
        heapKey.assign(n, 0.0);
        inHeap.assign(n, 0);
        stamp.assign(n, 0);
        stampCounter = 0;
        for (unsigned u = 0; u < n; ++u)
            refreshVertex(u);
    }

    struct CandMove {
        unsigned u, p, q, v; // v == UINT_MAX for single move
        double dW, dM, dRtM, dRtW;
        double proxyGain;
    };

    bool canMove(unsigned u, unsigned q) const
    {
        return ldL[q] + P.wl[u] <= P.capL[q] &&
               ldB[q] + P.wb[u] <= P.capB[q] &&
               ldD[q] + P.wd[u] <= P.capD[q];
    }

    bool canSwap(unsigned u, unsigned p, unsigned v, unsigned q) const
    {
        return ldL[p] - P.wl[u] + P.wl[v] <= P.capL[p] &&
               ldB[p] - P.wb[u] + P.wb[v] <= P.capB[p] &&
               ldD[p] - P.wd[u] + P.wd[v] <= P.capD[p] &&
               ldL[q] - P.wl[v] + P.wl[u] <= P.capL[q] &&
               ldB[q] - P.wb[v] + P.wb[u] <= P.capB[q] &&
               ldD[q] - P.wd[v] + P.wd[u] <= P.capD[q];
    }

    inline void singleDeltas(unsigned u, unsigned q, double& dW, double& dM, double& dRtM, double& dRtW) const
    {
        const unsigned p = pa[u];
        const size_t b = (size_t)u * A;
        dW   = connW[b + q] - connW[b + p];
        dM   = connM[b + q] - connM[b + p];
        dRtM = rtMTable[b + p] - rtMTable[b + q];
        dRtW = rtWTable[b + p] - rtWTable[b + q];
    }

    inline void findEdge(unsigned u, unsigned v, double& ew, double& ec) const
    {
        auto b = P.nbr.begin() + P.off[u];
        auto e = P.nbr.begin() + P.off[u + 1];
        auto it = std::lower_bound(b, e, v);
        if (it != e && *it == v) {
            const size_t idx = it - P.nbr.begin();
            ew = P.wgt[idx];
            ec = P.cnt[idx];
        } else {
            ew = 0.0;
            ec = 0.0;
        }
    }

    double optimisticKey(unsigned u) const
    {
        const unsigned p = pa[u];
        const size_t b = (size_t)u * A;
        if (degM[u] - connM[b + p] <= 0.0)
            return -1e30;
        double best = -1e30;
        for (unsigned q = 0; q < A; ++q) {
            if (q == p)
                continue;
            double dW, dM, dRtM, dRtW;
            singleDeltas(u, q, dW, dM, dRtM, dRtW);
            const double g = wBiasW * scaleW * dW + wBiasM * scaleM * dM + wBiasRM * scaleRM * dRtM;
            if (g > best)
                best = g;
        }
        return best;
    }

    void refreshVertex(unsigned u)
    {
        if (inHeap[u])
            heap.erase({heapKey[u], u});
        heapKey[u] = optimisticKey(u);
        if (heapKey[u] > -1e20) {
            heap.insert({heapKey[u], u});
            inHeap[u] = 1;
        } else {
            inHeap[u] = 0;
        }
    }

    void applyMove(unsigned u, unsigned q)
    {
        const unsigned p = pa[u];
        if (p == q)
            return;
        double dW, dM, dRtM, dRtW;
        singleDeltas(u, q, dW, dM, dRtM, dRtW);
        curCutW -= dW;
        curCutM -= dM;
        curRtM  -= dRtM;
        curRtW  -= dRtW;

        const double* nlp = &nl[(size_t)p * A];
        const double* nlq = &nl[(size_t)q * A];
        for (size_t idx = P.off[u]; idx < P.off[u + 1]; ++idx) {
            const unsigned x = P.nbr[idx];
            const double w = P.wgt[idx];
            const double c = P.cnt[idx];
            const size_t bx = (size_t)x * A;
            connW[bx + p] -= w;
            connW[bx + q] += w;
            connM[bx + p] -= c;
            connM[bx + q] += c;
            double* rtm = &rtMTable[bx];
            double* rtw = &rtWTable[bx];
            for (unsigned t = 0; t < A; ++t) {
                const double diff = nlq[t] - nlp[t];
                if (diff != 0.0) {
                    rtm[t] += c * diff;
                    rtw[t] += w * diff;
                }
            }
        }

        ldL[p] -= P.wl[u]; ldB[p] -= P.wb[u]; ldD[p] -= P.wd[u];
        ldL[q] += P.wl[u]; ldB[q] += P.wb[u]; ldD[q] += P.wd[u];

        const unsigned lastV = members[p].back();
        members[p][pos[u]] = lastV;
        pos[lastV] = pos[u];
        members[p].pop_back();
        pos[u] = members[q].size();
        members[q].push_back(u);
        pa[u] = q;
    }

    void generateVertexCandidates(unsigned u, long step, int tenure, std::vector<CandMove>& out) const
    {
        const unsigned p = pa[u];
        const size_t bu = (size_t)u * A;
        if (degM[u] - connM[bu + p] <= 0.0)
            return;

        CandMove bestSingle{u, p, UINT_MAX, UINT_MAX, 0, 0, 0, 0, -1e300};
        std::array<std::pair<double, unsigned>, 5> topQ;
        size_t nTopQ = 0;

        for (unsigned q = 0; q < A; ++q) {
            if (q == p)
                continue;
            if (connM[bu + q] == 0.0 && rtMTable[bu + p] == 0.0)
                continue;
            double dW, dM, dRtM, dRtW;
            singleDeltas(u, q, dW, dM, dRtM, dRtW);
            const double pg = wBiasW * scaleW * dW + wBiasM * scaleM * dM + wBiasRM * scaleRM * dRtM;
            if (canMove(u, q) && pg > bestSingle.proxyGain) {
                bestSingle = CandMove{u, p, q, UINT_MAX, dW, dM, dRtM, dRtW, pg};
            }
            // Rank swap target partitions by raw connection affinity to u plus single-move proxy
            const double qAff = pg + 0.25 * scaleW * connW[bu + q];
            if (nTopQ < topQ.size()) {
                topQ[nTopQ++] = {qAff, q};
            } else {
                size_t mIdx = 0;
                for (size_t k = 1; k < nTopQ; ++k)
                    if (topQ[k].first < topQ[mIdx].first)
                        mIdx = k;
                if (qAff > topQ[mIdx].first)
                    topQ[mIdx] = {qAff, q};
            }
        }
        if (bestSingle.q != UINT_MAX)
            out.push_back(bestSingle);

        std::sort(topQ.begin(), topQ.begin() + nTopQ,
                  [](const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) {
                      return a.first > b.first;
                  });
        CandMove bestSwap{u, p, UINT_MAX, UINT_MAX, 0, 0, 0, 0, -1e300};
        const size_t maxQ = std::min<size_t>(nTopQ, 3);
        for (size_t qi = 0; qi < maxQ; ++qi) {
            const unsigned q = topQ[qi].second;
            double uDW, uDM, uDRtM, uDRtW;
            singleDeltas(u, q, uDW, uDM, uDRtM, uDRtW);
            const double nlPQ = nl[(size_t)p * A + q];
            for (unsigned v : members[q]) {
                if (step - lastMoved[v] < tenure || !canSwap(u, p, v, q))
                    continue;
                double vDW, vDM, vDRtM, vDRtW;
                singleDeltas(v, p, vDW, vDM, vDRtM, vDRtW);
                double ew = 0.0, ec = 0.0;
                findEdge(u, v, ew, ec);
                const double tDW   = uDW + vDW - 2.0 * ew;
                const double tDM   = uDM + vDM - 2.0 * ec;
                const double tDRtM = uDRtM + vDRtM - 2.0 * ec * nlPQ;
                const double tDRtW = uDRtW + vDRtW - 2.0 * ew * nlPQ;
                const double pg = wBiasW * scaleW * tDW + wBiasM * scaleM * tDM + wBiasRM * scaleRM * tDRtM;
                if (pg > bestSwap.proxyGain) {
                    bestSwap = CandMove{u, p, q, v, tDW, tDM, tDRtM, tDRtW, pg};
                }
            }
        }
        if (bestSwap.q != UINT_MAX)
            out.push_back(bestSwap);
    }

    void buildShortlist(long step, int tenure, size_t K, std::vector<CandMove>& cands) const
    {
        cands.clear();
        size_t scanned = 0;
        for (auto it = heap.begin(); it != heap.end() && scanned < 3 * K; ++it) {
            const unsigned u = it->second;
            if (step - lastMoved[u] < tenure)
                continue;
            ++scanned;
            generateVertexCandidates(u, step, tenure, cands);
        }
        std::sort(cands.begin(), cands.end(), [](const CandMove& a, const CandMove& b) {
            if (a.proxyGain != b.proxyGain)
                return a.proxyGain > b.proxyGain;
            return a.u != b.u ? a.u < b.u : a.q < b.q;
        });
        if (cands.size() > K)
            cands.resize(K);
    }

    void computeFeatures(const CandMove& c, long step, size_t maxSteps,
                         double refW, double refM, double refRM, FeatureVec& f) const
    {
        auto clip = [](double x) { return std::max(-4.0, std::min(4.0, x)); };
        const double nDW  = clip(scaleW * c.dW);
        const double nDM  = clip(scaleM * c.dM);
        const double nDRM = clip(scaleRM * c.dRtM);
        const double nDRW = clip(scaleRW * c.dRtW);

        f[0] = 1.0;
        f[1] = nDW;
        f[2] = nDM;
        f[3] = nDRM;
        f[4] = nDRW;
        f[5] = (c.v == UINT_MAX) ? 0.0 : 1.0;

        const double dw = std::max(1.0, degW[c.u]);
        f[6] = connW[(size_t)c.u * A + c.q] / dw;
        f[7] = connW[(size_t)c.u * A + c.p] / dw;

        const double slackL = 1.0 - (double)ldL[c.q] / (double)std::max(1u, P.capL[c.q]);
        f[8] = std::max(0.0, std::min(1.0, slackL));

        const double progress = 1.0 - (double)step / (double)std::max<size_t>(1, maxSteps);
        f[9] = (0.4 * nDW + 0.35 * nDM + 0.25 * nDRM) * progress;

        const bool paretoSafe = (curCutW - c.dW <= refW + 1e-6) &&
                                (curCutM - c.dM <= refM + 1e-6) &&
                                (curRtM - c.dRtM <= refRM + 1e-6);
        f[10] = paretoSafe ? 1.0 : 0.0;
        f[11] = std::min(3.0, relSize[c.u] / avgRelSize);
    }

    // Run an RL episode with non-myopic hill-climbing + exact prefix rollback
    double runEpisode(const PolicyWeights& theta, std::mt19937* rng, std::vector<TrajectoryStep>* traceOut = nullptr)
    {
        const double startW  = curCutW;
        const double startM  = curCutM;
        const double startRM = curRtM;

        const double invW  = 1.0 / std::max(1.0, startW);
        const double invM  = 1.0 / std::max(1.0, startM);
        const double invRM = 1.0 / std::max(1.0, std::max(startRM, startM * 0.1));

        auto combinedObj = [&]() -> double {
            return 0.45 * (curCutW * invW) + 0.35 * (curCutM * invM) + 0.20 * (curRtM * invRM);
        };

        const double initObj = combinedObj();
        double bestValidObj = initObj;
        double bestAnyObj = initObj;
        size_t bestPrefix = 0;
        size_t sinceBest = 0;

        const size_t maxSteps = std::min<size_t>(2 * (size_t)n, 3500);
        const size_t patience = std::min<size_t>((size_t)n / 2 + 120, 450);
        const int tenure = 18;
        const size_t shortlistK = 20;

        moveLog.clear();
        std::vector<CandMove> cands;
        std::vector<FeatureVec> feats;
        std::vector<double> probs;

        for (size_t step = 0; step < maxSteps && sinceBest < patience; ++step) {
            buildShortlist((long)step, tenure, shortlistK, cands);
            if (cands.empty())
                break;

            feats.resize(cands.size());
            probs.resize(cands.size());
            double maxLogit = -1e300;
            for (size_t i = 0; i < cands.size(); ++i) {
                computeFeatures(cands[i], (long)step, maxSteps, startW, startM, startRM, feats[i]);
                double z = 0.0;
                for (int k = 0; k < kFeatDim; ++k)
                    z += theta[k] * feats[i][k];
                probs[i] = z;
                if (z > maxLogit)
                    maxLogit = z;
            }
            double sumExp = 0.0;
            for (size_t i = 0; i < cands.size(); ++i) {
                probs[i] = std::exp(std::max(-40.0, probs[i] - maxLogit));
                sumExp += probs[i];
            }
            for (size_t i = 0; i < cands.size(); ++i)
                probs[i] /= sumExp;

            size_t chosen = 0;
            if (rng) {
                std::uniform_real_distribution<double> U(0.0, 1.0);
                double r = U(*rng), acc = 0.0;
                chosen = cands.size() - 1;
                for (size_t i = 0; i < cands.size(); ++i) {
                    acc += probs[i];
                    if (r <= acc) {
                        chosen = i;
                        break;
                    }
                }
            } else {
                for (size_t i = 1; i < cands.size(); ++i) {
                    if (probs[i] > probs[chosen] + 1e-12)
                        chosen = i;
                }
            }

            const CandMove mv = cands[chosen];
            const double prevObj = combinedObj();

            // Execute chosen move or swap
            moveLog.emplace_back(mv.u, mv.p);
            applyMove(mv.u, mv.q);
            lastMoved[mv.u] = (long)step;
            if (mv.v != UINT_MAX) {
                moveLog.emplace_back(mv.v, mv.q);
                applyMove(mv.v, mv.p);
                lastMoved[mv.v] = (long)step;
            }

            // Refresh affected neighborhood in priority heap
            ++stampCounter;
            auto touchNeighbors = [&](unsigned vert) {
                for (size_t idx = P.off[vert]; idx < P.off[vert + 1]; ++idx) {
                    const unsigned nb = P.nbr[idx];
                    if (stamp[nb] != stampCounter) {
                        stamp[nb] = stampCounter;
                        refreshVertex(nb);
                    }
                }
            };
            touchNeighbors(mv.u);
            if (mv.v != UINT_MAX)
                touchNeighbors(mv.v);
            refreshVertex(mv.u);
            if (mv.v != UINT_MAX)
                refreshVertex(mv.v);

            const double newObj = combinedObj();
            const double stepReward = 100.0 * (prevObj - newObj);

            if (traceOut) {
                FeatureVec expF{};
                expF.fill(0.0);
                for (size_t i = 0; i < cands.size(); ++i)
                    for (int k = 0; k < kFeatDim; ++k)
                        expF[k] += probs[i] * feats[i][k];
                traceOut->push_back(TrajectoryStep{feats[chosen], expF, stepReward});
            }

            // Check if current state is Pareto-valid (non-worse than episode start on all three metrics)
            const bool validAll = (curCutW <= startW + 1e-6) &&
                                  (curCutM <= startM + 1e-6) &&
                                  (curRtM  <= startRM + 1e-6);
            if (validAll && newObj < bestValidObj - 1e-12) {
                bestValidObj = newObj;
                bestPrefix = moveLog.size();
            }
            if (newObj < bestAnyObj - 1e-12) {
                bestAnyObj = newObj;
                sinceBest = 0;
            } else {
                ++sinceBest;
            }
        }

        // Exact prefix rollback to best Pareto-valid state
        while (moveLog.size() > bestPrefix) {
            applyMove(moveLog.back().first, moveLog.back().second);
            moveLog.pop_back();
        }
        return 100.0 * (initObj - bestValidObj);
    }

    // Polish pass: executes policy-scored single moves, pairwise swaps, and 3-way cyclic Kempe-chain
    // rotations (u in p -> q, v in q -> r, w in r -> p) that strictly improve the RL policy objective
    // while keeping (curCutW, curCutM, curRtM) <= reference bounds.
    bool runParetoKempePolish(const PolicyWeights& theta, double refW, double refM, double refRM)
    {
        bool anyChanged = false;
        const double wW  = std::max(0.1, theta[1]) * scaleW;
        const double wM  = std::max(0.05, theta[2]) * scaleM;
        const double wRM = std::max(0.05, theta[3]) * scaleRM;
        const double wRW = std::max(0.0, theta[4]) * scaleRW;

        auto scoreMove = [&](double dW, double dM, double dRtM, double dRtW) -> double {
            return wW * dW + wM * dM + wRM * dRtM + wRW * dRtW;
        };

        auto isSafe = [&](double newW, double newM, double newRM) -> bool {
            return newW <= refW + 1e-6 && newM <= refM + 1e-6 && newRM <= refRM + 1e-6;
        };

        std::vector<unsigned> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
            return degW[a] != degW[b] ? degW[a] > degW[b] : a < b;
        });

        // 1. Exhaustive single-move & pairwise-swap Pareto sweep
        for (unsigned u : order) {
            const unsigned p = pa[u];
            const size_t bu = (size_t)u * A;
            if (degM[u] - connM[bu + p] <= 0.0)
                continue;

            double bestScore = 1e-12;
            unsigned bestQ = UINT_MAX;
            unsigned bestV = UINT_MAX;

            std::array<std::pair<double, unsigned>, 5> candQs;
            size_t nQ = 0;
            for (unsigned q = 0; q < A; ++q) {
                if (q == p)
                    continue;
                if (connM[bu + q] == 0.0 && rtMTable[bu + p] == 0.0)
                    continue;
                double dW, dM, dRtM, dRtW;
                singleDeltas(u, q, dW, dM, dRtM, dRtW);
                const double s = scoreMove(dW, dM, dRtM, dRtW);
                if (canMove(u, q) && isSafe(curCutW - dW, curCutM - dM, curRtM - dRtM) && s > bestScore) {
                    bestScore = s;
                    bestQ = q;
                    bestV = UINT_MAX;
                }
                const double aff = s + 0.3 * wW * connW[bu + q];
                if (nQ < candQs.size()) {
                    candQs[nQ++] = {aff, q};
                } else {
                    size_t mIdx = 0;
                    for (size_t k = 1; k < nQ; ++k)
                        if (candQs[k].first < candQs[mIdx].first)
                            mIdx = k;
                    if (aff > candQs[mIdx].first)
                        candQs[mIdx] = {aff, q};
                }
            }

            std::sort(candQs.begin(), candQs.begin() + nQ,
                      [](const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) {
                          return a.first > b.first;
                      });
            const bool checkAllQ = (bestQ == UINT_MAX);
            const size_t maxQSearch = checkAllQ ? A : nQ;
            for (size_t qi = 0; qi < maxQSearch; ++qi) {
                const unsigned q = checkAllQ ? static_cast<unsigned>(qi) : candQs[qi].second;
                if (q == p || (connM[bu + q] == 0.0 && rtMTable[bu + p] == 0.0))
                    continue;
                double uDW, uDM, uDRtM, uDRtW;
                singleDeltas(u, q, uDW, uDM, uDRtM, uDRtW);
                const double nlPQ = nl[(size_t)p * A + q];
                for (unsigned v : members[q]) {
                    if (!canSwap(u, p, v, q))
                        continue;
                    double vDW, vDM, vDRtM, vDRtW;
                    singleDeltas(v, p, vDW, vDM, vDRtM, vDRtW);
                    double ew = 0.0, ec = 0.0;
                    findEdge(u, v, ew, ec);
                    const double tDW   = uDW + vDW - 2.0 * ew;
                    const double tDM   = uDM + vDM - 2.0 * ec;
                    const double tDRtM = uDRtM + vDRtM - 2.0 * ec * nlPQ;
                    const double tDRtW = uDRtW + vDRtW - 2.0 * ew * nlPQ;
                    if (isSafe(curCutW - tDW, curCutM - tDM, curRtM - tDRtM)) {
                        const double s = scoreMove(tDW, tDM, tDRtM, tDRtW);
                        if (s > bestScore) {
                            bestScore = s;
                            bestQ = q;
                            bestV = v;
                        }
                    }
                }
            }

            if (bestQ != UINT_MAX) {
                applyMove(u, bestQ);
                if (bestV != UINT_MAX)
                    applyMove(bestV, p);
                anyChanged = true;
            }
        }

        // 2. 3-way cyclic Kempe-chain rotation (u in p -> q, v in q -> r, w in r -> p) for tightly packed FPGAs
        for (unsigned u : order) {
            const unsigned p = pa[u];
            const size_t bu = (size_t)u * A;
            if (degM[u] - connM[bu + p] <= 0.0)
                continue;

            unsigned bestQ = UINT_MAX, bestV = UINT_MAX, bestR = UINT_MAX, bestW = UINT_MAX;
            double bestCycleGain = 1e-12;

            // Consider top-3 target partitions q for u
            std::array<std::pair<double, unsigned>, 3> topQ;
            size_t nTopQ = 0;
            for (unsigned q = 0; q < A; ++q) {
                if (q == p || connM[bu + q] == 0.0)
                    continue;
                double uDW, uDM, uDRtM, uDRtW;
                singleDeltas(u, q, uDW, uDM, uDRtM, uDRtW);
                const double sU = scoreMove(uDW, uDM, uDRtM, uDRtW);
                if (nTopQ < topQ.size()) {
                    topQ[nTopQ++] = {sU, q};
                } else {
                    size_t mIdx = 0;
                    for (size_t k = 1; k < nTopQ; ++k)
                        if (topQ[k].first < topQ[mIdx].first)
                            mIdx = k;
                    if (sU > topQ[mIdx].first)
                        topQ[mIdx] = {sU, q};
                }
            }
            std::sort(topQ.begin(), topQ.begin() + nTopQ,
                      [](const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) {
                          return a.first > b.first;
                      });

            for (size_t qi = 0; qi < nTopQ; ++qi) {
                const unsigned q = topQ[qi].second;
                double uDW, uDM, uDRtM, uDRtW;
                singleDeltas(u, q, uDW, uDM, uDRtM, uDRtW);

                for (unsigned v : members[q]) {
                    if (ldL[q] - P.wl[v] + P.wl[u] > P.capL[q] ||
                        ldB[q] - P.wb[v] + P.wb[u] > P.capB[q] ||
                        ldD[q] - P.wd[v] + P.wd[u] > P.capD[q])
                        continue;
                    double ewUV = 0.0, ecUV = 0.0;
                    findEdge(u, v, ewUV, ecUV);
                    if (ecUV > 0.0)
                        continue; // independent vertices for exact additive delta
                    const size_t bv = (size_t)v * A;
                    for (unsigned r = 0; r < A; ++r) {
                        if (r == p || r == q || connM[bv + r] == 0.0)
                            continue;
                        double vDW, vDM, vDRtM, vDRtW;
                        singleDeltas(v, r, vDW, vDM, vDRtM, vDRtW);
                        if (uDW + vDW < -50.0 && uDM + vDM < -2.0)
                            continue;
                        for (unsigned w : members[r]) {
                            if (ldL[r] - P.wl[w] + P.wl[v] > P.capL[r] ||
                                ldB[r] - P.wb[w] + P.wb[v] > P.capB[r] ||
                                ldD[r] - P.wd[w] + P.wd[v] > P.capD[r] ||
                                ldL[p] - P.wl[u] + P.wl[w] > P.capL[p] ||
                                ldB[p] - P.wb[u] + P.wb[w] > P.capB[p] ||
                                ldD[p] - P.wd[u] + P.wd[w] > P.capD[p])
                                continue;
                            double ewUW = 0.0, ecUW = 0.0, ewVW = 0.0, ecVW = 0.0;
                            findEdge(u, w, ewUW, ecUW);
                            if (ecUW > 0.0)
                                continue;
                            findEdge(v, w, ewVW, ecVW);
                            if (ecVW > 0.0)
                                continue;
                            double wDW, wDM, wDRtM, wDRtW;
                            singleDeltas(w, p, wDW, wDM, wDRtM, wDRtW);
                            const double totDW = uDW + vDW + wDW;
                            const double totDM = uDM + vDM + wDM;
                            const double totDRM = uDRtM + vDRtM + wDRtM;
                            const double totDRW = uDRtW + vDRtW + wDRtW;
                            if (isSafe(curCutW - totDW, curCutM - totDM, curRtM - totDRM)) {
                                const double cycS = scoreMove(totDW, totDM, totDRM, totDRW);
                                if (cycS > bestCycleGain) {
                                    bestCycleGain = cycS;
                                    bestQ = q;
                                    bestV = v;
                                    bestR = r;
                                    bestW = w;
                                }
                            }
                        }
                    }
                }
            }

            if (bestQ != UINT_MAX && bestV != UINT_MAX && bestW != UINT_MAX) {
                // Execute 3-way rotation: u(p->q), v(q->r), w(r->p)
                applyMove(u, bestQ);
                applyMove(bestV, bestR);
                applyMove(bestW, p);
                anyChanged = true;
            }
        }

        return anyChanged;
    }

    const Problem& P;
    unsigned n, A;
    std::vector<unsigned> pa;
    std::vector<double> nl;
    std::vector<unsigned> ldL, ldB, ldD, pos;
    std::vector<std::vector<unsigned>> members;
    std::vector<double> connW, connM, degW, degM, rtMTable, rtWTable, relSize;
    double curCutW = 0.0, curCutM = 0.0, curRtW = 0.0, curRtM = 0.0;
    double scaleW = 1.0, scaleM = 1.0, scaleRM = 1.0, scaleRW = 1.0, avgRelSize = 1.0;
    double wBiasW = 0.50, wBiasM = 0.30, wBiasRM = 0.20;

    struct HeapCmp {
        bool operator()(const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) const
        {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        }
    };
    std::set<std::pair<double, unsigned>, HeapCmp> heap;
    std::vector<double> heapKey;
    std::vector<char> inHeap;
    std::vector<long> lastMoved;
    std::vector<unsigned> stamp;
    unsigned stampCounter = 0;
    std::vector<std::pair<unsigned, unsigned>> moveLog;
};

// Inference entry point: runs multi-episode deterministic RL rollout refinement + Kempe-chain Pareto polish
inline void refinePartition(const Problem& P, std::vector<unsigned>& pa, const PolicyWeights& theta,
                            Metrics* beforeOut = nullptr, Metrics* afterOut = nullptr, int maxEpisodes = 4)
{
    if (beforeOut)
        *beforeOut = evaluateMetrics(P, pa);
    if (P.A < 2 || P.n < 2 || P.totM <= 0.0 || P.totW <= 0.0) {
        if (afterOut)
            *afterOut = evaluateMetrics(P, pa);
        return;
    }

    // Episode weight schedules derived from the trained policy parameters
    const double sumT = std::max(1e-6, std::abs(theta[1]) + std::abs(theta[2]) + std::abs(theta[3]));
    const double baseW  = std::max(0.15, std::abs(theta[1]) / sumT);
    const double baseM  = std::max(0.10, std::abs(theta[2]) / sumT);
    const double baseRM = std::max(0.10, std::abs(theta[3]) / sumT);

    const std::array<std::array<double, 3>, 4> schedules = {{
        {{baseW, baseM, baseRM}},
        {{0.65,  0.20,  0.15}},
        {{0.30,  0.45,  0.25}},
        {{0.35,  0.25,  0.40}}
    }};

    const Metrics globalInit = evaluateMetrics(P, pa);

    for (int ep = 0; ep < maxEpisodes; ++ep) {
        RolloutEnv env(P, pa);
        const auto& sc = schedules[ep % schedules.size()];
        env.wBiasW  = sc[0];
        env.wBiasM  = sc[1];
        env.wBiasRM = sc[2];
        for (unsigned u = 0; u < P.n; ++u)
            env.refreshVertex(u);

        // Allow constructive trade-offs within the episode as long as we never exceed globalInit
        const double refW  = 0.25 * env.curCutW + 0.75 * globalInit.cutW;
        const double refM  = 0.25 * env.curCutM + 0.75 * globalInit.cutM;
        const double refRM = 0.25 * env.curRtM  + 0.75 * globalInit.rtM;

        const double gainPct = env.runEpisode(theta, nullptr, nullptr);
        const bool polished = env.runParetoKempePolish(theta, globalInit.cutW, globalInit.cutM, globalInit.rtM);
        const bool polished2 = env.runParetoKempePolish(theta, refW, refM, refRM);
        if (env.curCutW <= globalInit.cutW + 1e-6 &&
            env.curCutM <= globalInit.cutM + 1e-6 &&
            env.curRtM  <= globalInit.rtM  + 1e-6) {
            pa = env.pa;
        }
        if (gainPct <= 1e-6 && !polished && !polished2 && ep >= 1)
            break;
    }

    if (afterOut)
        *afterOut = evaluateMetrics(P, pa);
}

struct TrainInstance {
    std::string name;
    Problem P;
    std::vector<unsigned> initPa;
};

// Combined percentage reduction across cutEdges, cutWeight, and route-through edges
inline double computeReward(const Metrics& before, const Metrics& after)
{
    const double dW  = (before.cutW > 0.0) ? 100.0 * (before.cutW - after.cutW) / before.cutW : 0.0;
    const double dM  = (before.cutM > 0.0) ? 100.0 * (before.cutM - after.cutM) / before.cutM : 0.0;
    const double dRM = (before.rtM > 0.0)  ? 100.0 * (before.rtM  - after.rtM)  / before.rtM  : dM;
    return 0.40 * dW + 0.35 * dM + 0.25 * dRM;
}

// Train the RL policy using REINFORCE with sliding-window baseline + Evolution Strategies (ES) polish
inline PolicyWeights trainPolicyCorpus(const std::vector<TrainInstance>& instances, int epochs, double lr,
                                       const std::string& outPath)
{
    PolicyWeights theta = defaultPolicyWeights();
    if (instances.empty()) {
        savePolicyWeights(outPath, theta);
        return theta;
    }

    auto evalCorpus = [&](const PolicyWeights& w) -> double {
        double tot = 0.0;
        for (const auto& inst : instances) {
            std::vector<unsigned> pa = inst.initPa;
            Metrics b, a;
            refinePartition(inst.P, pa, w, &b, &a, 2);
            tot += computeReward(b, a);
        }
        return tot / (double)instances.size();
    };

    PolicyWeights bestTheta = theta;
    double bestScore = evalCorpus(theta);
    std::cout << "RL Train Initial Policy Validation Reward: +"
              << std::fixed << std::setprecision(4) << bestScore << "%" << std::endl;

    std::mt19937 rng(42);
    PolicyWeights mom{}, vel{};
    mom.fill(0.0);
    vel.fill(0.0);

    // Phase 1: Episodic REINFORCE Policy Gradient
    const int pgEpochs = std::max(1, epochs / 2);
    constexpr double kGamma = 0.92;
    constexpr size_t kWin = 48;

    for (int ep = 1; ep <= pgEpochs; ++ep) {
        PolicyWeights grad{};
        grad.fill(0.0);
        size_t nTraj = 0;

        for (const auto& inst : instances) {
            RolloutEnv env(inst.P, inst.initPa);
            std::vector<TrajectoryStep> trace;
            env.runEpisode(theta, &rng, &trace);
            const size_t T = trace.size();
            if (T == 0)
                continue;
            ++nTraj;
            std::vector<double> ret(T, 0.0);
            double acc = 0.0;
            for (size_t t = T; t-- > 0;) {
                acc = trace[t].reward + kGamma * acc;
                ret[t] = acc;
            }
            std::vector<double> pref(T + 1, 0.0);
            for (size_t t = 0; t < T; ++t)
                pref[t + 1] = pref[t] + ret[t];

            for (size_t t = 0; t < T; ++t) {
                const size_t lo = (t > kWin) ? (t - kWin) : 0;
                const size_t hi = std::min(T, t + kWin + 1);
                const double base = (pref[hi] - pref[lo]) / (double)(hi - lo);
                const double adv = ret[t] - base;
                for (int k = 0; k < kFeatDim; ++k) {
                    grad[k] += adv * (trace[t].chosenFeat[k] - trace[t].expectedFeat[k]) / (double)T;
                }
            }
        }

        if (nTraj > 0) {
            for (int k = 0; k < kFeatDim; ++k) {
                const double g = grad[k] / (double)nTraj;
                mom[k] = 0.9 * mom[k] + 0.1 * g;
                vel[k] = 0.999 * vel[k] + 0.001 * g * g;
                const double mh = mom[k] / (1.0 - std::pow(0.9, ep));
                const double vh = vel[k] / (1.0 - std::pow(0.999, ep));
                theta[k] += lr * mh / (std::sqrt(vh) + 1e-8);
            }
        }

        const double score = evalCorpus(theta);
        if (score > bestScore + 1e-6) {
            bestScore = score;
            bestTheta = theta;
            savePolicyWeights(outPath, bestTheta);
        }
        std::cout << "RL PG Epoch " << std::setw(2) << ep
                  << " | Corpus Multi-Metric Reduction: +" << std::fixed << std::setprecision(4)
                  << score << "% (best: +" << bestScore << "%)" << std::endl;
    }

    // Phase 2: Antithetic Evolution Strategies (ES) directly optimizing deterministic rollout reward
    theta = bestTheta;
    std::normal_distribution<double> N01(0.0, 1.0);
    const int esGens = std::max(1, epochs - pgEpochs);
    const double sigma = 0.25;
    for (int gen = 1; gen <= esGens; ++gen) {
        PolicyWeights eps;
        for (int k = 0; k < kFeatDim; ++k)
            eps[k] = sigma * N01(rng);
        PolicyWeights wPlus = theta, wMinus = theta;
        for (int k = 0; k < kFeatDim; ++k) {
            wPlus[k]  += eps[k];
            wMinus[k] -= eps[k];
        }
        const double sPlus = evalCorpus(wPlus);
        if (sPlus > bestScore + 1e-6) {
            bestScore = sPlus;
            bestTheta = wPlus;
            theta = wPlus;
            savePolicyWeights(outPath, bestTheta);
        } else {
            const double sMinus = evalCorpus(wMinus);
            if (sMinus > bestScore + 1e-6) {
                bestScore = sMinus;
                bestTheta = wMinus;
                theta = wMinus;
                savePolicyWeights(outPath, bestTheta);
            }
        }
        std::cout << "RL ES Gen   " << std::setw(2) << gen
                  << " | Best Corpus Multi-Metric Reduction: +" << std::fixed << std::setprecision(4)
                  << bestScore << "%" << std::endl;
    }

    for (const auto& inst : instances) {
        std::vector<unsigned> pa = inst.initPa;
        Metrics b, a;
        refinePartition(inst.P, pa, bestTheta, &b, &a, 3);
        const double r = computeReward(b, a);
        std::cout << "  [Trained RL on " << inst.name << "] "
                  << "cutEdges: " << (size_t)b.cutM << " -> " << (size_t)a.cutM
                  << " | cutWt: " << (size_t)b.cutW << " -> " << (size_t)a.cutW
                  << " | rtEdges: " << (size_t)b.rtM << " -> " << (size_t)a.rtM
                  << " | reward: +" << std::fixed << std::setprecision(3) << r << "%" << std::endl;
    }

    savePolicyWeights(outPath, bestTheta);
    return bestTheta;
}

} // namespace RL
