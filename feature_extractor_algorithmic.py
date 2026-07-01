#!/usr/bin/env python
"""AlgorithmicMixin for FeatureExtractor."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

class AlgorithmicMixin:
    def extract_algorithmic_annotations_features(self, algo_data):
        """
        Extract sleep architecture, fragmentation, bout, dynamics, periodicity,
        stability, uncertainty, and composite features from CAISR outputs.
        Output vector length: 84.
        """
        if not algo_data:
            return np.zeros(84, dtype=np.float32)

        EPOCH_SEC = 30.0
        WAKE = 5
        N1 = 3
        N2 = 2
        N3 = 1
        REM = 4
        SLEEP_STAGES = {N1, N2, N3, REM}
        VALID_STAGES = {N3, N2, N1, REM, WAKE}

        SHORT_BOUT_THRESHOLD_EPOCHS = 3
        HIGH_ENTROPY_THRESHOLD = 1.2
        LOW_CONFIDENCE_THRESHOLD = 0.6

        def _as_float_1d(x):
            return np.asarray(x, dtype=float).reshape(-1)

        def _safe_div(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        def _stage_epochs(arr, stage_code):
            return int(np.count_nonzero(arr == stage_code))

        def _build_bouts(stage_arr):
            """Return starts, ends, stage values and lengths in epochs."""
            n = len(stage_arr)
            if n == 0:
                return (
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                )
            starts = np.r_[0, 1 + np.where(np.diff(stage_arr) != 0)[0]].astype(int)
            ends = np.r_[starts[1:], n].astype(int)
            bout_stages = stage_arr[starts].astype(int)
            bout_lengths = (ends - starts).astype(int)
            return starts, ends, bout_stages, bout_lengths

        # ------------------------------------------------------------------
        # Stage sequence and primary architecture features.
        # CAISR coding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable.
        # ------------------------------------------------------------------
        raw_stages = _as_float_1d(algo_data.get('stage_caisr', np.array([])))
        stage_mask = np.isin(raw_stages, list(VALID_STAGES))
        stages = raw_stages[stage_mask].astype(int)
        n_epochs = len(stages)

        trt_sec = n_epochs * EPOCH_SEC

        n_w = _stage_epochs(stages, WAKE)
        n_n1 = _stage_epochs(stages, N1)
        n_n2 = _stage_epochs(stages, N2)
        n_n3 = _stage_epochs(stages, N3)
        n_rem = _stage_epochs(stages, REM)

        n_tst = n_n1 + n_n2 + n_n3 + n_rem
        tst_sec = n_tst * EPOCH_SEC
        se = _safe_div(tst_sec, trt_sec)

        sleep_epoch_idx = np.where(np.isin(stages, list(SLEEP_STAGES)))[0]
        first_sleep_idx = int(sleep_epoch_idx[0]) if len(sleep_epoch_idx) > 0 else None
        sol_sec = first_sleep_idx * EPOCH_SEC if first_sleep_idx is not None else trt_sec

        rem_epoch_idx = np.where(stages == REM)[0]
        if first_sleep_idx is not None:
            rem_after_onset = rem_epoch_idx[rem_epoch_idx >= first_sleep_idx]
            if len(rem_after_onset) > 0:
                first_rem_idx = int(rem_after_onset[0])
                rem_latency_sec = (first_rem_idx - first_sleep_idx) * EPOCH_SEC
            else:
                first_rem_idx = None
                rem_latency_sec = (n_epochs - first_sleep_idx) * EPOCH_SEC
        else:
            first_rem_idx = None
            rem_latency_sec = 0.0

        wake_time_sec = n_w * EPOCH_SEC
        if first_sleep_idx is not None:
            waso_epochs = int(np.count_nonzero(stages[first_sleep_idx:] == WAKE))
        else:
            waso_epochs = 0
        waso_sec = waso_epochs * EPOCH_SEC

        n1_pct = _safe_div(n_n1, n_tst)
        n2_pct = _safe_div(n_n2, n_tst)
        n3_pct = _safe_div(n_n3, n_tst)
        rem_pct = _safe_div(n_rem, n_tst)
        wake_pct = _safe_div(n_w, n_epochs)

        dur_w_sec = n_w * EPOCH_SEC
        dur_n1_sec = n_n1 * EPOCH_SEC
        dur_n2_sec = n_n2 * EPOCH_SEC
        dur_n3_sec = n_n3 * EPOCH_SEC
        dur_rem_sec = n_rem * EPOCH_SEC
        nrem_sec = dur_n1_sec + dur_n2_sec + dur_n3_sec

        nrem_pct = _safe_div(n_n1 + n_n2 + n_n3, n_tst)
        n3_n1_ratio = _safe_div(n_n3, n_n1)
        n3_n1n2_ratio = _safe_div(n_n3, n_n1 + n_n2)
        rem_nrem_ratio = _safe_div(dur_rem_sec, nrem_sec)

        # ------------------------------------------------------------------
        # Fragmentation and bout statistics.
        # ------------------------------------------------------------------
        if n_epochs > 1:
            stage_transition_count = int(np.count_nonzero(np.diff(stages) != 0))
        else:
            stage_transition_count = 0
        transition_rate = _safe_div(stage_transition_count, tst_sec / 3600.0)

        starts, ends, bout_stages, bout_lengths_epochs = _build_bouts(stages)

        wake_intrusions = 0
        if first_sleep_idx is not None and len(starts) > 0:
            for i in range(len(starts)):
                if bout_stages[i] != WAKE:
                    continue
                if starts[i] < first_sleep_idx:
                    continue
                if i > 0 and bout_stages[i - 1] in SLEEP_STAGES:
                    wake_intrusions += 1

        sleep_bout_lengths_epochs = bout_lengths_epochs[np.isin(bout_stages, list(SLEEP_STAGES))]
        short_bout_ratio = (
            float(np.mean(sleep_bout_lengths_epochs < SHORT_BOUT_THRESHOLD_EPOCHS))
            if len(sleep_bout_lengths_epochs) > 0 else 0.0
        )

        n1_bout_count = int(np.count_nonzero(bout_stages == N1))
        n2_bout_count = int(np.count_nonzero(bout_stages == N2))
        n3_bout_count = int(np.count_nonzero(bout_stages == N3))
        rem_bout_count = int(np.count_nonzero(bout_stages == REM))

        stage_order = [WAKE, N1, N2, N3, REM]  # W, N1, N2, N3, REM
        bout_lengths_sec_by_stage = {}
        for s in stage_order:
            bout_lengths_sec_by_stage[s] = (
                bout_lengths_epochs[bout_stages == s].astype(float) * EPOCH_SEC
                if len(bout_stages) > 0 else np.array([], dtype=float)
            )

        mean_bout = []
        median_bout = []
        max_bout = []
        p25_bout = []
        p75_bout = []
        p90_bout = []
        p95_bout = []
        for s in stage_order:
            lengths = bout_lengths_sec_by_stage[s]
            if len(lengths) == 0:
                mean_bout.append(0.0)
                median_bout.append(0.0)
                max_bout.append(0.0)
                p25_bout.append(0.0)
                p75_bout.append(0.0)
                p90_bout.append(0.0)
                p95_bout.append(0.0)
                continue
            mean_bout.append(float(np.mean(lengths)))
            median_bout.append(float(np.median(lengths)))
            max_bout.append(float(np.max(lengths)))
            p25_bout.append(float(np.quantile(lengths, 0.25)))
            p75_bout.append(float(np.quantile(lengths, 0.75)))
            p90_bout.append(float(np.quantile(lengths, 0.90)))
            p95_bout.append(float(np.quantile(lengths, 0.95)))

        # ------------------------------------------------------------------
        # Early/late night dynamics.
        # ------------------------------------------------------------------
        split_idx = n_epochs // 2
        early = stages[:split_idx]
        late = stages[split_idx:]

        early_tst = int(np.count_nonzero(np.isin(early, list(SLEEP_STAGES))))
        late_tst = int(np.count_nonzero(np.isin(late, list(SLEEP_STAGES))))

        early_n3_pct = _safe_div(np.count_nonzero(early == N3), early_tst)
        late_rem_pct = _safe_div(np.count_nonzero(late == REM), late_tst)

        late_n3_pct = _safe_div(np.count_nonzero(late == N3), late_tst)
        early_rem_pct = _safe_div(np.count_nonzero(early == REM), early_tst)
        early_w_half_pct = _safe_div(np.count_nonzero(early == WAKE), len(early))
        late_w_half_pct = _safe_div(np.count_nonzero(late == WAKE), len(late))

        delta_n3 = late_n3_pct - early_n3_pct
        delta_rem = late_rem_pct - early_rem_pct
        delta_w = late_w_half_pct - early_w_half_pct

        # ------------------------------------------------------------------
        # Sleep periodicity features (coarse NREM->REM cycles).
        # ------------------------------------------------------------------
        sleep_cycle_count = 0
        mean_cycle_duration_sec = 0.0
        first_cycle_nrem_duration_sec = 0.0

        if first_sleep_idx is not None:
            post_sleep_stages = stages[first_sleep_idx:]
            p_starts, p_ends, p_stage, _ = _build_bouts(post_sleep_stages)

            cycle_durations_sec = []
            in_nrem = False
            cycle_start = None
            for bs, b_start, b_end in zip(p_stage, p_starts, p_ends):
                if bs in (N1, N2, N3):
                    if not in_nrem:
                        in_nrem = True
                        cycle_start = b_start
                elif bs == REM and in_nrem and cycle_start is not None:
                    sleep_cycle_count += 1
                    cycle_durations_sec.append((b_end - cycle_start) * EPOCH_SEC)
                    in_nrem = False
                    cycle_start = None

            if len(cycle_durations_sec) > 0:
                mean_cycle_duration_sec = float(np.mean(cycle_durations_sec))

            if first_rem_idx is not None and first_rem_idx > first_sleep_idx:
                first_cycle_nrem_duration_sec = (
                    np.count_nonzero(np.isin(stages[first_sleep_idx:first_rem_idx], [N1, N2, N3]))
                    * EPOCH_SEC
                )
            else:
                first_cycle_nrem_duration_sec = (
                    np.count_nonzero(np.isin(stages[first_sleep_idx:], [N1, N2, N3]))
                    * EPOCH_SEC
                )

        # ------------------------------------------------------------------
        # Stability features.
        # ------------------------------------------------------------------
        sleep_binary = np.isin(stages, list(SLEEP_STAGES)).astype(int)
        if len(sleep_binary) > 0:
            diff_sleep = np.diff(sleep_binary, prepend=0, append=0)
            run_starts = np.where(diff_sleep == 1)[0]
            run_ends = np.where(diff_sleep == -1)[0]
            if len(run_starts) > 0 and len(run_ends) == len(run_starts):
                longest_cont_sleep_bout_sec = float(np.max((run_ends - run_starts) * EPOCH_SEC))
            else:
                longest_cont_sleep_bout_sec = 0.0
        else:
            longest_cont_sleep_bout_sec = 0.0

        longest_n3_bout_sec = float(np.max(bout_lengths_sec_by_stage[N3])) if len(bout_lengths_sec_by_stage[N3]) > 0 else 0.0
        longest_rem_bout_sec = float(np.max(bout_lengths_sec_by_stage[REM])) if len(bout_lengths_sec_by_stage[REM]) > 0 else 0.0

        # ------------------------------------------------------------------
        # Stage posterior uncertainty features from softmax probabilities.
        # ------------------------------------------------------------------
        def _first_available_prob(keys):
            for k in keys:
                arr = algo_data.get(k, None)
                if arr is not None and len(arr) > 0:
                    return _as_float_1d(arr)
            return np.array([], dtype=float)

        prob_n3 = _first_available_prob(['caisr_prob_n3'])
        prob_n2 = _first_available_prob(['caisr_prob_n2'])
        prob_n1 = _first_available_prob(['caisr_prob_n1'])
        prob_rem = _first_available_prob(['caisr_prob_r', 'caisr_prob_rem'])
        prob_w = _first_available_prob(['caisr_prob_w', 'caisr_prob_wake'])

        available_lengths = [len(x) for x in [prob_n3, prob_n2, prob_n1, prob_rem, prob_w] if len(x) > 0]
        mean_max_prob = 0.0
        std_max_prob = 0.0
        mean_stage_entropy = 0.0
        high_entropy_ratio = 0.0
        low_conf_ratio = 0.0
        posterior_volatility = 0.0

        if len(available_lengths) > 0:
            n_prob = int(min(available_lengths))
            prob_cols = []
            for arr in [prob_n3, prob_n2, prob_n1, prob_rem, prob_w]:
                if len(arr) == 0:
                    prob_cols.append(np.zeros(n_prob, dtype=float))
                else:
                    prob_cols.append(arr[:n_prob])

            post = np.column_stack(prob_cols)
            post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)
            post = np.clip(post, 0.0, 1.0)

            row_sums = post.sum(axis=1, keepdims=True)
            zero_rows = (row_sums[:, 0] <= 0)
            if np.any(zero_rows):
                post[zero_rows, :] = 1.0 / 5.0
                row_sums = post.sum(axis=1, keepdims=True)
            post = post / row_sums

            max_prob = np.max(post, axis=1)
            entropy = -np.sum(post * np.log(post + 1e-12), axis=1)

            mean_max_prob = float(np.mean(max_prob))
            std_max_prob = float(np.std(max_prob))
            mean_stage_entropy = float(np.mean(entropy))
            high_entropy_ratio = float(np.mean(entropy > HIGH_ENTROPY_THRESHOLD))
            low_conf_ratio = float(np.mean(max_prob < LOW_CONFIDENCE_THRESHOLD))

            if len(post) > 1:
                posterior_volatility = float(
                    np.mean(np.linalg.norm(np.diff(post, axis=0), ord=1, axis=1))
                )

        # ------------------------------------------------------------------
        # Composite indices.
        # ------------------------------------------------------------------
        sleep_fragmentation_index = transition_rate + (waso_sec / 3600.0) + short_bout_ratio

        n3_duration_hours = dur_n3_sec / 3600.0
        rem_duration_hours = dur_rem_sec / 3600.0
        n3_fragmentation = _safe_div(n3_bout_count, n3_duration_hours)
        rem_fragmentation = _safe_div(rem_bout_count, rem_duration_hours)

        longest_n3_hours = longest_n3_bout_sec / 3600.0
        mean_rem_bout_hours = mean_bout[4] / 3600.0  # stage order: [W, N1, N2, N3, REM]

        deep_sleep_preservation_index = n3_pct + longest_n3_hours - n3_fragmentation
        rem_integrity_index = rem_pct + mean_rem_bout_hours - rem_fragmentation

        features = [
            trt_sec, tst_sec, se, sol_sec, rem_latency_sec, wake_time_sec, waso_sec,
            n1_pct, n2_pct, n3_pct, rem_pct, wake_pct,
            dur_w_sec, dur_n1_sec, dur_n2_sec, dur_n3_sec, dur_rem_sec,
            nrem_pct, n3_n1_ratio, n3_n1n2_ratio, rem_nrem_ratio,
            float(stage_transition_count), transition_rate, float(wake_intrusions), short_bout_ratio,
            float(n1_bout_count), float(n2_bout_count), float(n3_bout_count), float(rem_bout_count),
        ]
        features.extend(mean_bout)
        features.extend(median_bout)
        features.extend(max_bout)
        features.extend(p25_bout)
        features.extend(p75_bout)
        features.extend(p90_bout)
        features.extend(p95_bout)
        features.extend([
            early_n3_pct, late_rem_pct, delta_n3, delta_rem, delta_w,
            float(sleep_cycle_count), mean_cycle_duration_sec, first_cycle_nrem_duration_sec,
            longest_cont_sleep_bout_sec, longest_n3_bout_sec, longest_rem_bout_sec,
            mean_max_prob, std_max_prob, mean_stage_entropy, high_entropy_ratio,
            low_conf_ratio, posterior_volatility,
            sleep_fragmentation_index, deep_sleep_preservation_index, rem_integrity_index
        ])

        if len(features) != 84:
            raise RuntimeError(
                f"Algorithmic feature dimension mismatch: expected {84}, got {len(features)}"
            )

        return np.asarray(features, dtype=np.float32)


    def extract_algorithmic_arousal_event_features(self, algo_data):
        """
        Extract arousal-burden and co-occurrence features from CAISR annotations.

        Inputs expected from the official algorithmic annotations:
            - stage_caisr: 30 s epochs, 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
            - arousal_caisr: 0.5 s labels, 0=No arousal, 1=Arousal
            - caisr_prob_arousal: 0.5 s arousal probability
            - resp_caisr: 1 s labels, 0=No event, 1=OA, 2=CA, 3=MA, 4=HY, 5=RERA
            - limb_caisr: 1 s labels, 0=No event, 1=Isolated, 2=Periodic

        Heuristic thresholds:
            - burst interval threshold: 30 s
            - transition-linked window: +/-15 s
            - respiratory-linked window: +/-10 s
            - limb-linked window: +/-10 s
            - high-probability arousal threshold: 0.5

        Output vector length: 30.
        """
        AROUSAL_EVENT_FEATURE_DIM = 30

        if not algo_data:
            return np.zeros(AROUSAL_EVENT_FEATURE_DIM, dtype=np.float32)

        EPOCH_SEC = 30.0
        WAKE = 5
        N1 = 3
        N2 = 2
        N3 = 1
        REM = 4
        VALID_STAGES = {N3, N2, N1, REM, WAKE}
        NREM_STAGES = {N1, N2, N3}
        SLEEP_STAGES = {N1, N2, N3, REM}

        BURST_INTERVAL_SEC = 30.0
        TRANSITION_WINDOW_SEC = 15.0
        RESP_LINK_WINDOW_SEC = 10.0
        LIMB_LINK_WINDOW_SEC = 10.0
        HIGH_PROB_THRESHOLD = 0.5

        def _as_float_1d(x):
            return np.asarray(x, dtype=float).reshape(-1)

        def _safe_div(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        def _segments_from_binary(mask, sample_sec):
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) == 0:
                empty = np.array([], dtype=float)
                return empty, empty, empty
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            starts_sec = starts.astype(float) * sample_sec
            ends_sec = ends.astype(float) * sample_sec
            durations_sec = (ends - starts).astype(float) * sample_sec
            return starts_sec, ends_sec, durations_sec

        def _stage_code_at_time(time_sec, stages):
            if len(stages) == 0:
                return None
            epoch_idx = int(np.clip(np.floor(time_sec / EPOCH_SEC), 0, len(stages) - 1))
            return int(stages[epoch_idx])

        def _interval_summary(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(values) == 0:
                return 0.0, 0.0, 0.0
            return float(np.mean(values)), float(np.std(values)), float(np.min(values))

        def _duration_summary(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(values) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            return (
                float(np.mean(values)),
                float(np.median(values)),
                float(np.max(values)),
                float(np.quantile(values, 0.75)),
                float(np.quantile(values, 0.90)),
                float(np.quantile(values, 0.95)),
            )

        def _event_overlap_ratio(base_starts_sec, base_ends_sec, ref_starts_sec, ref_ends_sec, window_sec):
            n_base = len(base_starts_sec)
            if n_base == 0 or len(ref_starts_sec) == 0:
                return 0.0

            linked = 0
            for s0, e0 in zip(base_starts_sec, base_ends_sec):
                hit = False
                for s1, e1 in zip(ref_starts_sec, ref_ends_sec):
                    if e0 >= (s1 - window_sec) and s0 <= (e1 + window_sec):
                        hit = True
                        break
                linked += int(hit)
            return _safe_div(linked, n_base)

        raw_stages = _as_float_1d(algo_data.get('stage_caisr', np.array([])))
        stage_mask = np.isin(raw_stages, list(VALID_STAGES))
        stages = raw_stages[stage_mask].astype(int)
        n_epochs = len(stages)
        trt_sec = n_epochs * EPOCH_SEC

        tst_sec = float(np.count_nonzero(np.isin(stages, list(SLEEP_STAGES))) * EPOCH_SEC)
        nrem_sec = float(np.count_nonzero(np.isin(stages, list(NREM_STAGES))) * EPOCH_SEC)
        rem_sec = float(np.count_nonzero(stages == REM) * EPOCH_SEC)
        n1_sec = float(np.count_nonzero(stages == N1) * EPOCH_SEC)
        n2_sec = float(np.count_nonzero(stages == N2) * EPOCH_SEC)
        n3_sec = float(np.count_nonzero(stages == N3) * EPOCH_SEC)

        arousal_labels = _as_float_1d(algo_data.get('arousal_caisr', np.array([])))
        arousal_prob = _as_float_1d(
            algo_data.get('caisr_prob_arousal', algo_data.get('caisr_prob_arous', np.array([])))
        )

        if len(arousal_labels) > 0:
            arousal_dt_sec = _safe_div(trt_sec, len(arousal_labels)) if trt_sec > 0 else 0.5
            arousal_binary = arousal_labels > 0
        elif len(arousal_prob) > 0:
            arousal_dt_sec = _safe_div(trt_sec, len(arousal_prob)) if trt_sec > 0 else 0.5
            arousal_binary = arousal_prob >= HIGH_PROB_THRESHOLD
        else:
            arousal_dt_sec = 0.5
            arousal_binary = np.array([], dtype=bool)

        arousal_starts_sec, arousal_ends_sec, arousal_durations_sec = _segments_from_binary(
            arousal_binary, arousal_dt_sec
        )
        arousal_count = int(len(arousal_starts_sec))
        arousal_duration_sec = float(np.sum(arousal_durations_sec))
        arousal_index = _safe_div(arousal_count, tst_sec / 3600.0)
        arousal_burden_ratio = _safe_div(arousal_duration_sec, tst_sec)

        mean_arousal_duration, median_arousal_duration, max_arousal_duration, p75_arousal_duration, p90_arousal_duration, p95_arousal_duration = _duration_summary(
            arousal_durations_sec
        )

        inter_arousal_intervals_sec = np.diff(arousal_starts_sec) if arousal_count > 1 else np.array([], dtype=float)
        mean_inter_arousal_interval, std_inter_arousal_interval, min_inter_arousal_interval = _interval_summary(
            inter_arousal_intervals_sec
        )
        burst_ratio = (
            float(np.mean(inter_arousal_intervals_sec < BURST_INTERVAL_SEC))
            if len(inter_arousal_intervals_sec) > 0 else 0.0
        )

        nrem_arousal_count = 0
        rem_arousal_count = 0
        n1_arousal_count = 0
        n2_arousal_count = 0
        n3_arousal_count = 0
        early_arousal_count = 0
        late_arousal_count = 0

        split_time_sec = trt_sec / 2.0
        for onset_sec in arousal_starts_sec:
            stage_code = _stage_code_at_time(onset_sec, stages)
            if stage_code in NREM_STAGES:
                nrem_arousal_count += 1
            if stage_code == REM:
                rem_arousal_count += 1
            if stage_code == N1:
                n1_arousal_count += 1
            if stage_code == N2:
                n2_arousal_count += 1
            if stage_code == N3:
                n3_arousal_count += 1

            if onset_sec < split_time_sec:
                early_arousal_count += 1
            else:
                late_arousal_count += 1

        early_stages = stages[: n_epochs // 2]
        late_stages = stages[n_epochs // 2 :]
        early_tst_sec = float(np.count_nonzero(np.isin(early_stages, list(SLEEP_STAGES))) * EPOCH_SEC)
        late_tst_sec = float(np.count_nonzero(np.isin(late_stages, list(SLEEP_STAGES))) * EPOCH_SEC)

        nrem_ari = _safe_div(nrem_arousal_count, nrem_sec / 3600.0)
        rem_ari = _safe_div(rem_arousal_count, rem_sec / 3600.0)
        n1_ari = _safe_div(n1_arousal_count, n1_sec / 3600.0)
        n2_ari = _safe_div(n2_arousal_count, n2_sec / 3600.0)
        n3_ari = _safe_div(n3_arousal_count, n3_sec / 3600.0)
        early_ari = _safe_div(early_arousal_count, early_tst_sec / 3600.0)
        late_ari = _safe_div(late_arousal_count, late_tst_sec / 3600.0)
        delta_ari = late_ari - early_ari

        if len(arousal_prob) > 0:
            arousal_prob = np.nan_to_num(arousal_prob, nan=0.0, posinf=0.0, neginf=0.0)
            arousal_prob = np.clip(arousal_prob, 0.0, 1.0)
            mean_arousal_prob = float(np.mean(arousal_prob))
            std_arousal_prob = float(np.std(arousal_prob))
            p90_arousal_prob = float(np.quantile(arousal_prob, 0.90))
            p95_arousal_prob = float(np.quantile(arousal_prob, 0.95))
            high_prob_arousal_ratio = float(np.mean(arousal_prob > HIGH_PROB_THRESHOLD))
        else:
            mean_arousal_prob = 0.0
            std_arousal_prob = 0.0
            p90_arousal_prob = 0.0
            p95_arousal_prob = 0.0
            high_prob_arousal_ratio = 0.0

        transition_times_sec = np.array([], dtype=float)
        if len(stages) > 1:
            transition_epoch_idx = np.where(np.diff(stages) != 0)[0] + 1
            transition_times_sec = transition_epoch_idx.astype(float) * EPOCH_SEC

        if arousal_count > 0 and len(transition_times_sec) > 0:
            transition_linked_count = 0
            for s0, e0 in zip(arousal_starts_sec, arousal_ends_sec):
                hit = np.any(
                    (transition_times_sec >= (s0 - TRANSITION_WINDOW_SEC)) &
                    (transition_times_sec <= (e0 + TRANSITION_WINDOW_SEC))
                )
                transition_linked_count += int(hit)
            transition_linked_arousal_ratio = _safe_div(transition_linked_count, arousal_count)
        else:
            transition_linked_arousal_ratio = 0.0

        resp_signal = _as_float_1d(algo_data.get('resp_caisr', np.array([])))
        resp_dt_sec = _safe_div(trt_sec, len(resp_signal)) if len(resp_signal) > 0 and trt_sec > 0 else 1.0
        resp_starts_sec, resp_ends_sec, _ = _segments_from_binary(resp_signal > 0, resp_dt_sec)
        resp_linked_arousal_ratio = _event_overlap_ratio(
            arousal_starts_sec,
            arousal_ends_sec,
            resp_starts_sec,
            resp_ends_sec,
            RESP_LINK_WINDOW_SEC,
        )

        limb_signal = _as_float_1d(algo_data.get('limb_caisr', np.array([])))
        limb_dt_sec = _safe_div(trt_sec, len(limb_signal)) if len(limb_signal) > 0 and trt_sec > 0 else 1.0
        limb_starts_sec, limb_ends_sec, _ = _segments_from_binary(limb_signal > 0, limb_dt_sec)
        limb_linked_arousal_ratio = _event_overlap_ratio(
            arousal_starts_sec,
            arousal_ends_sec,
            limb_starts_sec,
            limb_ends_sec,
            LIMB_LINK_WINDOW_SEC,
        )

        features = [
            float(arousal_count),
            arousal_duration_sec,
            arousal_index,
            arousal_burden_ratio,
            mean_arousal_duration,
            median_arousal_duration,
            max_arousal_duration,
            p75_arousal_duration,
            p90_arousal_duration,
            p95_arousal_duration,
            mean_inter_arousal_interval,
            std_inter_arousal_interval,
            min_inter_arousal_interval,
            burst_ratio,
            nrem_ari,
            rem_ari,
            n1_ari,
            n2_ari,
            n3_ari,
            early_ari,
            late_ari,
            delta_ari,
            mean_arousal_prob,
            std_arousal_prob,
            p90_arousal_prob,
            p95_arousal_prob,
            high_prob_arousal_ratio,
            transition_linked_arousal_ratio,
            resp_linked_arousal_ratio,
            limb_linked_arousal_ratio,
        ]

        return np.asarray(features, dtype=np.float32)


    def extract_algorithmic_respiratory_event_features(self, algo_data):
        """
        Extract respiratory-burden and respiratory-arousal coupling features from
        CAISR annotations.

        Inputs expected from the official algorithmic annotations:
            - stage_caisr: 30 s epochs, 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
            - resp_caisr: 1 s labels, 0=No event, 1=OA, 2=CA, 3=MA, 4=HY, 5=RERA
            - arousal_caisr: 0.5 s labels, 0=No arousal, 1=Arousal

        Heuristic thresholds:
            - burst interval threshold: 30 s
            - post-event arousal window: 30 s
            - transition-linked window: +/-15 s
            - HY weight in obstructive dominance: 0.5

        Output vector length: 42.
        """
        RESP_EVENT_FEATURE_DIM = 42

        if not algo_data:
            return np.zeros(RESP_EVENT_FEATURE_DIM, dtype=np.float32)

        EPOCH_SEC = 30.0
        WAKE = 5
        N1 = 3
        N2 = 2
        N3 = 1
        REM = 4
        VALID_STAGES = {N3, N2, N1, REM, WAKE}
        NREM_STAGES = {N1, N2, N3}
        SLEEP_STAGES = {N1, N2, N3, REM}

        RESP_OA = 1
        RESP_CA = 2
        RESP_MA = 3
        RESP_HY = 4
        RESP_RERA = 5
        RESP_EVENT_CLASSES = [RESP_OA, RESP_CA, RESP_MA, RESP_HY, RESP_RERA]

        BURST_INTERVAL_SEC = 30.0
        POST_EVENT_AROUSAL_WINDOW_SEC = 30.0
        TRANSITION_WINDOW_SEC = 15.0
        OBSTRUCTIVE_HY_WEIGHT = 0.5

        def _as_float_1d(x):
            return np.asarray(x, dtype=float).reshape(-1)

        def _safe_div(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        def _stage_code_at_time(time_sec, stages):
            if len(stages) == 0:
                return None
            epoch_idx = int(np.clip(np.floor(time_sec / EPOCH_SEC), 0, len(stages) - 1))
            return int(stages[epoch_idx])

        def _interval_summary(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(values) == 0:
                return 0.0, 0.0, 0.0
            return float(np.mean(values)), float(np.std(values)), float(np.min(values))

        def _segment_multiclass_events(signal, sample_sec):
            signal = np.asarray(signal, dtype=int).reshape(-1)
            events = []
            n = len(signal)
            if n == 0:
                return events

            start = None
            current_label = 0
            for i, value in enumerate(signal):
                if value > 0:
                    if start is None:
                        start = i
                        current_label = int(value)
                    elif int(value) != current_label:
                        events.append({
                            "label": current_label,
                            "start_sec": start * sample_sec,
                            "end_sec": i * sample_sec,
                            "duration_sec": (i - start) * sample_sec,
                        })
                        start = i
                        current_label = int(value)
                elif start is not None:
                    events.append({
                        "label": current_label,
                        "start_sec": start * sample_sec,
                        "end_sec": i * sample_sec,
                        "duration_sec": (i - start) * sample_sec,
                    })
                    start = None
                    current_label = 0

            if start is not None:
                events.append({
                    "label": current_label,
                    "start_sec": start * sample_sec,
                    "end_sec": n * sample_sec,
                    "duration_sec": (n - start) * sample_sec,
                })

            return events

        def _segments_from_binary(mask, sample_sec):
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) == 0:
                empty = np.array([], dtype=float)
                return empty, empty, empty
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            starts_sec = starts.astype(float) * sample_sec
            ends_sec = ends.astype(float) * sample_sec
            durations_sec = (ends - starts).astype(float) * sample_sec
            return starts_sec, ends_sec, durations_sec

        raw_stages = _as_float_1d(algo_data.get('stage_caisr', np.array([])))
        stage_mask = np.isin(raw_stages, list(VALID_STAGES))
        stages = raw_stages[stage_mask].astype(int)
        n_epochs = len(stages)
        trt_sec = n_epochs * EPOCH_SEC

        tst_sec = float(np.count_nonzero(np.isin(stages, list(SLEEP_STAGES))) * EPOCH_SEC)
        nrem_sec = float(np.count_nonzero(np.isin(stages, list(NREM_STAGES))) * EPOCH_SEC)
        rem_sec = float(np.count_nonzero(stages == REM) * EPOCH_SEC)
        n1_sec = float(np.count_nonzero(stages == N1) * EPOCH_SEC)
        n2_sec = float(np.count_nonzero(stages == N2) * EPOCH_SEC)
        n3_sec = float(np.count_nonzero(stages == N3) * EPOCH_SEC)

        resp_signal = _as_float_1d(algo_data.get('resp_caisr', np.array([])))
        resp_dt_sec = _safe_div(trt_sec, len(resp_signal)) if len(resp_signal) > 0 and trt_sec > 0 else 1.0
        resp_events = _segment_multiclass_events(resp_signal.astype(int), resp_dt_sec)

        total_resp_count = int(len(resp_events))
        total_resp_duration_sec = float(sum(evt["duration_sec"] for evt in resp_events))
        respiratory_event_index = _safe_div(total_resp_count, tst_sec / 3600.0)
        respiratory_burden_ratio = _safe_div(total_resp_duration_sec, tst_sec)

        counts_by_class = {label: 0 for label in RESP_EVENT_CLASSES}
        durations_by_class = {label: [] for label in RESP_EVENT_CLASSES}
        for evt in resp_events:
            label = evt["label"]
            if label in counts_by_class:
                counts_by_class[label] += 1
                durations_by_class[label].append(float(evt["duration_sec"]))

        oa_count = counts_by_class[RESP_OA]
        ca_count = counts_by_class[RESP_CA]
        ma_count = counts_by_class[RESP_MA]
        hy_count = counts_by_class[RESP_HY]
        rera_count = counts_by_class[RESP_RERA]

        oai = _safe_div(oa_count, tst_sec / 3600.0)
        cai = _safe_div(ca_count, tst_sec / 3600.0)
        mai = _safe_div(ma_count, tst_sec / 3600.0)
        hyi = _safe_div(hy_count, tst_sec / 3600.0)
        rerai = _safe_div(rera_count, tst_sec / 3600.0)

        oa_ratio = _safe_div(oa_count, total_resp_count)
        ca_ratio = _safe_div(ca_count, total_resp_count)
        hy_ratio = _safe_div(hy_count, total_resp_count)

        all_resp_durations = np.asarray([evt["duration_sec"] for evt in resp_events], dtype=float)
        if len(all_resp_durations) > 0:
            mean_resp_duration = float(np.mean(all_resp_durations))
            max_resp_duration = float(np.max(all_resp_durations))
            p75_resp_duration = float(np.quantile(all_resp_durations, 0.75))
            p90_resp_duration = float(np.quantile(all_resp_durations, 0.90))
            p95_resp_duration = float(np.quantile(all_resp_durations, 0.95))
        else:
            mean_resp_duration = 0.0
            max_resp_duration = 0.0
            p75_resp_duration = 0.0
            p90_resp_duration = 0.0
            p95_resp_duration = 0.0

        mean_oa_duration = float(np.mean(durations_by_class[RESP_OA])) if len(durations_by_class[RESP_OA]) > 0 else 0.0
        mean_ca_duration = float(np.mean(durations_by_class[RESP_CA])) if len(durations_by_class[RESP_CA]) > 0 else 0.0
        mean_hy_duration = float(np.mean(durations_by_class[RESP_HY])) if len(durations_by_class[RESP_HY]) > 0 else 0.0

        if total_resp_count > 1:
            inter_event_gaps_sec = np.asarray(
                [resp_events[i]["start_sec"] - resp_events[i - 1]["end_sec"] for i in range(1, total_resp_count)],
                dtype=float,
            )
        else:
            inter_event_gaps_sec = np.array([], dtype=float)

        respiratory_burst_ratio = (
            float(np.mean(inter_event_gaps_sec < BURST_INTERVAL_SEC))
            if len(inter_event_gaps_sec) > 0 else 0.0
        )

        longest_event_cluster_sec = 0.0
        if total_resp_count > 0:
            cluster_start = resp_events[0]["start_sec"]
            cluster_end = resp_events[0]["end_sec"]
            longest_event_cluster_sec = cluster_end - cluster_start
            for evt in resp_events[1:]:
                if (evt["start_sec"] - cluster_end) < BURST_INTERVAL_SEC:
                    cluster_end = evt["end_sec"]
                else:
                    longest_event_cluster_sec = max(longest_event_cluster_sec, cluster_end - cluster_start)
                    cluster_start = evt["start_sec"]
                    cluster_end = evt["end_sec"]
            longest_event_cluster_sec = max(longest_event_cluster_sec, cluster_end - cluster_start)

        nrem_resp_count = 0
        rem_resp_count = 0
        n1_resp_count = 0
        n2_resp_count = 0
        n3_resp_count = 0
        early_resp_count = 0
        late_resp_count = 0

        split_time_sec = trt_sec / 2.0
        for evt in resp_events:
            onset_sec = evt["start_sec"]
            stage_code = _stage_code_at_time(onset_sec, stages)
            if stage_code in NREM_STAGES:
                nrem_resp_count += 1
            if stage_code == REM:
                rem_resp_count += 1
            if stage_code == N1:
                n1_resp_count += 1
            if stage_code == N2:
                n2_resp_count += 1
            if stage_code == N3:
                n3_resp_count += 1

            if onset_sec < split_time_sec:
                early_resp_count += 1
            else:
                late_resp_count += 1

        early_stages = stages[: n_epochs // 2]
        late_stages = stages[n_epochs // 2 :]
        early_tst_sec = float(np.count_nonzero(np.isin(early_stages, list(SLEEP_STAGES))) * EPOCH_SEC)
        late_tst_sec = float(np.count_nonzero(np.isin(late_stages, list(SLEEP_STAGES))) * EPOCH_SEC)

        nrem_resp_index = _safe_div(nrem_resp_count, nrem_sec / 3600.0)
        rem_resp_index = _safe_div(rem_resp_count, rem_sec / 3600.0)
        rem_to_nrem_ratio = _safe_div(rem_resp_index, nrem_resp_index)
        n1_resp_index = _safe_div(n1_resp_count, n1_sec / 3600.0)
        n2_resp_index = _safe_div(n2_resp_count, n2_sec / 3600.0)
        n3_resp_index = _safe_div(n3_resp_count, n3_sec / 3600.0)
        early_resp_index = _safe_div(early_resp_count, early_tst_sec / 3600.0)
        late_resp_index = _safe_div(late_resp_count, late_tst_sec / 3600.0)
        delta_resp_index = late_resp_index - early_resp_index

        arousal_signal = _as_float_1d(algo_data.get('arousal_caisr', np.array([])))
        if len(arousal_signal) > 0:
            arousal_dt_sec = _safe_div(trt_sec, len(arousal_signal)) if trt_sec > 0 else 0.5
            arousal_starts_sec, arousal_ends_sec, _ = _segments_from_binary(arousal_signal > 0, arousal_dt_sec)
        else:
            arousal_starts_sec = np.array([], dtype=float)
            arousal_ends_sec = np.array([], dtype=float)

        post_event_arousal_hits = 0
        resp_to_arousal_delays_sec = []
        if total_resp_count > 0 and len(arousal_starts_sec) > 0:
            for evt in resp_events:
                evt_end = evt["end_sec"]
                candidate_mask = (
                    (arousal_starts_sec >= evt_end) &
                    (arousal_starts_sec <= (evt_end + POST_EVENT_AROUSAL_WINDOW_SEC))
                )
                if np.any(candidate_mask):
                    first_arousal_sec = float(arousal_starts_sec[candidate_mask][0])
                    post_event_arousal_hits += 1
                    resp_to_arousal_delays_sec.append(first_arousal_sec - evt_end)
        post_event_arousal_ratio = _safe_div(post_event_arousal_hits, total_resp_count)
        resp_to_arousal_delay = float(np.mean(resp_to_arousal_delays_sec)) if len(resp_to_arousal_delays_sec) > 0 else 0.0

        transition_times_sec = np.array([], dtype=float)
        if len(stages) > 1:
            transition_epoch_idx = np.where(np.diff(stages) != 0)[0] + 1
            transition_times_sec = transition_epoch_idx.astype(float) * EPOCH_SEC

        if total_resp_count > 0 and len(transition_times_sec) > 0:
            resp_linked_transition_hits = 0
            for evt in resp_events:
                s0 = evt["start_sec"]
                e0 = evt["end_sec"]
                hit = np.any(
                    (transition_times_sec >= (s0 - TRANSITION_WINDOW_SEC)) &
                    (transition_times_sec <= (e0 + TRANSITION_WINDOW_SEC))
                )
                resp_linked_transition_hits += int(hit)
            resp_linked_transition_ratio = _safe_div(resp_linked_transition_hits, total_resp_count)
        else:
            resp_linked_transition_ratio = 0.0

        apnea_hypopnea_count = oa_count + ca_count + ma_count + hy_count
        apnea_hypopnea_burden = _safe_div(apnea_hypopnea_count, tst_sec / 3600.0)
        obstructive_dominance = _safe_div(oa_count + OBSTRUCTIVE_HY_WEIGHT * hy_count, total_resp_count)
        central_dominance = _safe_div(ca_count, total_resp_count)

        features = [
            float(total_resp_count),
            total_resp_duration_sec,
            respiratory_event_index,
            respiratory_burden_ratio,
            float(oa_count),
            float(ca_count),
            float(ma_count),
            float(hy_count),
            float(rera_count),
            oai,
            cai,
            mai,
            hyi,
            rerai,
            oa_ratio,
            ca_ratio,
            hy_ratio,
            mean_resp_duration,
            max_resp_duration,
            mean_oa_duration,
            mean_ca_duration,
            mean_hy_duration,
            p75_resp_duration,
            p90_resp_duration,
            p95_resp_duration,
            respiratory_burst_ratio,
            longest_event_cluster_sec,
            nrem_resp_index,
            rem_resp_index,
            rem_to_nrem_ratio,
            n1_resp_index,
            n2_resp_index,
            n3_resp_index,
            early_resp_index,
            late_resp_index,
            delta_resp_index,
            post_event_arousal_ratio,
            resp_to_arousal_delay,
            resp_linked_transition_ratio,
            apnea_hypopnea_burden,
            obstructive_dominance,
            central_dominance,
        ]

        return np.asarray(features, dtype=np.float32)


    def extract_algorithmic_limb_event_features(self, algo_data):
        """
        Extract limb-movement burden and coupling features from CAISR annotations.

        Inputs expected from the official algorithmic annotations:
            - stage_caisr: 30 s epochs, 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
            - limb_caisr: 1 s labels, 0=No event, 1=Isolated, 2=Periodic
            - arousal_caisr: 0.5 s labels, 0=No arousal, 1=Arousal
            - resp_caisr: 1 s labels, 0=No event, 1=OA, 2=CA, 3=MA, 4=HY, 5=RERA

        Heuristic thresholds:
            - burst interval threshold: 30 s
            - limb-linked arousal window: +/-10 s
            - arousal-followed-by-limb window: 30 s
            - respiratory-linked limb window: +/-10 s
            - transition-linked window: +/-15 s

        Output vector length: 30.
        """
        LIMB_EVENT_FEATURE_DIM = 30

        if not algo_data:
            return np.zeros(LIMB_EVENT_FEATURE_DIM, dtype=np.float32)

        EPOCH_SEC = 30.0
        WAKE = 5
        N2 = 2
        N3 = 1
        REM = 4
        VALID_STAGES = {1, 2, 3, 4, 5}
        NREM_STAGES = {1, 2, 3}
        SLEEP_STAGES = {1, 2, 3, 4}

        LIMB_ISOLATED = 1
        LIMB_PLM = 2
        LIMB_EVENT_CLASSES = [LIMB_ISOLATED, LIMB_PLM]

        BURST_INTERVAL_SEC = 30.0
        LIMB_LINKED_AROUSAL_WINDOW_SEC = 10.0
        AROUSAL_FOLLOWED_BY_LIMB_WINDOW_SEC = 30.0
        RESP_LINKED_LIMB_WINDOW_SEC = 10.0
        TRANSITION_WINDOW_SEC = 15.0

        def _as_float_1d(x):
            return np.asarray(x, dtype=float).reshape(-1)

        def _safe_div(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        def _stage_code_at_time(time_sec, stages):
            if len(stages) == 0:
                return None
            epoch_idx = int(np.clip(np.floor(time_sec / EPOCH_SEC), 0, len(stages) - 1))
            return int(stages[epoch_idx])

        def _segment_multiclass_events(signal, sample_sec):
            signal = np.asarray(signal, dtype=int).reshape(-1)
            events = []
            n = len(signal)
            if n == 0:
                return events

            start = None
            current_label = 0
            for i, value in enumerate(signal):
                if value > 0:
                    if start is None:
                        start = i
                        current_label = int(value)
                    elif int(value) != current_label:
                        events.append({
                            "label": current_label,
                            "start_sec": start * sample_sec,
                            "end_sec": i * sample_sec,
                            "duration_sec": (i - start) * sample_sec,
                        })
                        start = i
                        current_label = int(value)
                elif start is not None:
                    events.append({
                        "label": current_label,
                        "start_sec": start * sample_sec,
                        "end_sec": i * sample_sec,
                        "duration_sec": (i - start) * sample_sec,
                    })
                    start = None
                    current_label = 0

            if start is not None:
                events.append({
                    "label": current_label,
                    "start_sec": start * sample_sec,
                    "end_sec": n * sample_sec,
                    "duration_sec": (n - start) * sample_sec,
                })

            return events

        def _segments_from_binary(mask, sample_sec):
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) == 0:
                empty = np.array([], dtype=float)
                return empty, empty, empty
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            starts_sec = starts.astype(float) * sample_sec
            ends_sec = ends.astype(float) * sample_sec
            durations_sec = (ends - starts).astype(float) * sample_sec
            return starts_sec, ends_sec, durations_sec

        def _event_overlap_ratio(base_starts_sec, base_ends_sec, ref_starts_sec, ref_ends_sec, window_sec):
            n_base = len(base_starts_sec)
            if n_base == 0 or len(ref_starts_sec) == 0:
                return 0.0

            linked = 0
            for s0, e0 in zip(base_starts_sec, base_ends_sec):
                hit = False
                for s1, e1 in zip(ref_starts_sec, ref_ends_sec):
                    if e0 >= (s1 - window_sec) and s0 <= (e1 + window_sec):
                        hit = True
                        break
                linked += int(hit)
            return _safe_div(linked, n_base)

        raw_stages = _as_float_1d(algo_data.get('stage_caisr', np.array([])))
        stage_mask = np.isin(raw_stages, list(VALID_STAGES))
        stages = raw_stages[stage_mask].astype(int)
        n_epochs = len(stages)
        trt_sec = n_epochs * EPOCH_SEC

        tst_sec = float(np.count_nonzero(np.isin(stages, list(SLEEP_STAGES))) * EPOCH_SEC)
        nrem_sec = float(np.count_nonzero(np.isin(stages, list(NREM_STAGES))) * EPOCH_SEC)
        rem_sec = float(np.count_nonzero(stages == REM) * EPOCH_SEC)
        n2_sec = float(np.count_nonzero(stages == N2) * EPOCH_SEC)
        n3_sec = float(np.count_nonzero(stages == N3) * EPOCH_SEC)

        limb_signal = _as_float_1d(algo_data.get('limb_caisr', np.array([])))
        limb_dt_sec = _safe_div(trt_sec, len(limb_signal)) if len(limb_signal) > 0 and trt_sec > 0 else 1.0
        limb_events = _segment_multiclass_events(limb_signal.astype(int), limb_dt_sec)

        total_limb_count = int(len(limb_events))
        total_limb_duration_sec = float(sum(evt["duration_sec"] for evt in limb_events))
        limb_movement_index = _safe_div(total_limb_count, tst_sec / 3600.0)
        limb_burden_ratio = _safe_div(total_limb_duration_sec, tst_sec)

        counts_by_class = {label: 0 for label in LIMB_EVENT_CLASSES}
        durations_by_class = {label: [] for label in LIMB_EVENT_CLASSES}
        for evt in limb_events:
            label = evt["label"]
            if label in counts_by_class:
                counts_by_class[label] += 1
                durations_by_class[label].append(float(evt["duration_sec"]))

        isolated_limb_count = counts_by_class[LIMB_ISOLATED]
        plm_count = counts_by_class[LIMB_PLM]
        isolated_limb_index = _safe_div(isolated_limb_count, tst_sec / 3600.0)
        plmi = _safe_div(plm_count, tst_sec / 3600.0)
        plm_ratio = _safe_div(plm_count, total_limb_count)

        all_limb_durations = np.asarray([evt["duration_sec"] for evt in limb_events], dtype=float)
        if len(all_limb_durations) > 0:
            mean_limb_duration = float(np.mean(all_limb_durations))
            max_limb_duration = float(np.max(all_limb_durations))
            p75_limb_duration = float(np.quantile(all_limb_durations, 0.75))
            p90_limb_duration = float(np.quantile(all_limb_durations, 0.90))
            p95_limb_duration = float(np.quantile(all_limb_durations, 0.95))
        else:
            mean_limb_duration = 0.0
            max_limb_duration = 0.0
            p75_limb_duration = 0.0
            p90_limb_duration = 0.0
            p95_limb_duration = 0.0

        inter_limb_intervals_sec = (
            np.diff(np.asarray([evt["start_sec"] for evt in limb_events], dtype=float))
            if total_limb_count > 1 else np.array([], dtype=float)
        )
        mean_inter_limb_interval = float(np.mean(inter_limb_intervals_sec)) if len(inter_limb_intervals_sec) > 0 else 0.0
        std_inter_limb_interval = float(np.std(inter_limb_intervals_sec)) if len(inter_limb_intervals_sec) > 0 else 0.0
        limb_burst_ratio = (
            float(np.mean(inter_limb_intervals_sec < BURST_INTERVAL_SEC))
            if len(inter_limb_intervals_sec) > 0 else 0.0
        )

        nrem_limb_count = 0
        rem_limb_count = 0
        n2_limb_count = 0
        n3_limb_count = 0
        early_limb_count = 0
        late_limb_count = 0

        split_time_sec = trt_sec / 2.0
        for evt in limb_events:
            onset_sec = evt["start_sec"]
            stage_code = _stage_code_at_time(onset_sec, stages)
            if stage_code in NREM_STAGES:
                nrem_limb_count += 1
            if stage_code == REM:
                rem_limb_count += 1
            if stage_code == N2:
                n2_limb_count += 1
            if stage_code == N3:
                n3_limb_count += 1

            if onset_sec < split_time_sec:
                early_limb_count += 1
            else:
                late_limb_count += 1

        early_stages = stages[: n_epochs // 2]
        late_stages = stages[n_epochs // 2 :]
        early_tst_sec = float(np.count_nonzero(np.isin(early_stages, list(SLEEP_STAGES))) * EPOCH_SEC)
        late_tst_sec = float(np.count_nonzero(np.isin(late_stages, list(SLEEP_STAGES))) * EPOCH_SEC)

        nrem_limb_index = _safe_div(nrem_limb_count, nrem_sec / 3600.0)
        rem_limb_index = _safe_div(rem_limb_count, rem_sec / 3600.0)
        n2_limb_index = _safe_div(n2_limb_count, n2_sec / 3600.0)
        n3_limb_index = _safe_div(n3_limb_count, n3_sec / 3600.0)
        early_limb_index = _safe_div(early_limb_count, early_tst_sec / 3600.0)
        late_limb_index = _safe_div(late_limb_count, late_tst_sec / 3600.0)
        delta_limb_index = late_limb_index - early_limb_index

        arousal_signal = _as_float_1d(algo_data.get('arousal_caisr', np.array([])))
        if len(arousal_signal) > 0:
            arousal_dt_sec = _safe_div(trt_sec, len(arousal_signal)) if trt_sec > 0 else 0.5
            arousal_starts_sec, arousal_ends_sec, _ = _segments_from_binary(arousal_signal > 0, arousal_dt_sec)
        else:
            arousal_starts_sec = np.array([], dtype=float)
            arousal_ends_sec = np.array([], dtype=float)

        limb_starts_sec = np.asarray([evt["start_sec"] for evt in limb_events], dtype=float)
        limb_ends_sec = np.asarray([evt["end_sec"] for evt in limb_events], dtype=float)

        limb_linked_arousal_ratio = _event_overlap_ratio(
            limb_starts_sec,
            limb_ends_sec,
            arousal_starts_sec,
            arousal_ends_sec,
            LIMB_LINKED_AROUSAL_WINDOW_SEC,
        )

        arousal_followed_by_limb_hits = 0
        if len(arousal_starts_sec) > 0 and total_limb_count > 0:
            for arousal_end_sec in arousal_ends_sec:
                hit = np.any(
                    (limb_starts_sec >= arousal_end_sec) &
                    (limb_starts_sec <= (arousal_end_sec + AROUSAL_FOLLOWED_BY_LIMB_WINDOW_SEC))
                )
                arousal_followed_by_limb_hits += int(hit)
        arousal_followed_by_limb_ratio = _safe_div(arousal_followed_by_limb_hits, len(arousal_starts_sec))

        resp_signal = _as_float_1d(algo_data.get('resp_caisr', np.array([])))
        resp_dt_sec = _safe_div(trt_sec, len(resp_signal)) if len(resp_signal) > 0 and trt_sec > 0 else 1.0
        resp_starts_sec, resp_ends_sec, _ = _segments_from_binary(resp_signal > 0, resp_dt_sec)
        resp_linked_limb_ratio = _event_overlap_ratio(
            limb_starts_sec,
            limb_ends_sec,
            resp_starts_sec,
            resp_ends_sec,
            RESP_LINKED_LIMB_WINDOW_SEC,
        )

        transition_times_sec = np.array([], dtype=float)
        if len(stages) > 1:
            transition_epoch_idx = np.where(np.diff(stages) != 0)[0] + 1
            transition_times_sec = transition_epoch_idx.astype(float) * EPOCH_SEC

        if total_limb_count > 0 and len(transition_times_sec) > 0:
            limb_linked_transition_hits = 0
            for s0, e0 in zip(limb_starts_sec, limb_ends_sec):
                hit = np.any(
                    (transition_times_sec >= (s0 - TRANSITION_WINDOW_SEC)) &
                    (transition_times_sec <= (e0 + TRANSITION_WINDOW_SEC))
                )
                limb_linked_transition_hits += int(hit)
            limb_linked_transition_ratio = _safe_div(limb_linked_transition_hits, total_limb_count)
        else:
            limb_linked_transition_ratio = 0.0

        plm_duration_ratio = _safe_div(np.sum(durations_by_class[LIMB_PLM]), tst_sec)
        isolated_duration_ratio = _safe_div(np.sum(durations_by_class[LIMB_ISOLATED]), tst_sec)
        periodic_limb_burden = plmi + plm_duration_ratio
        non_periodic_limb_burden = isolated_limb_index + isolated_duration_ratio

        features = [
            float(total_limb_count),
            total_limb_duration_sec,
            limb_movement_index,
            limb_burden_ratio,
            float(isolated_limb_count),
            float(plm_count),
            isolated_limb_index,
            plmi,
            plm_ratio,
            mean_limb_duration,
            max_limb_duration,
            p75_limb_duration,
            p90_limb_duration,
            p95_limb_duration,
            mean_inter_limb_interval,
            std_inter_limb_interval,
            limb_burst_ratio,
            nrem_limb_index,
            rem_limb_index,
            n2_limb_index,
            n3_limb_index,
            early_limb_index,
            late_limb_index,
            delta_limb_index,
            limb_linked_arousal_ratio,
            arousal_followed_by_limb_ratio,
            resp_linked_limb_ratio,
            limb_linked_transition_ratio,
            periodic_limb_burden,
            non_periodic_limb_burden,
        ]

        return np.asarray(features, dtype=np.float32)


    def extract_all_algorithmic_features(self, algo_data):
        """
        Concatenate all algorithmic annotation features into one vector.
        """
        if not algo_data:
            return np.zeros(186, dtype=np.float32)

        features = np.hstack([
            self.extract_algorithmic_annotations_features(algo_data),
            self.extract_algorithmic_arousal_event_features(algo_data),
            self.extract_algorithmic_respiratory_event_features(algo_data),
            self.extract_algorithmic_limb_event_features(algo_data),
        ]).astype(np.float32)

        if len(features) != 186:
            raise RuntimeError(
                f"Combined algorithmic feature dimension mismatch: expected {186}, got {len(features)}"
            )

        return features
