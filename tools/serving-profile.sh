#!/usr/bin/env bash
# Shared by run-sharar.sh (and tests/unit/test_serving_profile.py): the default each serving profile gives
# every speed/fidelity knob that the environment or the config file does not set explicitly
# (#27, docs/optimizations.md#serving-profiles).
#   exact     the target's own distribution: exact acceptance, no lossy weight copies (benchmarks, traces)
#   balanced  relaxed greedy acceptance top-2 within 1.0 logit; the 27B's calibrated 3-bit decode copies
#   fast      (default) everything measured to be worth it: relaxed top-3 within 2.0 logits for greedy and
#             sampled requests, the 27B's calibrated 3-bit copies, Flash-Next's calibrated 3-bit trunk and
#             device-resident acceptance for its MTP rounds (#36)
#   throughput  fast, with up to 16 requests decoded together (#59: evals, many agents). Flash-Next verifies 3
#             drafts at 9-16 requests, within the 64 rows its 3-bit trunk copies serve (fast's weights at every
#             load, x1.71 wall against 4 drafts); the 27B keeps its 15 drafts at every batch (GSM8K x1.74 against
#             x1.29 for 8:7,16:3), so its batches of five or more verify on the original NVFP4/FP8 weights
# serving_profile PROFILE TARGET sets P_ACCEPT, P_ACCEPT_SAMPLED, P_W3, P_W3_DRAFT, P_FN_W3, P_DEVICE_ACCEPT,
# P_MAX_BATCH and P_SPEC_DEPTHS; it returns 1 for an unknown profile. TARGET is SHARAR_TARGET (empty: the 27B).
serving_profile() {
  P_MAX_BATCH=4; P_SPEC_DEPTHS=""
  case "$1" in
    exact)    P_ACCEPT=off;   P_ACCEPT_SAMPLED=0; P_W3=0; P_W3_DRAFT=0; P_FN_W3=off;  P_DEVICE_ACCEPT=0 ;;
    balanced) P_ACCEPT=2:1.0; P_ACCEPT_SAMPLED=0; P_W3=1; P_W3_DRAFT=1; P_FN_W3=off;  P_DEVICE_ACCEPT=0 ;;
    fast)     P_ACCEPT=3:2.0; P_ACCEPT_SAMPLED=1; P_W3=1; P_W3_DRAFT=1; P_FN_W3=auto
              if [[ "$2" == flashnext ]]; then P_DEVICE_ACCEPT=1; else P_DEVICE_ACCEPT=0; fi ;;  # the 27B's gain needs lazy round completion (#18)
    throughput) serving_profile fast "$2"; P_MAX_BATCH=16
              if [[ "$2" == flashnext ]]; then P_SPEC_DEPTHS=16:3; fi ;;  # the 27B: SHARAR_SPEC everywhere
    *) return 1 ;;
  esac
}
