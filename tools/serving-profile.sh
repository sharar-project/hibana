#!/usr/bin/env bash
# Shared by run-sharar.sh (and tests/unit/test_serving_profile.py): the default each serving profile gives
# every speed/fidelity knob that the environment or the config file does not set explicitly
# (#27, docs/optimizations.md#serving-profiles).
#   exact     the target's own distribution: exact acceptance, no lossy weight copies (benchmarks, traces)
#   balanced  relaxed greedy acceptance top-2 within 1.0 logit; the 27B's calibrated 3-bit decode copies
#   fast      (default) everything measured to be worth it: relaxed top-3 within 2.0 logits for greedy and
#             sampled requests, the calibrated 3-bit decode copies (the 27B's layers and FP8 draft, Flash-Next's
#             trunk) and device-resident acceptance for Flash-Next's MTP rounds (#36)
#   throughput  fast, with up to 16 requests decoded together (#59: evals, many agents). Flash-Next verifies 3
#             drafts at 9-16 requests, within the 64 rows its 3-bit trunk copies serve (fast's weights at every
#             load, for 1.4% of wall time against 4 drafts); the 27B keeps its 15 drafts at every batch (GSM8K x1.74 against
#             x1.29 for 8:7,16:3), so its batches of five or more verify on the original NVFP4/FP8 weights
# serving_profile PROFILE TARGET sets P_ACCEPT, P_ACCEPT_SAMPLED, P_W3, P_DEVICE_ACCEPT, P_MAX_BATCH and
# P_SPEC_DEPTHS; it returns 1 for an unknown profile. TARGET is SHARAR_TARGET (empty: the 27B). Where the 3-bit
# copies come from is not a profile choice: run-sharar.sh serves the target's calibrated checkpoint (#72).
serving_profile() {
  P_MAX_BATCH=4; P_SPEC_DEPTHS=""
  case "$1" in
    exact)    P_ACCEPT=off;   P_ACCEPT_SAMPLED=0; P_W3=0; P_DEVICE_ACCEPT=0 ;;
    balanced) P_ACCEPT=2:1.0; P_ACCEPT_SAMPLED=0; P_DEVICE_ACCEPT=0
              if [[ "$2" == flashnext ]]; then P_W3=0; else P_W3=1; fi ;;  # Flash-Next's 3-bit trunk is fast's
    fast)     P_ACCEPT=3:2.0; P_ACCEPT_SAMPLED=1; P_W3=1
              if [[ "$2" == flashnext ]]; then P_DEVICE_ACCEPT=1; else P_DEVICE_ACCEPT=0; fi ;;  # the 27B's gain needs lazy round completion (#18)
    throughput) serving_profile fast "$2"; P_MAX_BATCH=16
              if [[ "$2" == flashnext ]]; then P_SPEC_DEPTHS=16:3; fi ;;  # the 27B: SHARAR_SPEC everywhere
    *) return 1 ;;
  esac
}
