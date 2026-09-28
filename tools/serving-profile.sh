#!/usr/bin/env bash
# Shared by run-sharar.sh (and tests/unit/test_serving_profile.py): the default each serving profile gives
# every speed/fidelity knob that the environment or the config file does not set explicitly
# (#27, docs/optimizations.md#serving-profiles).
#   exact     the target's own distribution: exact acceptance, no lossy weight copies (benchmarks, traces)
#   balanced  relaxed greedy acceptance top-2 within 1.0 logit; the 27B's calibrated 3-bit decode copies
#   fast      (default) everything measured to be worth it: relaxed top-3 within 2.0 logits for greedy and
#             sampled requests, the 27B's calibrated 3-bit copies, Flash-Next's calibrated 3-bit trunk and
#             device-resident acceptance for its MTP rounds (#36)
# serving_profile PROFILE TARGET sets P_ACCEPT, P_ACCEPT_SAMPLED, P_W3, P_W3_DRAFT, P_FN_W3 and P_DEVICE_ACCEPT;
# it returns 1 for an unknown profile. TARGET is SHARAR_TARGET (empty: the 27B).
serving_profile() {
  case "$1" in
    exact)    P_ACCEPT=off;   P_ACCEPT_SAMPLED=0; P_W3=0; P_W3_DRAFT=0; P_FN_W3=off;  P_DEVICE_ACCEPT=0 ;;
    balanced) P_ACCEPT=2:1.0; P_ACCEPT_SAMPLED=0; P_W3=1; P_W3_DRAFT=1; P_FN_W3=off;  P_DEVICE_ACCEPT=0 ;;
    fast)     P_ACCEPT=3:2.0; P_ACCEPT_SAMPLED=1; P_W3=1; P_W3_DRAFT=1; P_FN_W3=auto
              if [[ "$2" == flashnext ]]; then P_DEVICE_ACCEPT=1; else P_DEVICE_ACCEPT=0; fi ;;  # the 27B's gain needs lazy round completion (#18)
    *) return 1 ;;
  esac
}
