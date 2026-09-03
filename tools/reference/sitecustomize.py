"""Patch SGLang's tensor dumper to record decoder-layer outputs, not every leaf.

Loaded automatically (PYTHONPATH) in every process of the reference engine when
SHARAR_REFERENCE_DUMP=1. It replaces TensorDumper._add_hook_recursive.

SGLang invokes the root model's forward() directly, so a root forward hook
never fires. Instead the embedding module's pre-hook marks the start of a pass:
anything collected so far belongs to the previous pass and is written then.
The caller must issue one extra request at the end to flush the last pass.

Each pass file contains: input_ids, embed, layers.N as [hidden_states,
residual], norm (final norm output) and logits (last-position logits).
"""

import os
import re

if os.environ.get("SHARAR_REFERENCE_DUMP") == "1":
    try:
        import sglang.srt.debug_utils.tensor_dump_forward_hook as _hook_mod
    except Exception:  # pragma: no cover - only the scheduler process has it
        _hook_mod = None

    if _hook_mod is not None:
        _LAYER = re.compile(r"(^|\.)layers\.\d+$")

        def _add_hook_recursive(self, model, prefix, top_level_module_name, layers_module_name):
            dumper = self

            def embed_pre_hook(module, inputs):
                if dumper._current_tensors:
                    dumper.dump_current_tensors()
                dumper.add_tensor("input_ids", inputs[0])

            def make_hook(name):
                def hook(module, inputs, output):
                    if output is None:
                        return
                    if name == "logits" and getattr(output, "next_token_logits", None) is None:
                        return  # passes that return hidden states only
                    dumper.add_tensor(name, output)

                return hook

            for name, module in model.named_modules():
                if not name:
                    continue
                if name.endswith("embed_tokens"):
                    module.register_forward_pre_hook(embed_pre_hook)
                    module.register_forward_hook(make_hook("embed"))
                elif _LAYER.search(name) is not None:
                    module.register_forward_hook(make_hook("layers." + name.rsplit(".", 1)[-1]))
                elif name.endswith("norm") and "layers" not in name:
                    module.register_forward_hook(make_hook("norm"))
                elif name.endswith("logits_processor"):
                    module.register_forward_hook(make_hook("logits"))
            return True, 1

        _hook_mod.TensorDumper._add_hook_recursive = _add_hook_recursive
