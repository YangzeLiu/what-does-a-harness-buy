"""Build configs/qwen36_nothink.jinja: Qwen3.6-35B-A3B's chat template with thinking hard-disabled.

Usage: python scripts/mk_nothink_template.py <model_dir>/chat_template.jinja [configs/qwen36_nothink.jinja]
The input is the chat_template.jinja shipped with the Qwen3.6-35B-A3B weights.  The script replaces
the enable_thinking branch (lines 149-153) with the unconditional empty think block and asserts the
lines it replaces, so a different template revision fails loudly instead of being patched blindly.
"""
import os, sys
if len(sys.argv) < 2:
    sys.exit(__doc__)
SRC = sys.argv[1]
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "qwen36_nothink.jinja")
Q=chr(39)
lines=open(SRC).read().split("\n")
blk=lines[148:153]
assert blk[0].strip()=="{%- if enable_thinking is defined and enable_thinking is false %}", blk[0]
assert blk[1].strip()=="{{- "+Q+"<think>\\n\\n</think>\\n\\n"+Q+" }}", blk[1]
assert blk[2].strip()=="{%- else %}" and blk[4].strip()=="{%- endif %}", blk
new=["    {#- NOTHINK: thinking hard-disabled regardless of enable_thinking / reasoning_effort (agent-harness thinking-off arm) -#}", "    "+blk[1].strip()]
out="\n".join(lines[:148]+new+lines[153:])
open(DST,"w").write(out)
print("ok", len(lines), "->", out.count(chr(10))+1)
