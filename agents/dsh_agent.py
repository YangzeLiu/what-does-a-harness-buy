"""Harbor adapter for DeepSeek Harness (dsh, npm @deepseek-ai/dsh), headless profile.

agent-harness ablation, 2026-09-08 (Harbor 0.22.0 custom agent via ``import_path``).

Job YAML::

    agents:
      - import_path: "dsh_agent:DshAgent"
        model_name: deepseek/deepseek-v4-flash
        kwargs:
          version: "0.1.2-rc.1"

Run harbor with ``PYTHONPATH=<dir containing this file>``. ``DEEPSEEK_API_KEY`` must be
in the harbor process environment; it reaches the container only as
per-exec env of the agent run, never as a file and never in logs (Harbor redacts it).

What the arm is: dsh exactly as shipped (default deepseek-official provider, default
reasoning effort, default tool set incl. subagents) except
  * web tools disabled (row ``tool-web``: web_search/web_fetch) -- no-web condition,
    needed because DeepSeek's web_search goes through api.deepseek.com, which the
    network allowlist must permit for the model itself;
  * approval never / sandbox off via DSH_PERMISSION_MODE=danger-full-access (headless
    has no UI to answer prompts; the Harbor container is the sandbox);
  * session JSONL uncompressed under /logs/agent so Harbor collects it.
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

DEFAULT_DSH_VERSION = "0.1.2-rc.1"  # npm dist-tag latest on 2026-09-03; alphas break daily
NODE_VERSION = "v22.20.0"  # dsh engines: ^22.19.0 || >=24.0.0
NODE_URL = (
    "https://cdn.npmmirror.com/binaries/node/"
    f"{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.gz"
)
# Host-local setup mirror on the docker gateway (host-a serves $D/mirror on :8090): tried first with
# a short timeout because the shared 100 Mb/s uplink is saturated by image pulls; CDN is the fallback.
NODE_MIRROR_URL = f"http://172.17.0.1:8090/node-{NODE_VERSION}-linux-x64.tar.gz"
NPM_MIRROR = "https://registry.npmmirror.com"

DSH_HOME = "/logs/agent/dsh/home"
PATCH_PATH = "/installed-agent/dsh.patch.yml"
OUTPUT_PATH = "/logs/agent/dsh.txt"
WORKDIR = "/testbed"

PATCH_YAML = """\
# agent-harness overlay: pin model, drop web tools, plain JSONL sessions in the log dir
- id: agent-default-model
  config:
    provider: deepseek-official
    model: {model}
- id: tool-web
  disabled: true
- id: session-persistence-jsonl
  config:
    root: {home}/sessions
    compression: none
"""


class DshAgent(BaseInstalledAgent):
    """DeepSeek Harness (dsh) run through ``dsh --profile headless``."""

    def __init__(
        self,
        *args: Any,
        soft_timeout_sec: int = 2760,
        kill_after_sec: int = 120,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if self._version is None:
            self._version = DEFAULT_DSH_VERSION
        # dsh buffers the session JSONL in memory and flushes it only when the
        # turn ends, so a hard kill at Harbor's agent timeout (3000 s for our
        # tasks) would lose the token accounting. Interrupt dsh ourselves a bit
        # earlier: SIGINT -> dsh exits 130 (non-zero -> NonZeroAgentExitCodeError,
        # verifier still runs), and the tree disposal gets a chance to flush.
        self._soft_timeout_sec = int(soft_timeout_sec)
        self._kill_after_sec = int(kill_after_sec)

    @staticmethod
    @override
    def name() -> str:
        return "dsh"

    @override
    def get_version_command(self) -> str | None:
        return "export PATH=/usr/local/bin:$PATH; dsh --version"

    @property
    def _model_id(self) -> str:
        if not self.model_name:
            raise ValueError("model_name is required, e.g. deepseek/deepseek-v4-flash")
        return self.model_name.split("/", 1)[-1]

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "bash", "coreutils"))
        pkg = shlex.quote(f"@deepseek-ai/dsh@{self._version}")
        npm_flags = "--no-fund --no-audit --loglevel=error"
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                "(curl -fsSL --connect-timeout 5 --max-time 300 --retry 2 "
                f"-o /tmp/node.tgz {NODE_MIRROR_URL} || "
                "curl -fsSL --connect-timeout 20 --max-time 900 --retry 4 "
                f"--retry-all-errors --retry-delay 10 -o /tmp/node.tgz {NODE_URL}) && "
                "tar -xzf /tmp/node.tgz -C /usr/local --strip-components=1 && "
                "rm -f /tmp/node.tgz && export PATH=/usr/local/bin:$PATH && hash -r && "
                "node -v && npm -v && "
                f"(npm install -g {npm_flags} --registry={NPM_MIRROR} {pkg} || "
                f"npm install -g {npm_flags} {pkg}) && "
                "dsh --version"
            ),
        )
        await self._upload_config_text(
            environment,
            content=PATCH_YAML.format(model=self._model_id, home=DSH_HOME),
            remote_path=PATCH_PATH,
            filename="dsh.patch.yml",
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in the harbor process env")
        env = {
            "DEEPSEEK_API_KEY": api_key,
            "DSH_HOME": DSH_HOME,
            "DSH_AGENTS_HOME": f"{DSH_HOME}/agents-home",
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_TELEMETRY_DISABLED": "1",
        }
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {DSH_HOME} && cd {WORKDIR} && "
                "export PATH=/usr/local/bin:$PATH && "
                f"timeout -s INT -k {self._kill_after_sec} {self._soft_timeout_sec} "
                f"dsh --patch {PATCH_PATH} --profile headless "
                f"{shlex.quote(self.render_instruction(instruction))} "
                f"2>&1 </dev/null | stdbuf -oL tee {OUTPUT_PATH}"
            ),
            env=env,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        sessions = self.logs_dir / "dsh" / "home" / "sessions"
        files = sorted(sessions.rglob("*.jsonl")) if sessions.is_dir() else []
        usage = {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0,
                 "cacheWriteTokens": 0, "reasoningTokens": 0}
        n_msgs = n_tools = 0
        end_reason: Any = None
        request_cfg: Any = None
        for path in files:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                t = ev.get("type")
                data = ev.get("data") or {}
                if t == "assistant/message":
                    n_msgs += 1
                    for k in usage:
                        usage[k] += (data.get("usage") or {}).get(k) or 0
                elif t == "turn/end":
                    end_reason = data.get("reason")
                elif t == "request/header" and request_cfg is None:
                    request_cfg = (data.get("header") or {}).get("config")
                elif isinstance(t, str) and t.startswith("tool/") and t.endswith("call"):
                    n_tools += 1
        if n_msgs:
            context.n_input_tokens = usage["inputTokens"] + usage["cacheReadTokens"]
            context.n_output_tokens = usage["outputTokens"]
            context.n_cache_tokens = usage["cacheReadTokens"]
        context.metadata = {
            "agent": "dsh",
            "dsh_version": self._version,
            "node_version": NODE_VERSION,
            "soft_timeout_sec": self._soft_timeout_sec,
            "n_session_files": len(files),
            "n_assistant_messages": n_msgs,
            "n_tool_events": n_tools,
            "usage_raw": usage,
            "turn_end_reason": end_reason,
            "request_config": request_cfg,
        }
