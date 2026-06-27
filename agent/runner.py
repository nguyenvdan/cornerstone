"""The agent loop.

Two execution modes over the *same* toolbelt:

* **autonomous** — when ``ANTHROPIC_API_KEY`` is set (and the ``anthropic`` SDK
  is installed), an LLM plans and calls tools itself, iterating until it can
  synthesize a grounded report.
* **scripted** — otherwise, a deterministic orchestrator routes the request to
  the relevant tools, runs them in sequence, and synthesizes the report from
  their outputs. No LLM required, so the system is fully demonstrable offline.

Both return an ``AgentResult`` with an inspectable reasoning trace and a final
report grounded in tool outputs.

    uv run python -m agent.runner "Analyze the Wizards' path with Dybantsa as cornerstone"
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

from pipelines import config

from . import report
from .tools import Context, Tool, build_tools

DEFAULT_QUERY = "Analyze the Wizards' path with AJ Dybantsa as their cornerstone."
AGENT_MODEL = os.environ.get("CORNERSTONE_AGENT_MODEL", "claude-sonnet-4-6")
MAX_TURNS = 8

SYSTEM_PROMPT = (
    "You are Cornerstone, a basketball decision-support analyst. Answer the user's "
    "question by calling the provided tools and grounding every claim in their "
    "outputs. Rules: (1) never invent a statistic — only cite numbers returned by "
    "tools; (2) projections are probability distributions, so always state "
    "uncertainty (tier probabilities, ranges) rather than single outcomes; (3) plan "
    "multiple steps — look up the player, find comparables, project development, and "
    "evaluate roster fit when the question is about team building; (4) finish with a "
    "structured, cited scouting & strategy report in Markdown."
)


@dataclass
class AgentStep:
    n: int
    thought: str
    tool: str
    args: dict
    result_summary: str


@dataclass
class AgentResult:
    query: str
    mode: str
    steps: list[AgentStep] = field(default_factory=list)
    report_markdown: str = ""
    tool_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _summarize(result: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    keys = [k for k in ("fit_score", "p_starter_plus", "expected_career_vorp",
                        "n_comparables", "n_rotation") if k in result]
    if "comparables" in result:
        return f"{len(result['comparables'])} comparables (top: " \
               f"{result['comparables'][0]['player_name']})"
    if keys:
        return ", ".join(f"{k}={result[k]}" for k in keys)
    return "ok"


# --------------------------------------------------------------------------
# Scripted orchestrator (no LLM)
# --------------------------------------------------------------------------
def _detect_cornerstone(query: str, ctx: Context) -> str:
    q = query.lower()
    if "dybantsa" in q or not query.strip():
        return "AJ Dybantsa"
    for name in ctx.prospects["player_name"]:
        if isinstance(name, str) and name.lower() in q:
            return name
    return "AJ Dybantsa"


def _plan(query: str) -> list[tuple[str, str]]:
    """Return (tool, thought) steps selected from the request's intent."""
    q = query.lower()
    plan = [
        ("lookup_prospect", "Establish the player's pre-draft profile."),
        ("find_comparables", "Ground the projection in real historical analogs."),
        ("project_development", "Quantify the development outlook as a distribution."),
    ]
    wants_team = any(w in q for w in ("roster", "team", "build", "wizards", "fit",
                                      "around", "complement", "construct", "path"))
    if wants_team or not query.strip():
        plan.append(("team_skill_summary", "Inventory the current roster's skills."))
        plan.append(("evaluate_roster_fit", "Assess fit and the highest-leverage needs."))
    return plan


def _run_scripted(query: str, ctx: Context, tools: dict[str, Tool]) -> AgentResult:
    res = AgentResult(query=query, mode="scripted")
    cornerstone = _detect_cornerstone(query, ctx)
    for i, (tool_name, thought) in enumerate(_plan(query), 1):
        tool = tools[tool_name]
        if tool_name in ("lookup_prospect", "find_comparables", "project_development"):
            args = {"name": cornerstone}
        elif tool_name == "evaluate_roster_fit":
            args = {"team": "WAS", "cornerstone": cornerstone}
        else:
            args = {"team": "WAS"}
        result = tool.func(**args)
        res.tool_results[tool_name] = result
        res.steps.append(AgentStep(i, thought, tool_name, args, _summarize(result)))
    res.report_markdown = report.synthesize(query, res.tool_results)
    return res


# --------------------------------------------------------------------------
# Autonomous orchestrator (LLM tool-calling)
# --------------------------------------------------------------------------
def _run_autonomous(query: str, ctx: Context, tools: dict[str, Tool]) -> AgentResult:
    import anthropic  # lazy: only needed in this mode

    client = anthropic.Anthropic()
    specs = [t.anthropic_spec() for t in tools.values()]
    messages = [{"role": "user", "content": query}]
    res = AgentResult(query=query, mode="autonomous")
    step_n = 0

    for _ in range(MAX_TURNS):
        resp = client.messages.create(model=AGENT_MODEL, max_tokens=2048,
                                      system=SYSTEM_PROMPT, tools=specs, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            res.report_markdown = text
            break
        tool_results_blocks = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            step_n += 1
            thought = next((b.text for b in resp.content if b.type == "text"), "")
            result = tools[block.name].func(**block.input)
            res.tool_results[block.name] = result
            res.steps.append(AgentStep(step_n, thought.strip()[:200], block.name,
                                       dict(block.input), _summarize(result)))
            tool_results_blocks.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": json.dumps(result, default=str)})
        messages.append({"role": "user", "content": tool_results_blocks})
    if not res.report_markdown:
        res.report_markdown = report.synthesize(query, res.tool_results)
    return res


# --------------------------------------------------------------------------
def run(query: str = DEFAULT_QUERY, ctx: Context | None = None,
        force_scripted: bool = False) -> AgentResult:
    ctx = ctx or Context()
    tools = build_tools(ctx)
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY")) and not force_scripted
    if use_llm:
        try:
            return _run_autonomous(query, ctx, tools)
        except Exception as exc:  # fall back rather than fail the demo
            result = _run_scripted(query, ctx, tools)
            result.steps.insert(0, AgentStep(0, f"(LLM unavailable: {exc}; using scripted "
                                             "orchestrator)", "—", {}, ""))
            return result
    return _run_scripted(query, ctx, tools)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    query = " ".join(argv) if argv else DEFAULT_QUERY
    result = run(query)

    print(f"Cornerstone agent — mode: {result.mode}")
    print(f"Request: {query}\n")
    print("Reasoning trace:")
    for s in result.steps:
        print(f"  [{s.n}] {s.thought}")
        if s.tool != "—":
            print(f"      → {s.tool}({', '.join(f'{k}={v}' for k, v in s.args.items())})  "
                  f"⇒ {s.result_summary}")
    print("\n" + "=" * 70 + "\n")
    print(result.report_markdown)

    path = config.PROCESSED / "agent_report.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"\n\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
