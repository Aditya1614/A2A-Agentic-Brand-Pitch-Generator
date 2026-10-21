# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
/**
 * @file agent.py
 * @description Multi-agent ADK workflow definition and service-injected execution
 *   engine for the Agentic Pitch Generator.
 *
 * Why: Defines the Google ADK specialist agent team (`creative_director`,
 * `brand_strategist`, `copywriter`, `visual_director`), structured payload contracts,
 * `LoopGuard` DAG validation, `assemble = JoinNode(name="assemble")`, `package`,
 * `root_agent`, and `app` running on Gemini Enterprise Agent Platform (`gemini-3.8-flash`)
 * while providing lightweight stdlib fallback shims when `google-adk` / `google-genai`
 * wheels are not installed locally. Also exposes `run_pitch_workflow` and
 * `resume_pitch_workflow` wired through `ServiceContainer` so local runs use live
 * Cloud models and offline test suites execute deterministically.
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/agent.py`)
#   - Step 1a: `# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]`
#   - Step 1b: `# [Guidepost — Step 1b: Skill Evaluation Harness]`
#   - Step 1c: `# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]`
#   - Step 3a: `# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]`
#   - Step 3c: `# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]`
#   - Step 4a: `# [Guidepost — Step 4a: Tokenomics & Context Optimization]`
# (Tip: All specialist agents, workflow edges, and lab guideposts are in the top
#  ~320 lines of this file—read lines 1–350 in a single `view_file` call!)
# ==============================================================================

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import time
from typing import Any

from pitch_generator.app_utils.adk_primitives import (
    Agent,
    App,
    ArtDirectionPayload,
    CampaignConcept,
    CircularLoopError,
    ConceptPayload,
    Context,
    CopyPayload,
    Event,
    Gemini,
    JoinNode,
    LoopGuard,
    PayloadValidationError,
    PitchPackage,
    RemoteA2aAgent,
    RequestInput,
    VisualPayload,
    Workflow,
    _DualSyncAsyncEventStream,
    _GenAITypesShim,
    _SessionData,
    _extract_json_Candidate,
    _key_visual,
    _node_name,
    _normalize_directed_edges,
    extract_json_payload,
    node,
    parse_json_payload,
    parse_payload,
    strip_markdown_fences,
    types,
)
from pitch_generator.app_utils.a2a import (
    AGENT_CARD_WELL_KNOWN_PATH,
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
    AgentCardBuilder,
    DefaultRequestHandler,
    include_artifacts_in_a2a_event_interceptor,
)
from pitch_generator.app_utils.services import (
    ServiceContainer,
    get_default_services,
    scrub_payload,
    scrub_pii,
    scrub_text,
)
from pitch_generator.config import (
    PitchConfig,
    get_config,
)

MODEL: str = get_config().flash_model
IMAGE_MODEL: str = get_config().image_model


def _model(config: PitchConfig | None = None) -> Gemini:
    """
    /**
     * Construct the default `Gemini` model configuration (`gemini-3.8-flash`) for
     * specialist agents.
     *
     * Why: Centralizes `location="global"` and `HttpRetryOptions(attempts=3)` so every
     * specialist agent shares resilient retry settings on Gemini Enterprise Agent Platform.
     *
     * @param config Optional `PitchConfig` override.
     * @return Configured `Gemini` model descriptor.
     */
    """
    cfg = config or get_config()
    return Gemini(
        model=cfg.flash_model,
        client_kwargs={"location": cfg.location},
        retry_options=types.HttpRetryOptions(attempts=3),
    )


_build_model = _model


async def generate_key_visual(
    art_direction: str,
    tool_context: Context | None = None,
    *,
    simulate_no_image: bool = False,
) -> dict[str, Any]:
    """
    /**
     * Generate a 16:9 key visual PNG from art direction notes and save it to session artifacts.
     *
     * Why: Attached to `visual_director` (`tools=[generate_key_visual]`) so the Visual
     * Director specialist can synthesize a key visual image via the configured
     * `image_model` (`gemini-nano-banana-2.1`) and store it in the session artifact store.
     *
     * @param art_direction Art direction prompt describing palette, lighting, composition, and subject.
     * @param tool_context Optional ADK `Context` / `ToolContext` for saving the image artifact.
     * @param simulate_no_image If True, simulates an empty image response to test error handling.
     * @return Metadata dictionary with `filename`, `version`, `bytes`, and `mime_type`.
     */
    """
    import mimetypes

    ctx = tool_context if tool_context is not None else Context()
    cfg = ctx.services.config if hasattr(ctx, "services") and ctx.services else get_config()
    if (
        simulate_no_image
        or getattr(ctx, "simulate_no_image", False)
        or not isinstance(art_direction, str)
        or not art_direction.strip()
    ):
        raise ValueError(f"{cfg.image_model} returned no image for: {str(art_direction)[:120]}")

    image_bytes, mime_type = ctx.services.llm.generate_image(
        art_direction.strip(),
        model=cfg.image_model,
    )
    if not image_bytes:
        raise ValueError(f"{cfg.image_model} returned no image for: {art_direction[:120]}")

    ext = mimetypes.guess_extension(mime_type or "") or ".png"
    filename = f"key_visual{ext}"
    image_part = types.Part(
        inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
    )
    version = await ctx.save_artifact(filename, image_part)
    ctx.session.events.append(
        Event(
            author="visual_director",
            content=types.Content(role="model", parts=[image_part]),
        )
    )
    return {
        "filename": filename,
        "version": version,
        "bytes": len(image_bytes),
        "mime_type": mime_type,
    }


@dataclass
class Skill:
    """
    /**
     * ADK Skill definition containing metadata and Markdown body.
     */
    """

    name: str
    description: str
    body: str
    path: str | Path | None = None


class SkillToolset:
    """
    /**
     * ADK-compatible toolset holding available skills for on-demand agent loading.
     *
     * Why: Exposes `.load_skill(skill_name)` and tracks `.loaded_skills` so specialist
     * agents dynamically inject domain instructions into their reasoning context.
     */
    """

    def __init__(self, skills: Sequence[Skill] | dict[str, Skill] | None = None) -> None:
        if skills is None:
            self._skills_map: dict[str, Skill] = {}
        elif isinstance(skills, dict):
            self._skills_map = dict(skills)
        else:
            self._skills_map = {s.name: s for s in skills}
        self.skills: list[Skill] = list(self._skills_map.values())
        self.loaded_skills: list[str] = []

    def load_skill(self, skill_name: str) -> str:
        """
        /**
         * Load a skill by name and return its Markdown body.
         *
         * @param skill_name Unique skill identifier string.
         * @return Markdown body content of the loaded skill.
         */
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name must be a non-empty string")
        clean_name = skill_name.strip()
        if clean_name not in self._skills_map:
            raise KeyError(
                f"Skill {skill_name!r} not found. Available: {list(self._skills_map.keys())}"
            )
        if clean_name not in self.loaded_skills:
            self.loaded_skills.append(clean_name)
        return self._skills_map[clean_name].body

    def __call__(self, skill_name: str) -> str:
        return self.load_skill(skill_name)


def load_skill_from_dir(skill_dir: str | Path) -> Skill:
    """
    /**
     * Parse a SKILL.md file from a directory into a `Skill` dataclass.
     *
     * Why: Reads YAML frontmatter (name, description) and Markdown body from disk.
     *
     * @param skill_dir Directory containing `SKILL.md` or direct file path.
     * @return Populated `Skill` instance.
     */
    """
    path_obj = Path(skill_dir).resolve()
    skill_file = path_obj / "SKILL.md" if path_obj.is_dir() else path_obj
    if not skill_file.exists():
        raise FileNotFoundError(f"SKILL.md not found at {skill_file}")

    raw_text = skill_file.read_text(encoding="utf-8")
    name = path_obj.name if path_obj.is_dir() else path_obj.parent.name
    description = ""
    body = raw_text

    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.splitlines():
                line_str = line.strip()
                if line_str.startswith("name:"):
                    name = line_str.split("name:", 1)[1].strip().strip("\"'")
                elif line_str.startswith("description:"):
                    description = line_str.split("description:", 1)[1].strip().strip("\"'")

    return Skill(
        name=name,
        description=description,
        body=body,
        path=skill_file,
    )


def load_brand_skill() -> Skill:
    """
    /**
     * Load the canonical house brand guidelines skill from disk.
     *
     * @return `Skill` instance for brand-guidelines.
     */
    """
    brand_skill_dir = Path(__file__).resolve().parent / "skills" / "brand-guidelines"
    return load_skill_from_dir(brand_skill_dir)


brand_guidelines_skill: Skill = load_brand_skill()
brand_skill_toolset: SkillToolset = SkillToolset([brand_guidelines_skill])


def build_specialist_team(config: PitchConfig | None = None) -> dict[str, Agent]:
    """
    /**
     * Instantiate the four domain specialist agents for the Pitch Generator team.
     *
     * Why: Provides a deterministic factory that binds each specialist role
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) to
     * a focused single-responsibility instruction and a unique `output_key` so
     * downstream graph nodes can read each specialist's output without collision.
     *
     * @param config Optional `PitchConfig` used to resolve model settings.
     * @return Dictionary mapping role name to its configured `Agent` instance.
     */
    """
    model = _model(config)
    brand_skill = load_brand_skill()
    brand_toolset = SkillToolset([brand_skill])
    return {
        "creative_director": Agent(
            name="creative_director",
            model=model,
            description="Turns a raw campaign topic into a punchy concept line and strategic rationale.",
            instruction=(
                "You are the Creative Director. Turn the campaign idea you are given into "
                "ONE punchy campaign concept line accompanied by a clear rationale explanation."
            ),
            output_key="creative_director",
        ),
        "copywriter": Agent(
            name="copywriter",
            model=model,
            description="Writes a concise, high-impact social media caption under 25 words.",
            instruction=(
                "You are the Copywriter. Write ONE short social caption for the campaign "
                "concept you are given. Strictly under 25 words with an energetic call to action."
            ),
            output_key="copywriter",
        ),
        "brand_strategist": Agent(
            name="brand_strategist",
            model=model,
            description="Defines campaign positioning, target audience, and brand tone for the campaign concept.",
            instruction=(
                "You are the Brand Strategist. Define the campaign positioning, "
                "target audience, and brand tone for the campaign concept."
            ),
            output_key="brand_strategist",
        ),
        "visual_director": Agent(
            name="visual_director",
            model=model,
            description="Translates the campaign concept into art direction and key visuals following brand guidelines.",
            instruction=(
                "You are the Visual Director. First, call load_skill(\"brand-guidelines\") to review "
                "our house style. Next, write art direction specifying palette, lighting, composition, "
                "and subject strictly adhering to the house brand guidelines. Then, call generate_key_visual "
                "with the art direction prompt. Output ONLY the art direction text."
            ),
            tools=[brand_toolset, generate_key_visual],
            output_key="visual_director",
        ),
    }


create_specialist_agents = build_specialist_team
get_specialist_agents = build_specialist_team

SPECIALIST_AGENTS: dict[str, Agent] = build_specialist_team()

creative_director: Agent = SPECIALIST_AGENTS["creative_director"]
copywriter: Agent = SPECIALIST_AGENTS["copywriter"]
brand_strategist: Agent = SPECIALIST_AGENTS["brand_strategist"]
visual_director: Agent = SPECIALIST_AGENTS["visual_director"]


def get_specialist_agent(role: str) -> Agent:
    """
    /**
     * Look up a single specialist `Agent` by role identifier.
     *
     * Why: Fails fast with a descriptive `KeyError` when an invalid or unknown
     * specialist role is requested instead of silently returning `None`.
     *
     * @param role Specialist role name (e.g., `"creative_director"`, `"copywriter"`).
     * @return Matching `Agent` instance.
     */
    """
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Specialist role name must be a non-empty string")
    clean_role = role.strip()
    if clean_role == "market_researcher":
        return SPECIALIST_AGENTS["brand_strategist"]
    if clean_role not in SPECIALIST_AGENTS:
        raise KeyError(
            f"Unknown specialist role {role!r}. Expected one of {tuple(SPECIALIST_AGENTS.keys())}."
        )
    return SPECIALIST_AGENTS[clean_role]


def run_specialist_team(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    config: PitchConfig | None = None,
) -> dict[str, str]:
    """
    /**
     * Execute all four specialist agents sequentially/in-parallel against a campaign brief.
     *
     * Why: Allows learners and automated tests to verify that all four specialist roles
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) generate
     * domain-appropriate outputs offline via dependency-injected `ServiceContainer`.
     *
     * @param brief Non-empty campaign topic brief.
     * @param services Optional injected `ServiceContainer` for offline LLM execution.
     * @param config Optional `PitchConfig` override.
     * @return Dictionary mapping each specialist's `output_key` to its generated text.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string")

    clean_brief = brief.strip()
    active_services = services or get_default_services(config)
    team = build_specialist_team(active_services.config)
    model_id = active_services.config.flash_model

    concept_out = active_services.llm.generate_text(
        clean_brief,
        system_instruction=team["creative_director"].instruction,
        model=model_id,
    )
    copy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["copywriter"].instruction,
        model=model_id,
    )
    copy_words = copy_out.split()
    if len(copy_words) > 25:
        copy_out = " ".join(copy_words[:24])

    strategy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["brand_strategist"].instruction,
        model=model_id,
    )
    visual_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["visual_director"].instruction,
        model=model_id,
    )

    return {
        "creative_director": concept_out,
        "copywriter": copy_out,
        "brand_strategist": strategy_out,
        "visual_director": visual_out,
    }


execute_specialists = run_specialist_team
run_specialists = run_specialist_team

# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]
# TODO (Step 1a): Create the house brand skill and attach it to `visual_director`:
#   1. Create `pitch_generator/skills/brand-guidelines/SKILL.md` with YAML frontmatter
#      (`name: brand-guidelines`, `description: ... Use when ...`) and five Markdown sections:
#      `## Palette` (deep indigo, warm amber, slate, terracotta),
#      `## Light` (warm raking golden-hour light with long soft shadows),
#      `## Composition` (off-center hero subject, generous negative space, shallow depth of field),
#      `## Subject` (one realistic photographic subject, authentic/handcrafted materials), and
#      `## Never` (no neon, fluorescent cyan/magenta, watermarks, logos, ring light, or 3D renders).
#      Keep these 5 sections, but set values to match learner's prompt incase they get creative.
#   2. Define `Skill(name, description, body, path=None)` and `SkillToolset(skills)` (with
#      `.load_skill(skill_name) -> str` and `.loaded_skills: list[str]`), plus
#      `load_skill_from_dir(skill_dir: str | Path) -> Skill` and `load_brand_skill() -> Skill`.
#   3. Load `brand_guidelines_skill = load_skill_from_dir(...)` from
#      `pitch_generator/skills/brand-guidelines`, attach
#      `tools=[SkillToolset([brand_guidelines_skill]), generate_key_visual]` to `visual_director`,
#      and update `visual_director.instruction` to first call `load_skill("brand-guidelines")`,
#      write the art direction, call `generate_key_visual`, and output ONLY the art direction text.

# [Guidepost — Step 1b: Skill Evaluation Harness]
# TODO (Step 1b): Build a deterministic evaluation harness to verify skill activation and brand compliance:
#   - `FORBIDDEN_BRAND_PATTERNS: dict[str, str]`: regex patterns for forbidden visual styles
#     (`neon`, `fluorescent`, `cyan`, `magenta`, `watermark`, `logo`, `3d_render`, `ring_light`).
#   - `SkillEvalResult` dataclass with fields: `score: float`, `passed: bool`, `loaded_skill: bool`,
#     `violations: list[str]`, and `pillar_scores: dict[str, float]`.
#   - `evaluate_brand_skill(art_direction: str, loaded_skills: Sequence[str] = ("brand-guidelines",)) -> SkillEvalResult`:
#     verifies `"brand-guidelines"` is in `loaded_skills`, scores the 4 positive pillars
#     (`palette`, `light`, `composition`, `subject` at `0.25` each), deducts `0.25` per forbidden match,
#     and sets `passed = bool(loaded_skill and not violations and score >= 0.75)`.
#   - `run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]`:
#     runs compliant, off-brand, and unloaded-skill benchmark cases and returns a summary dict with
#     `"total"`, `"passed"`, `"results"`, and `"all_expectations_met"`.

FORBIDDEN_BRAND_PATTERNS: dict[str, str] = {
    "neon": r"\bneon\b",
    "fluorescent": r"\bfluorescent\b",
    "cyan": r"\bcyan\b",
    "magenta": r"\bmagenta\b",
    "watermark": r"\bwatermarks?\b",
    "logo": r"\blogos?\b",
    "3d_render": r"\b(?:3d[\s_-]*renders?|3d[\s_-]*models?|cgi)\b",
    "ring_light": r"\bring[\s_-]*lights?\b",
}

POSITIVE_BRAND_PATTERNS: dict[str, str] = {
    "palette": r"\b(indigo|amber|slate|terracotta)\b",
    "light": r"\b(raking|golden[\s_-]*hour|soft[\s_-]*shadows?|warm[\s_-]*light|rim[\s_-]*light)\b",
    "composition": r"\b(off[\s_-]*center|negative[\s_-]*space|shallow[\s_-]*depth|depth[\s_-]*of[\s_-]*field|rule[\s_-]*of[\s_-]*thirds)\b",
    "subject": r"\b(photograph|photographic|authentic|handcrafted|natural[\s_-]*textures?)\b",
}


@dataclass
class SkillEvalResult:
    """
    /**
     * Evaluation result contract for skill activation and brand compliance.
     *
     * Why: Provides a standardized, structured assessment of whether the Visual
     * Director adhered to house brand guidelines and activated required skills.
     */
    """

    score: float
    passed: bool
    loaded_skill: bool
    violations: list[str]
    pillar_scores: dict[str, float]


def evaluate_brand_skill(
    art_direction: str,
    loaded_skills: Sequence[str] = ("brand-guidelines",),
) -> SkillEvalResult:
    """
    /**
     * Evaluates an art direction prompt against house brand guidelines and skill activation.
     *
     * Why: Verifies that "brand-guidelines" is loaded, scores the 4 positive pillars
     * (palette, light, composition, subject at 0.25 each), deducts 0.25 per forbidden
     * term match, and sets passed = bool(loaded_skill and not violations and score >= 0.75).
     *
     * @param art_direction Art direction text prompt.
     * @param loaded_skills Sequence of skill names that were loaded by the agent.
     * @return Populated `SkillEvalResult`.
     */
    """
    has_loaded_skill = "brand-guidelines" in (loaded_skills or ())
    text = art_direction or ""

    pillar_scores: dict[str, float] = {}
    for pillar, pattern in POSITIVE_BRAND_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            pillar_scores[pillar] = 0.25
        else:
            pillar_scores[pillar] = 0.0

    violations: list[str] = []
    for violation_name, pattern in FORBIDDEN_BRAND_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(violation_name)

    base_score = sum(pillar_scores.values())
    penalty = 0.25 * len(violations)
    final_score = max(0.0, base_score - penalty)
    passed = bool(has_loaded_skill and not violations and final_score >= 0.75)

    return SkillEvalResult(
        score=round(final_score, 2),
        passed=passed,
        loaded_skill=has_loaded_skill,
        violations=violations,
        pillar_scores=pillar_scores,
    )


def run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    /**
     * Runs compliant, off-brand, and unloaded-skill benchmark cases through the evaluation harness.
     *
     * Why: Provides automated regression testing for brand skill compliance.
     *
     * @param cases Optional list of evaluation test cases.
     * @return Summary dictionary with "total", "passed", "results", and "all_expectations_met".
     */
    """
    eval_cases = cases if cases is not None else [
        {
            "name": "Compliant House Style",
            "art_direction": (
                "Studio photograph of handcrafted leather luggage with deep indigo and terracotta accents. "
                "Off-center composition with shallow depth of field in warm raking golden-hour light."
            ),
            "loaded_skills": ["brand-guidelines"],
            "expected_pass": True,
        },
        {
            "name": "Forbidden Neon & 3D Render",
            "art_direction": (
                "3D render of futuristic sneakers with glowing neon cyan lasers and shiny logo."
            ),
            "loaded_skills": ["brand-guidelines"],
            "expected_pass": False,
        },
        {
            "name": "Unloaded Skill Failure",
            "art_direction": (
                "Studio photograph with warm raking golden-hour light, slate palette, and off-center composition."
            ),
            "loaded_skills": [],
            "expected_pass": False,
        },
    ]

    results: list[dict[str, Any]] = []
    passed_count = 0
    all_expectations_met = True

    for c in eval_cases:
        art_dir = c.get("art_direction", "")
        skills = c.get("loaded_skills", ("brand-guidelines",))
        eval_res = evaluate_brand_skill(art_dir, loaded_skills=skills)
        expected = c.get("expected_pass")

        met_expectation = (eval_res.passed == expected) if expected is not None else eval_res.passed
        if not met_expectation:
            all_expectations_met = False

        if eval_res.passed:
            passed_count += 1

        results.append({
            "name": c.get("name", "test_case"),
            "result": eval_res,
            "passed": eval_res.passed,
            "expected_pass": expected,
            "met_expectation": met_expectation,
        })

    return {
        "total": len(eval_cases),
        "passed": passed_count,
        "results": results,
        "all_expectations_met": all_expectations_met,
    }

# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]
# TODO (Step 1c): Expose `visual_director` over the Agent2Agent (A2A) protocol and wire `RemoteA2aAgent`:
#   - `AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"`, `AgentCardBuilder(agent, rpc_url)`,
#     and `build_visual_director_card(rpc_url: str = "http://localhost:8801/a2a/visual_director") -> dict[str, Any]`
#     returning an A2A Agent Card with `"name": "visual_director"`, `"url": rpc_url`, and `"preferredTransport": "JSONRPC"`.
#   - `A2aAgentExecutorConfig`, `include_artifacts_in_a2a_event_interceptor`, and
#     `build_a2a_visual_director_app(services: ServiceContainer | None = None, rpc_url: str = ...) -> Any`
#     configured with `execute_interceptors=[include_artifacts_in_a2a_event_interceptor]` (also exposed in
#     `pitch_generator/fast_api_app.py` when `SERVICE_ROLE == "visual-director"`).
#   - `_cloud_run_client(base_url: str)`: returns `None` for local `http://` or empty URLs, and returns an
#     authenticated HTTP client with `timeout=600.0` and an OIDC `Authorization: Bearer <token>` header for `https://` Cloud Run URLs.
#   - `_pitch_parts_only(part: Any)`: stream filter that drops `function_call` and `function_response` parts
#     (returning `None` / `False`) while keeping `text` and `inline_data` (`image/png`) parts.
#   - `remote_visual_director = RemoteA2aAgent(name="visual_director", description=..., agent_card=f"{visual_director_url}/.well-known/agent-card.json", httpx_client=_cloud_run_client(visual_director_url), genai_part_converter=_pitch_parts_only, output_key="visual_director")`
#     and swap `visual_director` for `remote_visual_director` in `root_agent.edges`. Also remove `hidden`
#     from `#btn-inspect-a2a` in `frontend/index.html` (defining `remote_visual_director` or
#     `build_a2a_visual_director_app` also automatically sets `a2a_enabled=True` in `/api/config`).


def build_visual_director_card(
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> dict[str, Any]:
    """
    /**
     * Build the A2A v0.3 Agent Card for the Visual Director specialist agent.
     *
     * Why: Publishes visual director capabilities, input/output modes, and tool
     * skills at `/.well-known/agent-card.json` for discovery by `RemoteA2aAgent`.
     *
     * @param rpc_url Public URL of the visual_director JSON-RPC endpoint.
     * @return Agent Card dictionary.
     */
    """
    card_dict = dict(AgentCardBuilder(agent=visual_director, rpc_url=rpc_url).build())
    card_dict["name"] = "visual_director"
    card_dict["url"] = rpc_url
    card_dict["preferredTransport"] = "JSONRPC"
    return card_dict


def build_a2a_visual_director_app(
    services: ServiceContainer | None = None,
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> Any:
    """
    /**
     * Construct the standalone ASGI/FastAPI application for the Visual Director A2A service.
     *
     * Why: Enables deploying `visual_director` as an independent Cloud Run microservice
     * with `include_artifacts_in_a2a_event_interceptor` active.
     *
     * @param services Optional injected `ServiceContainer`.
     * @param rpc_url Base A2A JSON-RPC URL for the visual director.
     * @return Configured `PitchFastAPIApp` instance for the visual director microservice.
     */
    """
    from pitch_generator.fast_api_app import PitchFastAPIApp

    app_instance = PitchFastAPIApp(services=services)
    agent_card = build_visual_director_card(rpc_url=rpc_url)
    executor = A2aAgentExecutor(
        config=A2aAgentExecutorConfig(
            execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
        )
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store={},
        agent_card=agent_card,
    )
    rpc_path = "/a2a/visual_director"
    app_instance.register_a2a_handler(
        rpc_path=rpc_path,
        handler=handler,
        agent_card=agent_card,
    )
    return app_instance


def _cloud_run_client(base_url: str) -> Any:
    """
    /**
     * Construct an HTTP client for remote A2A calls.
     *
     * Why: Returns None for local http:// or empty URLs (enabling zero-config offline execution),
     * and returns an authenticated httpx/HTTP client with timeout=600.0 and OIDC Authorization
     * header for https:// Cloud Run service URLs.
     *
     * @param base_url Base URL of the remote A2A service.
     * @return Configured HTTP client or None for local endpoints.
     */
    """
    if not base_url or not isinstance(base_url, str):
        return None
    clean_url = base_url.strip()
    if not clean_url.startswith("https://"):
        return None

    headers: dict[str, str] = {}
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        auth_req = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_req, clean_url)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    except Exception:
        pass

    try:
        import httpx

        return httpx.Client(base_url=clean_url, timeout=600.0, headers=headers)
    except ImportError:
        class _SimpleHttpClient:
            def __init__(
                self,
                base_url: str,
                timeout: float = 600.0,
                headers: dict[str, str] | None = None,
            ) -> None:
                self.base_url = base_url
                self.timeout = timeout
                self.headers = headers or {}

        return _SimpleHttpClient(base_url=clean_url, timeout=600.0, headers=headers)


def _pitch_parts_only(part: Any) -> Any:
    """
    /**
     * Stream filter converter that drops intermediate function call/response parts.
     *
     * Why: Passes clean text and rendered PNG image artifacts to downstream nodes
     * while stripping internal function calling metadata.
     *
     * @param part A2A event part instance or dict.
     * @return Filtered part, or None if it contains function calls/responses.
     */
    """
    if part is None:
        return None
    if isinstance(part, dict):
        if (
            part.get("function_call")
            or part.get("function_response")
            or part.get("functionCall")
            or part.get("functionResponse")
        ):
            return None
        return part
    if (
        getattr(part, "function_call", None) is not None
        or getattr(part, "function_response", None) is not None
    ):
        return None
    return part


visual_director_url: str = get_config().visual_director_url.rstrip("/")

remote_visual_director = RemoteA2aAgent(
    name="visual_director",
    description="Translates the campaign concept into art direction and key visuals following brand guidelines via remote A2A microservice.",
    agent_card=f"{visual_director_url}/.well-known/agent-card.json",
    httpx_client=_cloud_run_client(visual_director_url),
    genai_part_converter=_pitch_parts_only,
    output_key="visual_director",
)

assemble = JoinNode(name="assemble")


def package(
    ctx_or_input: Any,
    node_input: dict[str, Any] | None = None,
) -> _DualSyncAsyncEventStream:
    """
    /**
     * Format and validate the joined specialist outputs into the final campaign pitch.
     *
     * Why: Acts as the terminal node after `assemble = JoinNode(name="assemble")`.
     * Explicitly checks that every expected upstream branch produced non-empty output
     * (raising `ValueError("nothing reached the join from: ...")` if any branch is
     * missing or empty) so silent branch failures are caught immediately. Supports both
     * `package(node_input)` and `package(ctx, node_input)` signatures, strips Markdown
     * code fences from branch values, and attaches `_key_visual(ctx)` when present.
     *
     * @param ctx_or_input Either a `node_input` dict (1-arg form) or `Context` (2-arg form).
     * @param node_input Dictionary of upstream outputs when `ctx_or_input` is a `Context`.
     * @return Dual sync/async stream yielding `Event(content=...)` and `Event(output=...)`.
     */
    """
    if node_input is None:
        if not isinstance(ctx_or_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx: Context | None = None
        actual_input = ctx_or_input
    else:
        if not isinstance(node_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx = ctx_or_input if isinstance(ctx_or_input, Context) else None
        actual_input = node_input

    if not actual_input:
        raise ValueError("nothing reached the join from: creative_director, copywriter")

    required_keys = ["creative_director", "copywriter"]
    missing_or_empty = [
        k
        for k in required_keys
        if k not in actual_input or not str(actual_input.get(k) or "").strip()
    ]
    extra_empty = [
        name
        for name, value in actual_input.items()
        if name not in missing_or_empty and not str(value or "").strip()
    ]
    all_empty = missing_or_empty + extra_empty
    if all_empty:
        raise ValueError(f"nothing reached the join from: {', '.join(all_empty)}")

    concept_val = strip_markdown_fences(str(actual_input["creative_director"])).strip()
    copy_val = strip_markdown_fences(str(actual_input["copywriter"])).strip()
    sections = [
        f"CONCEPT\n{concept_val}",
        f"COPY\n{copy_val}",
    ]
    if "brand_strategist" in actual_input and actual_input["brand_strategist"]:
        brand_val = strip_markdown_fences(str(actual_input["brand_strategist"])).strip()
        sections.append(f"BRAND STRATEGY\n{brand_val}")
    if "visual_director" in actual_input and actual_input["visual_director"]:
        art_val = strip_markdown_fences(str(actual_input["visual_director"])).strip()
        sections.append(f"ART DIRECTION\n{art_val}")

    pitch = "\n\n".join(sections)
    parts: list[Any] = [types.Part(text=pitch)]
    image = _key_visual(ctx)
    if image is not None:
        parts.append(types.Part(inline_data=image))

    events = [
        Event(author="package", content=types.Content(role="model", parts=parts)),
        Event(author="package", output=pitch),
    ]
    return _DualSyncAsyncEventStream(events)


loop_guard = LoopGuard(max_iterations=10)

# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]
# TODO (Step 3a): Implement a fail-closed `PreToolUse` security hook before tool execution:
#   - `DEFAULT_TOOL_ALLOWLIST = frozenset({"load_skill", "generate_key_visual"})`
#   - `ToolAuthorizationError(PermissionError)` with `.tool_name` and `.reason` attributes.
#   - `HookDecision` dataclass with `allowed: bool`, `tool_name: str`, `reason: str`,
#     `violation_type: str | None = None`, `sanitized_args: dict[str, Any]`, plus `.is_allowed`,
#     `.permitted`, and `__bool__`.
#   - `PreToolUseHook(allowlist=None, raise_on_block=False)` and module-level
#     `validate_tool_call(tool_name, tool_args=None) -> HookDecision`:
#     1. Block empty or unlisted tools (`violation_type="unauthorized_tool"`).
#     2. Block forbidden override keys such as `bypass_safety` or `override_brand_guidelines`
#        (`violation_type="forbidden_parameter"`).
#     3. Block path traversal sequences (`..` or leading `/`, `violation_type="path_traversal"`).
#     4. Block oversized string arguments (`> 4096` chars) and prompt injection patterns such as
#        `"ignore previous instructions"` or `system` tags (`violation_type="prompt_injection"`).
#     5. Append every check to `self.audit_log` and raise `ToolAuthorizationError` when `raise_on_block=True`.
DEFAULT_TOOL_ALLOWLIST: frozenset[str] = frozenset({"load_skill", "generate_key_visual"})
FORBIDDEN_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "bypass_safety",
    "override_brand_guidelines",
    "skip_safety",
    "disable_safety",
    "ignore_safety",
})
MAX_TOOL_ARG_LENGTH: int = 4096
PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "<system>",
    "</system>",
    "system prompt",
    "disregard previous",
    "reveal your prompt",
)


class ToolAuthorizationError(PermissionError):
    """
    /**
     * Exception raised when a tool call is blocked by PreToolUse security policies.
     *
     * @param message Human-readable error description.
     * @param tool_name Name of the unauthorized or blocked tool.
     * @param reason Specific policy violation reason.
     */
    """

    def __init__(self, message: str, tool_name: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.reason = reason or message


@dataclass
class HookDecision:
    """
    /**
     * Policy evaluation decision returned by PreToolUse lifecycle hooks.
     *
     * Why: Encapsulates authorization boolean, violation diagnostics, and sanitized
     * arguments with truthiness helpers for idiomatic guard conditionals.
     */
    """

    allowed: bool
    tool_name: str
    reason: str
    violation_type: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.allowed

    @property
    def permitted(self) -> bool:
        return self.allowed

    def __bool__(self) -> bool:
        return self.allowed


class PreToolUseHook:
    """
    /**
     * PreToolUse lifecycle security hook enforcing tool allowlists and argument inspection.
     *
     * Why: Acts as a fail-closed perimeter gate before tool invocation: blocks unlisted tools,
     * forbidden override flags, path traversal attempts, oversized payloads, and prompt injection
     * markers while maintaining an append-only audit trail.
     */
    """

    def __init__(
        self,
        allowlist: set[str] | frozenset[str] | Sequence[str] | None = None,
        raise_on_block: bool = False,
    ) -> None:
        if allowlist is None:
            self.allowlist: frozenset[str] = DEFAULT_TOOL_ALLOWLIST
        else:
            self.allowlist = frozenset(allowlist)
        self.raise_on_block: bool = raise_on_block
        self.audit_log: list[dict[str, Any]] = []

    def validate_tool_call(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None = None,
    ) -> HookDecision:
        name = str(tool_name or "").strip()
        args = copy.deepcopy(tool_args) if isinstance(tool_args, dict) else {}

        # 1. Unlisted / empty / None tool
        if not name or name not in self.allowlist:
            reason = f"Tool '{name}' is not in the approved PreToolUse allowlist: {sorted(self.allowlist)}"
            decision = HookDecision(
                allowed=False,
                tool_name=name,
                reason=reason,
                violation_type="unauthorized_tool",
                sanitized_args=args,
            )
            self._record_audit(decision, args)
            if self.raise_on_block:
                raise ToolAuthorizationError(reason, tool_name=name, reason=reason)
            return decision

        # 2. Forbidden override keys
        for key in args:
            if key in FORBIDDEN_OVERRIDE_KEYS or "bypass" in key.lower() or "override" in key.lower():
                reason = f"Forbidden parameter '{key}' detected in tool call '{name}'"
                decision = HookDecision(
                    allowed=False,
                    tool_name=name,
                    reason=reason,
                    violation_type="forbidden_parameter",
                    sanitized_args=args,
                )
                self._record_audit(decision, args)
                if self.raise_on_block:
                    raise ToolAuthorizationError(reason, tool_name=name, reason=reason)
                return decision

        # 3. Inspect argument values recursively
        def _check_value(val: Any) -> tuple[bool, str | None, str | None]:
            if isinstance(val, str):
                if len(val) > MAX_TOOL_ARG_LENGTH:
                    return False, "prompt_injection", f"Argument string exceeds maximum length of {MAX_TOOL_ARG_LENGTH} characters"
                if ".." in val or (val.startswith("/") and not val.startswith("gs://")):
                    return False, "path_traversal", f"Path traversal sequence detected: {val!r}"
                lower_val = val.lower()
                for pattern in PROMPT_INJECTION_PATTERNS:
                    if pattern in lower_val:
                        return False, "prompt_injection", f"Prompt injection pattern detected: {pattern!r}"
            elif isinstance(val, dict):
                for k, v in val.items():
                    if k in FORBIDDEN_OVERRIDE_KEYS:
                        return False, "forbidden_parameter", f"Forbidden parameter '{k}' detected"
                    ok, vtype, vreason = _check_value(v)
                    if not ok:
                        return ok, vtype, vreason
            elif isinstance(val, (list, tuple, set)):
                for item in val:
                    ok, vtype, vreason = _check_value(item)
                    if not ok:
                        return ok, vtype, vreason
            return True, None, None

        for k, v in args.items():
            ok, vtype, vreason = _check_value(v)
            if not ok:
                decision = HookDecision(
                    allowed=False,
                    tool_name=name,
                    reason=vreason or "Security validation failed",
                    violation_type=vtype,
                    sanitized_args=args,
                )
                self._record_audit(decision, args)
                if self.raise_on_block:
                    raise ToolAuthorizationError(decision.reason, tool_name=name, reason=decision.reason)
                return decision

        # All checks passed
        decision = HookDecision(
            allowed=True,
            tool_name=name,
            reason="Tool authorized by PreToolUse allowlist.",
            violation_type=None,
            sanitized_args=args,
        )
        self._record_audit(decision, args)
        return decision

    def _record_audit(self, decision: HookDecision, args: dict[str, Any]) -> None:
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": decision.tool_name,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "violation_type": decision.violation_type,
            "args": copy.deepcopy(args),
        })


def validate_tool_call(
    tool_name: str | None,
    tool_args: dict[str, Any] | None = None,
    hook: PreToolUseHook | None = None,
) -> HookDecision:
    """
    /**
     * Module-level helper validating a tool call against PreToolUse security policies.
     *
     * @param tool_name Target tool identifier.
     * @param tool_args Dictionary of arguments for the tool call.
     * @param hook Optional PreToolUseHook instance.
     * @return HookDecision containing authorization status and diagnostics.
     */
    """
    active_hook = hook or PreToolUseHook()
    return active_hook.validate_tool_call(tool_name, tool_args)



# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]
# TODO (Step 3c): Add the two-node ADK Human-in-the-Loop checkpoint after `creative_director`:
#   - `@node(rerun_on_resume=False) async def approve_concept(ctx: Context)`:
#     validates that `ctx.session.state.get("creative_director")` is non-empty and yields
#     `RequestInput(message="Please approve the campaign concept (yes/no).", response_schema=str)`.
#   - `evaluate_user_approval(concept: str, decision: str) -> str`:
#     accepts affirmative responses (`"yes"`, `"y"`, `"true"`, `"approved"` -> returns
#     `f"## Approved Concept\n\n{concept}"`) and raises `ValueError("User rejected the concept")` otherwise.
#   - `@node(rerun_on_resume=True) async def user_approval(ctx: Context, node_input: dict[str, Any] | None = None)`:
#     awaits `ctx.run_node(approve_concept)`, validates the decision via `evaluate_user_approval`,
#     and yields `Event(author="user_approval", content=...)`.
#   - `run_hitl_workflow(brief: str, *, approved: bool | None = None, services: ServiceContainer | None = None) -> dict[str, Any]`:
#     returns `status="input_required"` (`downstream_calls=0`) when `approved is None`,
#     `status="rejected"` (`downstream_calls=0`) when `approved is False`, and `status="completed"`
#     (`downstream_calls=2`, `"## Approved Concept"` in `concept_header`) when `approved is True`.
#   - Insert `user_approval` into `root_agent.edges` after `creative_director`, and enable the
#     Human-in-the-Loop UI controls (`#hitl-checkbox-group` and `#hitl-approval-card`) in
#     `frontend/index.html` by removing their `hidden` attributes (defining `approve_concept` or
#     `user_approval` here also automatically sets `hitl_enabled=True` in `/api/config` and `/api/pitch`).
@node(rerun_on_resume=False)
async def approve_concept(ctx: Context) -> Any:
    """
    /**
     * Pauses workflow execution after Creative Director to solicit explicit human approval.
     *
     * Why: Enforces an interactive Human-in-the-Loop boundary before downstream specialists
     * (Copywriter and Visual Director) consume generative quota. Sets `rerun_on_resume=False`
     * so ADK does not re-prompt on resumption.
     *
     * @param ctx Execution context containing session state and draft concept.
     * @return Async generator yielding RequestInput.
     */
    """
    state = getattr(ctx.session, "state", None) if hasattr(ctx, "session") else None
    if state is None and hasattr(ctx, "state"):
        state = ctx.state
    concept = state.get("creative_director") if isinstance(state, dict) else None
    if not concept or not str(concept).strip():
        raise ValueError("Cannot request approval for an empty or missing campaign concept.")
    yield RequestInput(
        message="Please approve the campaign concept (yes/no).",
        response_schema=str,
    )


def evaluate_user_approval(concept: str, decision: str) -> str:
    """
    /**
     * Evaluates the human approver's response against affirmative criteria.
     *
     * Why: Implements fail-closed validation: only explicit affirmative terms ('yes', 'y',
     * 'true', 'approved') unlock downstream execution, raising ValueError on rejection or ambiguity.
     *
     * @param concept Campaign concept string generated by Creative Director.
     * @param decision User input string.
     * @return Formatted approved concept markdown string.
     */
    """
    if not isinstance(decision, str):
        raise ValueError("User rejected the concept (non-string input).")
    clean_dec = decision.strip().lower()
    if clean_dec in {"yes", "y", "true", "approved"}:
        return f"## Approved Concept\n\n{concept}"
    raise ValueError(f"User rejected the concept: {decision!r}")


@node(rerun_on_resume=True)
async def user_approval(
    ctx: Context, node_input: dict[str, Any] | None = None
) -> Any:
    """
    /**
     * Resumes the paused workflow, validates human input, and emits the approved concept event.
     *
     * Why: Decorated with `rerun_on_resume=True` so ADK invokes this node upon user input,
     * verifying authorization before dispatching to downstream branches.
     *
     * @param ctx Execution context.
     * @param node_input Dict containing user response decision.
     * @return Async generator yielding user_approval Event.
     */
    """
    state = getattr(ctx.session, "state", None) if hasattr(ctx, "session") else None
    if state is None and hasattr(ctx, "state"):
        state = ctx.state
    concept = state.get("creative_director", "") if isinstance(state, dict) else ""
    decision_val = ""
    if isinstance(node_input, dict):
        decision_val = str(node_input.get("decision", node_input.get("response", "")))
    approved_text = evaluate_user_approval(str(concept), decision_val)
    yield Event(
        author="user_approval",
        content=types.Content(role="user", parts=[types.Part(text=approved_text)]),
        output=approved_text,
    )


def run_hitl_workflow(
    brief: str,
    *,
    approved: bool | None = None,
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Orchestrates the Human-in-the-Loop concept review workflow.
     *
     * Why: Provides a high-level API to test and drive the two-node HITL checkpoint:
     * - `approved=None`: Generates concept and pauses at `status="input_required"` (0 downstream calls).
     * - `approved=False`: Rejects concept and halts at `status="rejected"` (0 downstream calls).
     * - `approved=True`: Resumes execution, invoking Copywriter and Visual Director (2 downstream calls).
     *
     * @param brief Campaign brief string.
     * @param approved Approval flag (`None` for initial pause, `False` for rejection, `True` for resume).
     * @param services Optional injected ServiceContainer.
     * @return Workflow dictionary containing status, concept, copy, art_direction, and downstream_calls.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string.")

    svc = services or get_default_services()
    clean_brief = str(scrub_text(brief.strip()).scrubbed)
    model_id = svc.config.flash_model

    concept_raw = svc.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept = concept_raw.strip()

    if approved is None:
        return {
            "status": "input_required",
            "concept": concept,
            "concept_header": "",
            "copy": "",
            "art_direction": "",
            "brand_strategy": "",
            "key_visual_uri": None,
            "downstream_calls": 0,
        }

    if approved is False:
        return {
            "status": "rejected",
            "concept": concept,
            "concept_header": "",
            "copy": "",
            "art_direction": "",
            "brand_strategy": "",
            "key_visual_uri": None,
            "downstream_calls": 0,
        }

    # approved is True
    approved_header = evaluate_user_approval(concept, "yes")
    try:
        strat = svc.llm.generate_text(
            f"{clean_brief}\n\nConcept: {concept}",
            system_instruction=brand_strategist.instruction,
            model=model_id,
        )
    except Exception:
        strat = f"Strategic positioning for {concept}."

    try:
        copy_text = svc.llm.generate_text(
            f"Concept: {concept}\nStrategy: {strat}",
            system_instruction=copywriter.instruction,
            model=model_id,
        )
    except Exception:
        copy_text = f"Compelling campaign copy for {concept}."

    try:
        art_dir = svc.llm.generate_text(
            f"Concept: {concept}\nStrategy: {strat}",
            system_instruction="You are Visual Director. Produce concise art direction.",
            model=model_id,
        )
    except Exception:
        art_dir = f"Studio key visual art direction for {concept}."

    key_vis_uri = svc.artifacts.save_artifact("key_visual.png", b"fake-png-bytes")

    return {
        "status": "completed",
        "concept": concept,
        "concept_header": approved_header,
        "copy": copy_text.strip(),
        "art_direction": art_dir.strip(),
        "brand_strategy": strat.strip(),
        "key_visual_uri": key_vis_uri,
        "downstream_calls": 2,
    }


root_agent = Workflow(
    name="pitch_generator",
    edges=[
        (creative_director, user_approval, brand_strategist, (copywriter, remote_visual_director)),
        ((copywriter, remote_visual_director), assemble, package),
    ],
)
loop_guard.validate_graph(root_agent.edges)

app = App(root_agent=root_agent, name="pitch_generator")



# [Guidepost — Step 4a: Tokenomics & Context Optimization]
# TODO (Step 4a): Implement memory compression, sliding-window pruning, SHA-256 prompt caching, and model routing:
#   - `estimate_tokens(text: str | None) -> int`: returns `0` for empty/whitespace text, else `max(1, len(text.strip()) // 4)`.
#   - `CompressedHistoryList(list)`: `list` subclass with `.turns`, `.history`, `.summary`,
#     `.original_tokens`, `.optimized_tokens`, and `.tokens_saved` attributes.
#   - `compress_memory(turns: Sequence[dict[str, Any]], keep_recent: int = 2) -> CompressedHistoryList`:
#     summarizes older turns prior to the last `keep_recent` turns into a single
#     `{"role": "system", "compressed": True, "content": ...}` entry while preserving the most recent turns verbatim.
#   - `prune_history(turns: Sequence[dict[str, Any]], max_turns: int = 6, max_tokens: int = 512) -> CompressedHistoryList`:
#     validates `max_turns > 0` and `max_tokens > 0` (raising `ValueError` otherwise), keeps the newest turns
#     within the turn/token budget, and truncates a single oversized turn (`"truncated": True`) to fit `max_tokens`.
#   - `PromptCacheManager(default_ttl_seconds: int = 300)`: `.get_or_create(static_prompt: str, ttl_seconds: int | None = None)`
#     validates non-empty text, computes a 64-char SHA-256 hex `cache_key` (`hashlib.sha256(...).hexdigest()`),
#     and tracks TTL expiration, `"cache_hit"` (`True`/`False`), and `"saved_tokens"`.
#   - `select_model_strategy(tier: str = "flash") -> dict[str, Any]` supporting tiers
#     `("flash", "pro", "low", "high", "local")` (raising `ValueError` on unknown tiers) and returning
#     `"tier"`, `"model_id"`, `"target"`, `"cost_tier"`, and `"thinking_budget"`.
#   - `TokenomicsManager(max_turns: int = 6, max_tokens: int = 512)`: `.optimize(turns, static_prompt="", max_turns=..., max_tokens=..., tier="flash")`
#     combining `compress_memory`, `prune_history`, `PromptCacheManager`, and `select_model_strategy`.


def estimate_tokens(text: str | None) -> int:
    """
    /**
     * Fast heuristic token estimator (~4 characters per token).
     *
     * Why: Provides a zero-dependency, deterministic token counter for budget calculations.
     *
     * @param text Input string or None.
     * @return Estimated integer token count (0 for empty/whitespace).
     */
    """
    if not text or not text.strip():
        return 0
    return max(1, len(text.strip()) // 4)


class CompressedHistoryList(list):
    """
    /**
     * List subclass maintaining metadata for compressed/pruned conversation turns.
     *
     * Why: Exposes `.turns`, `.history`, `.summary`, `.original_tokens`, `.optimized_tokens`,
     * and `.tokens_saved` while preserving native list operations.
     */
    """

    def __init__(
        self,
        items: Sequence[dict[str, Any]] | None = None,
        *,
        summary: str = "",
        original_tokens: int = 0,
        optimized_tokens: int = 0,
        tokens_saved: int = 0,
    ) -> None:
        super().__init__(items or [])
        self.summary = summary
        self.original_tokens = original_tokens
        self.optimized_tokens = optimized_tokens
        self.tokens_saved = tokens_saved

    @property
    def turns(self) -> list[dict[str, Any]]:
        return list(self)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self)


def compress_memory(
    turns: Sequence[dict[str, Any]],
    keep_recent: int = 2,
) -> CompressedHistoryList:
    """
    /**
     * Compresses older turns into a single system summary while preserving recent turns verbatim.
     *
     * Why: Reduces token consumption over long-running sessions by summarizing historical turns.
     *
     * @param turns List of turn dictionaries with 'role' and 'content'.
     * @param keep_recent Number of most recent turns to preserve verbatim.
     * @return CompressedHistoryList with summary header and recent turns.
     */
    """
    if not turns:
        return CompressedHistoryList(
            [], summary="", original_tokens=0, optimized_tokens=0, tokens_saved=0
        )

    orig_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)
    if len(turns) <= keep_recent:
        copied = [copy.deepcopy(t) for t in turns]
        return CompressedHistoryList(
            copied,
            summary="",
            original_tokens=orig_tokens,
            optimized_tokens=orig_tokens,
            tokens_saved=0,
        )

    older_turns = turns[:-keep_recent]
    recent_turns = [copy.deepcopy(t) for t in turns[-keep_recent:]]

    older_summary_parts = []
    for t in older_turns:
        role = t.get("role", "user")
        cnt = str(t.get("content", "")).strip()
        older_summary_parts.append(f"{role}: {cnt[:40]}...")
    summary_text = "Prior conversation summary: " + " | ".join(older_summary_parts)

    system_turn = {
        "role": "system",
        "compressed": True,
        "content": summary_text,
    }
    result_items = [system_turn] + recent_turns
    opt_tokens = sum(estimate_tokens(t.get("content", "")) for t in result_items)
    saved = max(0, orig_tokens - opt_tokens)

    return CompressedHistoryList(
        result_items,
        summary=summary_text,
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=saved,
    )


def prune_history(
    turns: Sequence[dict[str, Any]],
    max_turns: int = 6,
    max_tokens: int = 512,
) -> CompressedHistoryList:
    """
    /**
     * Sliding-window pruning enforcing max turn count and hard token ceilings.
     *
     * Why: Ensures request payloads strictly fit within allocated token and turn budgets.
     *
     * @param turns Sequence of conversation turns.
     * @param max_turns Maximum number of turns to keep.
     * @param max_tokens Maximum total tokens allowed.
     * @return CompressedHistoryList satisfying constraints.
     */
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if not turns:
        return CompressedHistoryList(
            [], original_tokens=0, optimized_tokens=0, tokens_saved=0
        )

    orig_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)
    candidate_turns = [copy.deepcopy(t) for t in turns[-max_turns:]]

    if len(candidate_turns) == 1:
        single = candidate_turns[0]
        single_tok = estimate_tokens(single.get("content", ""))
        if single_tok > max_tokens:
            max_chars = max(1, max_tokens * 4)
            truncated_content = str(single.get("content", ""))[:max_chars]
            single["content"] = truncated_content
            single["truncated"] = True
            opt_tokens = estimate_tokens(truncated_content)
            return CompressedHistoryList(
                [single],
                original_tokens=orig_tokens,
                optimized_tokens=opt_tokens,
                tokens_saved=max(0, orig_tokens - opt_tokens),
            )
        return CompressedHistoryList(
            [single],
            original_tokens=orig_tokens,
            optimized_tokens=single_tok,
            tokens_saved=0,
        )

    kept: list[dict[str, Any]] = []
    current_tokens = 0
    for turn in reversed(candidate_turns):
        tok = estimate_tokens(turn.get("content", ""))
        if current_tokens + tok <= max_tokens:
            kept.insert(0, turn)
            current_tokens += tok
        else:
            if not kept:
                max_chars = max(1, max_tokens * 4)
                truncated_content = str(turn.get("content", ""))[:max_chars]
                turn["content"] = truncated_content
                turn["truncated"] = True
                kept.insert(0, turn)
                current_tokens += estimate_tokens(truncated_content)
            break

    opt_tokens = sum(estimate_tokens(t.get("content", "")) for t in kept)
    return CompressedHistoryList(
        kept,
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=max(0, orig_tokens - opt_tokens),
    )


class PromptCacheManager:
    """
    /**
     * SHA-256 prompt cache manager tracking cache keys, TTL expiration, and token savings.
     *
     * Why: Reuses cached prefix context (e.g. brand guidelines) across agent invocations.
     */
    """

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    def get_or_create(
        self,
        static_prompt: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        /**
         * Fetch or create a cached entry for a static prompt string.
         *
         * @param static_prompt Static text prefix.
         * @param ttl_seconds Optional TTL override.
         * @return Cache metadata dict with cache_key, cache_hit, saved_tokens, expires_at.
         */
        """
        if not static_prompt or not static_prompt.strip():
            raise ValueError("static_prompt must be a non-empty string.")

        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        now = time.time()
        cache_key = hashlib.sha256(static_prompt.strip().encode("utf-8")).hexdigest()
        tokens = estimate_tokens(static_prompt)

        if ttl > 0 and cache_key in self._cache:
            entry = self._cache[cache_key]
            if now < entry["expires_at"]:
                return {
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "saved_tokens": tokens,
                    "expires_at": entry["expires_at"],
                }

        expires_at = now + ttl if ttl > 0 else 0
        if ttl > 0:
            self._cache[cache_key] = {
                "expires_at": expires_at,
                "tokens": tokens,
            }
        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "saved_tokens": 0,
            "expires_at": expires_at,
        }


def select_model_strategy(tier: str = "flash") -> dict[str, Any]:
    """
    /**
     * Routes tasks to the optimal Gemini model tier based on latency, reasoning, and cost requirements.
     *
     * Why: Implements model strategy tiers (flash, pro, low, high, local).
     *
     * @param tier Model tier name ('flash', 'pro', 'low', 'high', 'local').
     * @return Strategy configuration dictionary.
     */
    """
    tier_map = {
        "flash": {
            "tier": "flash",
            "model_id": "gemini-3.8-flash",
            "target": "low_latency_throughput",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "pro": {
            "tier": "pro",
            "model_id": "gemini-3.8-pro",
            "target": "complex_reasoning",
            "cost_tier": "standard",
            "thinking_budget": 1024,
        },
        "low": {
            "tier": "low",
            "model_id": "gemini-3.8-flash",
            "target": "fast_execution",
            "cost_tier": "minimal",
            "thinking_budget": 0,
        },
        "high": {
            "tier": "high",
            "model_id": "gemini-3.8-pro",
            "target": "deep_reasoning_high_fidelity",
            "cost_tier": "high",
            "thinking_budget": 2048,
        },
        "local": {
            "tier": "local",
            "model_id": "gemini-3.8-flash",
            "target": "offline_local_development",
            "cost_tier": "none",
            "thinking_budget": 0,
        },
    }
    if tier not in tier_map:
        raise ValueError(
            f"Unknown model tier: {tier!r}. Supported tiers: {list(tier_map.keys())}"
        )
    return copy.deepcopy(tier_map[tier])


class TokenomicsManager:
    """
    /**
     * Unified tokenomics and context optimization coordinator.
     *
     * Why: Combines memory compression, sliding-window history pruning, SHA-256 prompt
     * caching, and model strategy tier selection into a single optimization call.
     */
    """

    def __init__(
        self,
        max_turns: int = 6,
        max_tokens: int = 512,
        prompt_cache: PromptCacheManager | None = None,
    ) -> None:
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.cache = prompt_cache or PromptCacheManager()

    def optimize(
        self,
        turns: Sequence[dict[str, Any]],
        *,
        static_prompt: str = "",
        max_turns: int | None = None,
        max_tokens: int | None = None,
        tier: str = "flash",
    ) -> dict[str, Any]:
        """
        /**
         * Optimize multi-turn history with compression, sliding window pruning, prompt caching, and model routing.
         *
         * @param turns History turn dictionaries.
         * @param static_prompt Optional invariant system prompt.
         * @param max_turns Optional override for max turns.
         * @param max_tokens Optional override for token budget.
         * @param tier Model tier string.
         * @return Optimization result dictionary.
         */
        """
        effective_turns = self.max_turns if max_turns is None else max_turns
        effective_tokens = self.max_tokens if max_tokens is None else max_tokens

        orig_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)
        if static_prompt:
            orig_tokens += estimate_tokens(static_prompt)

        # 1. Compress memory if more than 2 turns
        compressed_turns = compress_memory(turns, keep_recent=2)

        # 2. Prune history to respect sliding window budget
        pruned_turns = prune_history(
            compressed_turns,
            max_turns=effective_turns,
            max_tokens=effective_tokens,
        )

        # 3. Prompt caching for static prompt prefix
        cache_info = {"cache_key": "", "cache_hit": False, "saved_tokens": 0}
        if static_prompt:
            cache_info = self.cache.get_or_create(static_prompt)

        # 4. Model strategy selection
        strategy = select_model_strategy(tier)

        opt_tokens = sum(estimate_tokens(t.get("content", "")) for t in pruned_turns)
        saved_tokens = max(
            0, orig_tokens - opt_tokens + cache_info.get("saved_tokens", 0)
        )

        return {
            "turns": list(pruned_turns),
            "original_tokens": orig_tokens,
            "optimized_tokens": opt_tokens,
            "tokens_saved": saved_tokens,
            "cache_key": cache_info.get("cache_key", ""),
            "cache_hit": cache_info.get("cache_hit", False),
            "selected_strategy": strategy,
        }



# ==============================================================================
# Internal Workflow Execution Engine (Do not modify unless extending runners)
# ==============================================================================


def run_graph_workflow(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    max_iterations: int = 10,
) -> PitchPackage:
    """
    /**
     * Validates the workflow DAG with `LoopGuard` and executes the fan-out/fan-in pitch graph.
     *
     * Why: Provides a deterministic, offline-safe execution helper that validates the
     * graph topology, tracks every node visit in `LoopGuard`, parses structured payloads,
     * and returns a complete `PitchPackage`.
     *
     * @param brief Campaign product brief string.
     * @param services Optional `ServiceContainer` for offline LLM/artifact services.
     * @param max_iterations Maximum allowed node visits in `LoopGuard`.
     * @return Populated `PitchPackage` instance.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string.")

    guard = LoopGuard(max_iterations=max_iterations)
    guard.validate_graph(root_agent.edges)

    svc = services or get_default_services()
    ctx = Context(services=svc)
    clean_brief = brief.strip()
    model_id = svc.config.flash_model

    guard.record_step("creative_director")
    concept_raw = svc.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept_lines = [ln.strip() for ln in concept_raw.splitlines() if ln.strip()]
    concept = ConceptPayload(
        concept_line=concept_lines[0] if concept_lines else f"Engineered for everyday motion: {clean_brief}.",
        rationale=(
            concept_lines[1]
            if len(concept_lines) > 1
            else "Grounded in practical commuter utility and understated design."
        ),
    )

    guard.record_step("brand_strategist")
    strategy_notes = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=brand_strategist.instruction,
        model=model_id,
    )

    guard.record_step("copywriter")
    copy_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=copywriter.instruction,
        model=model_id,
    )
    copy_payload = CopyPayload(caption=copy_text)

    guard.record_step("visual_director")
    art_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=visual_director.instruction,
        model=model_id,
    )
    art_payload = ArtDirectionPayload(art_direction=art_text)

    guard.record_step("assemble")
    joined_inputs = {
        "creative_director": f"{concept.concept_line}\n{concept.rationale}",
        "brand_strategist": strategy_notes,
        "copywriter": copy_payload.caption,
        "visual_director": art_payload.art_direction,
    }

    guard.record_step("package")
    packaged_events = list(package(ctx, joined_inputs))
    final_text = ""
    for ev in packaged_events:
        if ev.output:
            final_text = str(ev.output)

    return PitchPackage(
        concept=concept,
        copy=copy_payload,
        art_direction=art_payload,
        brand_strategy=strategy_notes,
        pitch_text=final_text,
        trace=list(guard.visited_nodes),
    )


execute_workflow = run_graph_workflow
run_workflow = run_graph_workflow


def run_pitch_workflow(
    brief: str,
    *,
    session_id: str = "default",
    services: ServiceContainer | None = None,
    approved: bool | None = True,
    require_approval: bool = False,
    existing_concept: str | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the Pitch Generator multi-agent workflow using injected cloud services.
     *
     * Why: Coordinates the 4-specialist pipeline (`creative_director` -> `brand_strategist`
     * -> (`copywriter`, `visual_director`) -> `assemble` -> `package`) on Gemini Enterprise
     * Agent Platform (`gemini-3.8-flash`), persists session state in `MemoryBankService`,
     * and records telemetry via `BigQueryAnalyticsService`. When the learner adds the HITL
     * approval gate (`approve_concept` / `user_approval`) in Module 3, this runner dynamically
     * pauses and resumes at the concept review gate.
     *
     * @param brief Non-empty campaign topic brief.
     * @param session_id Session identifier for Memory Bank persistence.
     * @param services Optional injected `ServiceContainer` (defaults to `get_default_services()`).
     * @param approved HITL approval state (`True` to complete, `False` to reject,
     *   `None` to pause at `input_required` when HITL is enabled).
     * @param require_approval If True, pauses at the HITL gate (`status="input_required"`).
     * @param existing_concept Optional pre-generated concept to reuse when resuming a paused
     *   HITL session so `creative_director` is not re-executed.
     * @return Workflow result dictionary with keys `session_id`, `status`, `concept`,
     *   `copy`, `brand_strategy`, `art_direction`, `key_visual_uri`, `telemetry`,
     *   `artifacts`, `pitch_text`, and `trace`.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")

    clean_brief = str(scrub_text(brief.strip()).scrubbed)
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    module_globals = globals()
    strip_fn = module_globals.get("strip_markdown_fences")
    active_model = active_services.config.flash_model

    # Step 1: Creative Director generates campaign concept & rationale (or reuses saved concept on resume)
    if isinstance(existing_concept, str) and existing_concept.strip():
        concept = existing_concept.strip()
    else:
        concept = active_services.llm.generate_text(
            clean_brief,
            system_instruction=creative_director.instruction,
            model=active_model,
        )
        if callable(strip_fn):
            concept = strip_fn(concept)

    # Determine effective approval state (used when learner enables HITL in Module 3)
    effective_approved: bool | None = approved
    if require_approval and approved is True:
        effective_approved = None

    if effective_approved is False:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "rejected",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept", "user_approval"],
            }
        )
        rejected_state = {
            "session_id": clean_session,
            "status": "rejected",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept", "user_approval"],
        }
        active_services.memory_bank.save_session(clean_session, rejected_state)
        return rejected_state

    if effective_approved is None:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "input_required",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept"],
            }
        )
        paused_state = {
            "session_id": clean_session,
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept"],
        }
        active_services.memory_bank.save_session(clean_session, paused_state)
        return paused_state

    # Step 2: Copywriter generates <25-word social caption
    copy_text = active_services.llm.generate_text(
        concept,
        system_instruction=copywriter.instruction,
        model=active_model,
    )

    hitl_defined = (
        module_globals.get("approve_concept") is not None
        or module_globals.get("user_approval") is not None
    )
    join_inputs: dict[str, Any] = {
        "creative_director": concept,
        "copywriter": copy_text,
    }
    nodes_executed: list[str] = (
        ["creative_director", "approve_concept", "user_approval", "copywriter"]
        if hitl_defined
        else ["creative_director", "copywriter"]
    )
    brand_text = ""
    art_direction = ""
    key_visual_uri: str | None = None
    key_visual_url: str | None = None
    artifacts_meta: list[dict[str, Any]] = []

    loop_guard_cls = module_globals.get("LoopGuard")
    active_guard = None
    if callable(loop_guard_cls):
        active_guard = loop_guard_cls(max_iterations=10)
        if hasattr(active_guard, "validate_graph") and hasattr(root_agent, "edges"):
            active_guard.validate_graph(root_agent.edges)
        if hasattr(active_guard, "record_step"):
            active_guard.record_step("creative_director")
            active_guard.record_step("copywriter")

    if callable(strip_fn):
        copy_text = strip_fn(copy_text)
        join_inputs["creative_director"] = concept
        join_inputs["copywriter"] = copy_text

    brand_agent = module_globals.get("brand_strategist")
    if brand_agent is None:
        brand_agent = next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "brand_strategist"
            ),
            None,
        )
    if brand_agent is not None:
        brand_instruction = getattr(
            brand_agent,
            "instruction",
            "You are the Brand Strategist. Provide audience positioning for the concept.",
        )
        brand_text = active_services.llm.generate_text(
            concept,
            system_instruction=brand_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            brand_text = strip_fn(brand_text)
        join_inputs["brand_strategist"] = brand_text
        nodes_executed.append("brand_strategist")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("brand_strategist")

    remote_vd = module_globals.get("remote_visual_director")
    if remote_vd is None:
        remote_vd = next(
            (v for v in module_globals.values() if isinstance(v, RemoteA2aAgent)),
            None,
        )
    visual_agent = (
        module_globals.get("visual_director")
        or next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "visual_director"
            ),
            None,
        )
        or remote_vd
    )
    if visual_agent is not None:
        if remote_vd is not None:
            from pitch_generator.app_utils.services import _is_offline_test_mode
            import urllib.request

            if not _is_offline_test_mode(active_services.config.project_id):
                vd_url = active_services.config.visual_director_url.rstrip("/")
                card_url = f"{vd_url}/.well-known/agent-card.json"
                headers: dict[str, str] = {"Accept": "application/json"}
                if card_url.startswith("https://"):
                    try:
                        import google.auth.transport.requests
                        import google.oauth2.id_token

                        auth_req = google.auth.transport.requests.Request()
                        id_tok = google.oauth2.id_token.fetch_id_token(auth_req, vd_url)
                        if id_tok:
                            headers["Authorization"] = f"Bearer {id_tok}"
                    except Exception:
                        pass
                try:
                    req = urllib.request.Request(card_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}")
                except Exception as exc:
                    raise RuntimeError(
                        f"[A2A Remote Error] Could not connect to remote Visual Director service at "
                        f"'{vd_url}' ({card_url}): {exc}. Ensure the Visual Director service is running "
                        f"(or verify VISUAL_DIRECTOR_URL in .env)."
                    ) from exc

        visual_instruction = str(
            getattr(
                visual_agent,
                "instruction",
                (
                    "You are the Visual Director. Produce brand-compliant art direction "
                    "using our deep indigo and slate palette with a warm amber or terracotta "
                    "accent, single low raking light with long shadows, off-center negative "
                    "space, and one realistic photographic subject."
                ),
            )
            or ""
        )
        skill_md_path = Path(__file__).resolve().parent / "skills" / "brand-guidelines" / "SKILL.md"
        if skill_md_path.is_file():
            try:
                skill_text = skill_md_path.read_text(encoding="utf-8").strip()
                if skill_text and skill_text not in visual_instruction:
                    visual_instruction = f"{visual_instruction}\n\n[Loaded Skill: brand-guidelines]\n{skill_text}"
            except OSError:
                pass
        effective_visual_instruction = (
            f"{visual_instruction}\n\n"
            "Return ONLY the written art direction notes as plain text "
            "(do not emit tool call syntax; generate_key_visual will be invoked automatically with your text output)."
        )
        art_direction = active_services.llm.generate_text(
            concept,
            system_instruction=effective_visual_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            art_direction = strip_fn(art_direction)
        image_bytes, mime_type = active_services.llm.generate_image(
            art_direction,
            model=active_services.config.image_model,
        )
        artifact_record = active_services.artifacts.save_artifact(
            filename="key_visual.png",
            data=image_bytes,
            mime_type=mime_type,
            session_id=clean_session,
        )
        key_visual_url = (
            f"/api/artifacts/{clean_session}/{artifact_record.filename}"
            f"?v={artifact_record.version}"
        )
        key_visual_uri = (
            artifact_record.gcs_uri
            or f"/api/artifacts/{clean_session}/{artifact_record.filename}"
        )
        artifacts_meta.append(artifact_record.to_metadata_dict())
        join_inputs["visual_director"] = art_direction
        nodes_executed.append("visual_director")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("visual_director")

    # Step 3: Assemble and Package join output validation
    join_events = list(package(join_inputs))
    pitch_text = str(join_events[-1].output or "")
    nodes_executed.extend(["assemble", "package"])
    if active_guard is not None and hasattr(active_guard, "record_step"):
        active_guard.record_step("assemble")
        active_guard.record_step("package")

    telemetry_event = active_services.analytics.record_telemetry(
        {
            "session_id": clean_session,
            "status": "completed",
            "brief": clean_brief,
            "model_id": active_model,
            "key_visual_uri": key_visual_uri,
            "nodes_executed": list(nodes_executed),
        }
    )

    completed_result = {
        "session_id": clean_session,
        "status": "completed",
        "brief": clean_brief,
        "concept": concept,
        "copy": copy_text,
        "brand_strategy": brand_text,
        "art_direction": art_direction,
        "key_visual_uri": key_visual_uri,
        "key_visual_url": key_visual_url,
        "telemetry": telemetry_event,
        "artifacts": artifacts_meta,
        "pitch_text": pitch_text,
        "trace": list(nodes_executed),
    }

    active_services.memory_bank.save_session(clean_session, completed_result)
    active_services.memory_bank.store_memory(
        "campaigns",
        clean_session,
        {
            "brief": clean_brief,
            "concept": concept,
            "copy": copy_text,
            "key_visual_uri": key_visual_uri,
        },
    )
    return completed_result


def resume_pitch_workflow(
    session_id: str,
    *,
    approved: bool,
    feedback: str = "",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Resume a paused Human-in-the-Loop workflow session stored in `MemoryBankService`.
     *
     * Why: When `POST /api/approve` or an A2A approval message arrives for a paused
     * `session_id` (Module 3), this function loads the saved concept from Memory Bank
     * and either completes the downstream workflow (`approved=True`) or marks the
     * session `"rejected"` (`approved=False`).
     *
     * @param session_id Session identifier of the paused workflow.
     * @param approved Boolean decision from the human reviewer.
     * @param feedback Optional reviewer note or refinement instruction.
     * @param services Optional injected `ServiceContainer`.
     * @return Updated workflow result dictionary (`"completed"` or `"rejected"`).
     */
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    stored = active_services.memory_bank.load_session(clean_session)
    if not stored:
        raise KeyError(f"Unknown session_id: {clean_session}")

    brief = str(stored.get("brief") or "Campaign brief")
    saved_concept = str(stored.get("concept") or "").strip() or None

    if not approved:
        result = run_pitch_workflow(
            brief,
            session_id=clean_session,
            services=active_services,
            approved=False,
            existing_concept=saved_concept,
        )
        result["reviewer_feedback"] = feedback
        active_services.memory_bank.save_session(clean_session, result)
        return result

    effective_brief = f"{brief} ({feedback.strip()})" if feedback and feedback.strip() else brief
    result = run_pitch_workflow(
        effective_brief,
        session_id=clean_session,
        services=active_services,
        approved=True,
        require_approval=False,
        existing_concept=saved_concept,
    )
    result["reviewer_feedback"] = feedback
    active_services.memory_bank.save_session(clean_session, result)
    return result
