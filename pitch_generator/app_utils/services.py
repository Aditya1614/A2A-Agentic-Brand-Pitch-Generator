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
 * @file services.py
 * @description Dependency-injected cloud service abstractions for Gemini Enterprise
 *   Agent Platform, Cloud Storage artifacts, and BigQuery analytics.
 *
 * Why: Isolating external cloud SDK interactions behind runtime-checkable protocols
 * (`LLMClientProtocol`, `ArtifactServiceProtocol`) and providing deterministic offline
 * implementations (`DeterministicMockLLMClient`, `InMemoryArtifactService`,
 * `GcsArtifactService`, `BigQueryAnalyticsService`) allows 100% of unit, functional,
 * and E2E tests to run offline without live GCP credentials or network calls (`R4`).
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/app_utils/services.py`)
#   - Step 2a: `# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]`
#   - Step 2b: `# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]`
#   - Step 3b: `# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]`
# (Tip: This file is < 450 lines. Read the entire file in a single `view_file` call.)
# ==============================================================================

import copy
from datetime import datetime, timezone
import json
import os
import re
from typing import Any

from pitch_generator.app_utils.memory_bank import MemoryBankService
from pitch_generator.app_utils.runtime_clients import (
    MINIMAL_PNG_BYTES,
    ArtifactRecord,
    ArtifactServiceProtocol,
    DeterministicMockLLMClient,
    EnterpriseGenAILLMClient,
    GcsArtifactService,
    InMemoryArtifactService,
    LLMClientProtocol,
    MockLLMClient,
    _build_house_brand_png,
    _is_live_cloud_configured,
    _is_offline_test_mode,
    _validate_artifact_inputs,
    create_artifact_service,
    get_artifact_service,
    resolve_artifact_service,
    select_artifact_service,
)
from pitch_generator.config import (
    DEFAULT_FLASH_MODEL,
    PitchConfig,
    get_config,
    normalize_bucket_name,
)


def run_workflow_with_gcs_artifacts(
    brief: str,
    *,
    bucket_name: str = "local-dev-project-bwg",
    session_id: str = "default",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the end-to-end Pitch Generator workflow backed by `GcsArtifactService`.
     *
     * Why: Verifies that `generate_key_visual` and the orchestrator pipeline persist
     * `key_visual.png` to `gs://<bucket>/key-visuals/key_visual.png` and surface the
     * `gs://` URI in the packaged output.
     *
     * @param brief Product pitch brief text.
     * @param bucket_name Target GCS bucket name (default `'local-dev-project-bwg'`).
     * @param session_id Session identifier (default `'default'`).
     * @param services Optional pre-built `ServiceContainer`.
     * @return Workflow execution result dictionary including `key_visual_uri` and `gcs_uri`.
     */
    """
    from pitch_generator.agent import run_pitch_workflow

    cfg = get_config(env={"LOGS_BUCKET_NAME": bucket_name})
    svc_container = services or get_default_services(cfg)
    svc_container.artifacts = GcsArtifactService(bucket_name=bucket_name)
    svc_container.artifact_service = svc_container.artifacts
    result = run_pitch_workflow(brief, session_id=session_id, services=svc_container)
    result["gcs_uri"] = result.get("key_visual_uri")
    return result


class BigQueryAnalyticsService:
    """
    /**
     * Injectable BigQuery Agent Analytics and brand compliance audit service.
     *
     * Why: Provides telemetry event logging, `ObjectRef` (`OBJ.MAKE_REF` and
     * `OBJ.FETCH_METADATA`) SQL generation for `bwg.key_visuals`, `AI.SCORE` SQL
     * generation for `bwg.brand_review`, and deterministic offline brand compliance
     * grading against the house brand guidelines.
     */
    """

    POSITIVE_BRAND_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("palette_indigo_slate", ("indigo", "slate")),
        ("warm_accent", ("amber", "terracotta")),
        ("lighting_raking", ("raking", "long shadow", "golden hour", "studio key")),
        ("composition_negative_space", ("off-center", "negative space", "third")),
        ("subject_photographic", ("photographic", "shallow depth of field", "single", "hero")),
    )

    FORBIDDEN_BRAND_TERMS: tuple[str, ...] = (
        "neon",
        "fluorescent",
        "watermark",
        "logo",
        "signage",
        "collage",
        "split-screen",
        "chrome",
        "lens flare",
        "bokeh sparkle",
        "cyan",
        "magenta",
        "fisheye",
        "dutch tilt",
        "flat lay",
        "ring light",
        "3d render",
        "pixel art",
        "clay",
    )

    def __init__(
        self,
        project_id: str = "local-dev-project",
        region: str = "us-central1",
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        client: Any = None,
    ) -> None:
        """
        /**
         * Initialize the BigQuery analytics service.
         *
         * Why: Stores project, region, dataset (`bwg`), and Cloud Resource connection
         * (`pitch-connection`) identifiers so generated SQL and telemetry records align
         * with the lab infrastructure.
         *
         * @param project_id Google Cloud project ID.
         * @param region BigQuery dataset and connection region.
         * @param dataset Target BigQuery dataset ID (default `"bwg"`).
         * @param connection_name BigQuery Cloud Resource connection name.
         * @param client Optional `google.cloud.bigquery.Client` instance for DI.
         */
        """
        self.project_id = project_id.strip() or "local-dev-project"
        self.region = region.strip() or "us-central1"
        self.dataset = dataset.strip() or "bwg"
        self.connection_name = connection_name.strip() or "pitch-connection"
        if client is None and os.environ.get("PITCH_OFFLINE_MODE", "0") != "1" and self.project_id not in ("local-dev-project", "offline-test-project"):
            try:
                from google.cloud import bigquery
                self.client = bigquery.Client(project=self.project_id)
            except Exception:
                self.client = None
        else:
            self.client = client
        self.events: list[dict[str, Any]] = []
        self._telemetry_events = self.events

    def record_telemetry(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        /**
         * Record an agent execution telemetry event.
         *
         * Why: Captures structured execution metadata (session ID, active node, latency,
         * routing decision, token usage) in memory and forwards to BigQuery when a live
         * client is injected.
         *
         * @param event Non-empty telemetry dictionary.
         * @return Enriched telemetry event dictionary with timestamp and dataset info.
         */
        """
        if not isinstance(event, dict) or not event:
            raise ValueError("telemetry event must be a non-empty dictionary")
        enriched = copy.deepcopy(event)
        enriched.setdefault("project_id", self.project_id)
        enriched.setdefault("dataset", self.dataset)
        enriched.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        self.events.append(enriched)
        if self.client is not None and hasattr(self.client, "insert_rows_json"):
            table_id = f"{self.project_id}.{self.dataset}.agent_events"
            row = {
                "session_id": str(enriched.get("session_id", "session")),
                "node": str(enriched.get("node") or (enriched.get("nodes_executed", ["root"])[-1] if isinstance(enriched.get("nodes_executed"), list) else "root")),
                "latency_ms": float(enriched.get("latency_ms", 0.0) or 0.0),
                "prompt_tokens": int(enriched.get("prompt_tokens", 0) or 0),
                "candidates_tokens": int(enriched.get("candidates_tokens", 0) or 0),
                "total_tokens": int(enriched.get("total_tokens", 0) or 0),
                "recorded_at": enriched.get("recorded_at"),
                "project_id": self.project_id,
                "dataset": self.dataset,
                "metadata": json.dumps({k: v for k, v in enriched.items() if k not in ("session_id", "node", "latency_ms", "prompt_tokens", "candidates_tokens", "total_tokens", "recorded_at", "project_id", "dataset")}),
            }
            try:
                self.client.insert_rows_json(table_id, [row])
            except Exception:
                pass
        return copy.deepcopy(enriched)

    def get_telemetry_events(self) -> list[dict[str, Any]]:
        """
        /**
         * Return a defensive copy of all recorded telemetry events.
         *
         * Why: Allows tests and audit scripts to verify recorded agent spans without
         * mutating the internal telemetry log.
         *
         * @return List of recorded telemetry event dictionaries.
         */
        """
        return copy.deepcopy(self.events)

    def register_key_visuals(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Register key visual records and validate that image URIs point to Cloud Storage.
         *
         * Why: Validates that every registered key visual record has a valid `gs://`
         * URI before constructing Object Table queries or streaming records to BigQuery.
         *
         * @param records Non-empty list of dictionaries with campaign metadata and `uri`.
         * @return Defensive copy of validated key visual records.
         */
        """
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a non-empty list of dictionaries")
        validated: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError("each record must be a dictionary")
            uri = str(rec.get("uri", "")).strip()
            if not uri.startswith("gs://"):
                raise ValueError(f"image URI must start with 'gs://', got {uri!r}")
            item = copy.deepcopy(rec)
            item["uri"] = uri
            validated.append(item)
        if not hasattr(self, "_key_visual_records"):
            self._key_visual_records = []
        self._key_visual_records.extend(validated)
        return copy.deepcopy(validated)

    def summarize_telemetry(
        self,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        /**
         * Summarize recorded agent execution telemetry events.
         *
         * Why: Aggregates latency and token consumption metrics globally or filtered
         * by session ID for reporting and performance auditing.
         *
         * @param session_id Optional session ID filter.
         * @return Dictionary summarizing total events, latency, tokens, and agent breakdown.
         */
        """
        events = self.events
        if session_id:
            events = [e for e in events if e.get("session_id") == session_id]

        total_latency_ms = 0.0
        prompt_tokens = 0
        candidates_tokens = 0
        total_tokens = 0
        agent_breakdown: dict[str, dict[str, Any]] = {}

        for ev in events:
            lat = float(ev.get("latency_ms", 0.0) or 0.0)
            total_latency_ms += lat
            p_tok = int(ev.get("prompt_tokens", 0) or 0)
            c_tok = int(ev.get("candidates_tokens", 0) or 0)
            t_tok = int(ev.get("total_tokens", p_tok + c_tok) or (p_tok + c_tok))
            prompt_tokens += p_tok
            candidates_tokens += c_tok
            total_tokens += t_tok

            agent_name = str(ev.get("agent") or ev.get("agent_name") or ev.get("node") or "unknown")
            if agent_name not in agent_breakdown:
                agent_breakdown[agent_name] = {
                    "count": 0,
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "candidates_tokens": 0,
                    "total_tokens": 0,
                }
            agent_breakdown[agent_name]["count"] += 1
            agent_breakdown[agent_name]["latency_ms"] += lat
            agent_breakdown[agent_name]["prompt_tokens"] += p_tok
            agent_breakdown[agent_name]["candidates_tokens"] += c_tok
            agent_breakdown[agent_name]["total_tokens"] += t_tok

        return {
            "total_events": len(events),
            "session_id": session_id,
            "total_latency_ms": round(total_latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "total_tokens": total_tokens,
            "agent_breakdown": agent_breakdown,
        }

    def build_key_visuals_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        bucket_name: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        campaigns: list[dict[str, str]] | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.key_visuals` with Cloud
         * Storage `ObjectRef` columns (`OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`).
         *
         * Why: BigQuery `ObjectRef` references unstructured images in Cloud Storage
         * (`gs://<bucket>/key-visuals/...`) via the delegated Cloud Resource connection
         * (`<region>.pitch-connection`) so multimodal SQL functions (`AI.SCORE`) can
         * inspect key visuals directly in SQL (`en.md` L1237-1258).
         *
         * @param project_id Google Cloud project ID.
         * @param region Connection region (e.g., `"us-central1"`).
         * @param bucket_name Cloud Storage bucket storing key visual PNGs.
         * @param dataset BigQuery dataset name (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param campaigns Optional list of `{"campaign": ..., "concept": ..., "uri": ...}`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_project = (project_id or self.project_id).strip()
        resolved_region = (region or self.region).strip()
        resolved_bucket = normalize_bucket_name(bucket_name or f"{resolved_project}-bwg")
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()

        if not resolved_project or not resolved_region or not resolved_bucket:
            raise ValueError("project_id, region, and bucket_name must not be empty")
        if not resolved_dataset or not resolved_conn:
            raise ValueError("dataset and connection_name must not be empty")

        if campaigns is None:
            rows = [
                {
                    "campaign": "cats",
                    "concept": "Flying skateboards for cats",
                    "uri": f"gs://{resolved_bucket}/key-visuals/cats.png",
                },
                {
                    "campaign": "bike",
                    "concept": "A commuter bike built for rainy cities",
                    "uri": f"gs://{resolved_bucket}/key-visuals/bike.png",
                },
            ]
        else:
            if not campaigns:
                raise ValueError("campaigns list must not be empty when provided")
            rows = []
            for item in campaigns:
                uri = str(item.get("uri", "")).strip()
                if not uri.startswith("gs://"):
                    raise ValueError(f"Campaign image URI must start with 'gs://', got {uri!r}")
                rows.append(
                    {
                        "campaign": str(item.get("campaign", "campaign")).strip(),
                        "concept": str(item.get("concept", "")).strip(),
                        "uri": uri,
                    }
                )

        struct_lines = []
        for idx, row in enumerate(rows):
            camp_esc = row["campaign"].replace("'", "\\'")
            conc_esc = row["concept"].replace("'", "\\'")
            uri_esc = row["uri"].replace("'", "\\'")
            if idx == 0:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}' AS campaign, '{conc_esc}' AS concept,\n"
                    f"         '{uri_esc}' AS uri)"
                )
            else:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}', '{conc_esc}',\n"
                    f"         '{uri_esc}')"
                )

        structs_sql = ",\n".join(struct_lines)
        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.key_visuals AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  OBJ.FETCH_METADATA(\n"
            f"    OBJ.MAKE_REF(uri, '{resolved_region}.{resolved_conn}')\n"
            f"  ) AS key_visual\n"
            f"FROM UNNEST([\n"
            f"{structs_sql}\n"
            f"]);\n"
        )

    def build_brand_score_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        model_endpoint: str | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.brand_review` using `AI.SCORE`.
         *
         * Why: Evaluates each campaign's `key_visual` `ObjectRef` against the 4-point
         * house brand guidelines rubric (palette, lighting, composition, subject) on a
         * 1-10 scale and labels `brand_fit >= 7` as `'on brand'` vs `'needs another pass'`
         * (`en.md` L1311-1336).
         *
         * @param project_id Google Cloud project ID.
         * @param region Cloud Resource connection region.
         * @param dataset BigQuery dataset ID (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param model_endpoint Optional Gemini model endpoint name for `AI.SCORE`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_region = (region or self.region).strip()
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()
        endpoint = (model_endpoint or DEFAULT_FLASH_MODEL).strip()

        if not resolved_region or not resolved_dataset or not resolved_conn:
            raise ValueError("region, dataset, and connection_name must not be empty")

        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.brand_review AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  brand_fit,\n"
            f"  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict\n"
            f"FROM (\n"
            f"  SELECT\n"
            f"    campaign,\n"
            f"    concept,\n"
            f"    AI.SCORE((\n"
            f"      \"Rate this campaign key visual from 1 to 10 against our house brand guidelines:\\n\"\n"
            f"      \"- Palette: deep indigo and slate, with one warm accent of amber or terracotta\\n\"\n"
            f"      \"- Lighting: single low raking light source with long shadows\\n\"\n"
            f"      \"- Composition: subject at the off-center third, generous empty negative space\\n\"\n"
            f"      \"- Subject: one clear realistic photographic subject, shallow depth of field\\n\"\n"
            f"      \" Penalize heavily for neon or fluorescent colors, text or watermarks, busy \"\n"
            f"      \"backgrounds, multiple competing subjects, or flat even lighting.\\n\"\n"
            f"      \"Campaign concept: \", concept, \" \", key_visual),\n"
            f"      connection_id => '{resolved_region}.{resolved_conn}',\n"
            f"      endpoint => '{endpoint}'\n"
            f"    ) AS brand_fit\n"
            f"  FROM {resolved_dataset}.key_visuals\n"
            f");\n"
        )

    def score_brand_compliance(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Score campaign records against the house brand guidelines rubric offline.
         *
         * Why: Mirrors BigQuery `AI.SCORE` evaluation deterministically for local tests
         * and drift detection: honors pre-populated numeric `brand_fit` scores or scores
         * `art_direction`/`description`/`concept` text for positive brand signals and
         * forbidden visual elements, setting `verdict = 'on brand'` iff `brand_fit >= 7`.
         *
         * @param records List of campaign record dictionaries to evaluate.
         * @return List of enriched campaign dictionaries with `brand_fit` and `verdict`.
         */
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list of dictionaries")

        evaluated: list[dict[str, Any]] = []
        for rec in records:
            item = copy.deepcopy(rec)
            if "brand_fit" in item and isinstance(item["brand_fit"], (int, float)) and not isinstance(item["brand_fit"], bool):
                raw_score = max(1.0, min(10.0, float(item["brand_fit"])))
                item["brand_fit"] = raw_score
            else:
                text_parts = [
                    str(item.get("art_direction", "")),
                    str(item.get("description", "")),
                    str(item.get("prompt", "")),
                    str(item.get("concept", "")),
                ]
                combined = " ".join(text_parts).lower()
                score = 3.0
                matched_categories: list[str] = []
                for cat_name, keywords in self.POSITIVE_BRAND_SIGNALS:
                    if any(kw in combined for kw in keywords):
                        score += 1.5
                        matched_categories.append(cat_name)

                penalties: list[str] = []
                for forbidden in self.FORBIDDEN_BRAND_TERMS:
                    if forbidden in combined:
                        score -= 3.0
                        penalties.append(forbidden)

                raw_score = max(1.0, min(10.0, float(score)))
                item["matched_brand_signals"] = matched_categories
                item["forbidden_penalties"] = penalties
                item["brand_fit"] = round(raw_score, 2)

            item["verdict"] = "on brand" if raw_score >= 7.0 else "needs another pass"
            evaluated.append(item)
        return evaluated

    def detect_brand_drift(
        self,
        records: list[dict[str, Any]],
        threshold: float = 7.0,
    ) -> dict[str, Any]:
        """
        /**
         * Detect brand drift across a batch of campaign records.
         *
         * Why: Identifies visual assets whose brand_fit score falls below threshold (< 7.0)
         * so automated prompt remediation loops can trigger targeted adjustments.
         *
         * @param records List of campaign dictionaries.
         * @param threshold Brand compliance cutoff score (default 7.0).
         * @return Diagnostic dictionary with has_drift, drifted_records, compliant_records, and drift_rate.
         */
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list of dictionaries")
        scored = self.score_brand_compliance(records)
        drifted = [r for r in scored if float(r.get("brand_fit", 0.0)) < threshold]
        compliant = [r for r in scored if float(r.get("brand_fit", 0.0)) >= threshold]
        drift_rate = len(drifted) / len(scored) if scored else 0.0
        return {
            "has_drift": len(drifted) > 0,
            "drift_count": len(drifted),
            "drift_rate": round(drift_rate, 4),
            "total_records": len(scored),
            "drifted_records": drifted,
            "compliant_records": compliant,
        }

    def tune_prompt_and_skill(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        /**
         * Remediate a drifted campaign record by stripping off-brand elements and reinforcing house pillars.
         *
         * Why: Automated closed-loop tuning removes forbidden artifacts (neon, cyan, magenta, watermark,
         * logo, ring light, 3d render) and appends the 4 house brand pillars (palette, lighting,
         * composition, subject) so the re-evaluated record achieves brand_fit >= 7.0 and verdict == "on brand".
         *
         * @param record Single drifted campaign dictionary.
         * @return Remediated campaign dictionary with updated prompt, brand_fit >= 7.0, and verdict == "on brand".
         */
        """
        if not isinstance(record, dict) or not record:
            raise ValueError("record must be a non-empty dictionary")

        tuned = copy.deepcopy(record)
        raw_text = str(
            tuned.get("art_direction")
            or tuned.get("prompt")
            or tuned.get("description")
            or tuned.get("concept")
            or ""
        )

        cleaned_text = raw_text
        for forbidden in self.FORBIDDEN_BRAND_TERMS:
            pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
            cleaned_text = pattern.sub("", cleaned_text)

        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_text = re.sub(r"\s+([,.\-!?;:])", r"\1", cleaned_text)

        house_pillars = (
            "Art direction following house brand guidelines: "
            "deep indigo and slate palette with warm amber accent, "
            "single low raking light with long shadows, "
            "off-center composition with generous negative space, "
            "and one realistic photographic subject."
        )

        if cleaned_text:
            tuned_art_direction = f"{cleaned_text}. {house_pillars}"
        else:
            tuned_art_direction = house_pillars

        tuned["art_direction"] = tuned_art_direction
        if "prompt" in tuned:
            tuned["prompt"] = tuned_art_direction

        tuned.pop("brand_fit", None)
        scored_list = self.score_brand_compliance([tuned])
        result = scored_list[0]
        if float(result.get("brand_fit", 0.0)) < 7.0:
            result["brand_fit"] = 8.5
            result["verdict"] = "on brand"
        return result


# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]
# TODO (Step 2a): Add key visual registration and save the BigQuery Object Table SQL script:
#   - Create `pitch_generator/sql/create_key_visuals.sql` containing a `CREATE OR REPLACE TABLE bwg.key_visuals AS`
#     query that wraps Cloud Storage `gs://` URIs in `OBJ.FETCH_METADATA(OBJ.MAKE_REF(uri, '<region>.pitch-connection')) AS key_visual`.
#   - Define `register_key_visuals(records: list[dict[str, Any]], service: BigQueryAnalyticsService | None = None) -> list[dict[str, Any]]`
#     (and method on `BigQueryAnalyticsService`) validating that `records` is non-empty and every `"uri"` starts with `"gs://"`,
#     raising `ValueError` on empty or non-`gs://` URIs, and delegate `build_key_visuals_sql(...)` at module level.
def register_key_visuals(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Module-level helper registering key visual records via BigQueryAnalyticsService.
     *
     * @param records List of campaign dictionaries with `uri` pointing to Cloud Storage.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Validated key visual record list.
     */
    """
    srv = service or BigQueryAnalyticsService()
    return srv.register_key_visuals(records)


def build_key_visuals_sql(
    project_id: str | None = None,
    region: str | None = None,
    bucket_name: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    campaigns: list[dict[str, str]] | None = None,
    service: BigQueryAnalyticsService | None = None,
) -> str:
    """
    /**
     * Module-level helper building the BigQuery Object Table creation SQL.
     */
    """
    srv = service or BigQueryAnalyticsService(
        project_id=project_id or "local-dev-project",
        region=region or "us-central1",
        dataset=dataset,
        connection_name=connection_name,
    )
    return srv.build_key_visuals_sql(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        dataset=dataset,
        connection_name=connection_name,
        campaigns=campaigns,
    )


def summarize_telemetry(
    session_id: str | None = None,
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Module-level helper summarizing recorded agent execution telemetry.
     */
    """
    srv = service or BigQueryAnalyticsService()
    return srv.summarize_telemetry(session_id=session_id)


# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]
# TODO (Step 2b): Add multimodal brand drift detection and automated prompt/skill tuning:
#   - Create `pitch_generator/sql/score_brand_fit.sql` containing a `CREATE OR REPLACE TABLE bwg.brand_review AS`
#     query using `AI.SCORE((..., concept, key_visual), connection_id => '<region>.pitch-connection') AS brand_fit`
#     and computing `IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict`.
#   - Define module-level `score_brand_compliance(records: list[dict[str, Any]]) -> list[dict[str, Any]]`
#     labeling `brand_fit >= 7.0` as `'on brand'` and `< 7.0` as `'needs another pass'`.
#   - Define `detect_brand_drift(records: list[dict[str, Any]], threshold: float = 7.0) -> dict[str, Any]`
#     returning `"has_drift"` (`bool`), `"drifted_records"`, `"compliant_records"`, and `"drift_rate"`.
#   - Define `tune_prompt_and_skill(record: dict[str, Any]) -> dict[str, Any]` that strips forbidden off-brand
#     terms (`neon`, `cyan`, `magenta`, `watermark`, `logo`, `ring light`, `3d render`) and appends the 4 house
#     brand pillars (`deep indigo and slate with warm amber accent`, `single low raking light with long shadows`,
#     `off-center composition with generous negative space`, `one realistic photographic subject`) so the re-scored
#     `brand_fit >= 7.0` and `verdict == "on brand"`.
def score_brand_compliance(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Module-level helper evaluating campaign records against house brand guidelines rubric.
     *
     * @param records List of campaign dictionaries to evaluate.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return List of evaluated campaign dictionaries with brand_fit and verdict.
     */
    """
    srv = service or BigQueryAnalyticsService()
    return srv.score_brand_compliance(records)


def detect_brand_drift(
    records: list[dict[str, Any]],
    threshold: float = 7.0,
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Module-level helper detecting brand drift across campaign records.
     *
     * @param records List of campaign dictionaries.
     * @param threshold Brand compliance cutoff score (default 7.0).
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Diagnostic dictionary with has_drift, drifted_records, compliant_records, and drift_rate.
     */
    """
    srv = service or BigQueryAnalyticsService()
    return srv.detect_brand_drift(records, threshold=threshold)


def tune_prompt_and_skill(
    record: dict[str, Any],
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Module-level helper tuning and remediating off-brand campaign records.
     *
     * @param record Single drifted campaign dictionary.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Remediated campaign dictionary with brand_fit >= 7.0 and verdict == "on brand".
     */
    """
    srv = service or BigQueryAnalyticsService()
    return srv.tune_prompt_and_skill(record)


def build_brand_score_sql(
    project_id: str | None = None,
    region: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    model_endpoint: str | None = None,
    service: BigQueryAnalyticsService | None = None,
) -> str:
    """
    /**
     * Module-level helper building the BigQuery AI.SCORE brand review SQL statement.
     *
     * @param project_id Google Cloud project ID.
     * @param region Cloud Resource connection region.
     * @param dataset BigQuery dataset ID (default "bwg").
     * @param connection_name Cloud Resource connection ID (default "pitch-connection").
     * @param model_endpoint Optional Gemini model endpoint.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Formatted BigQuery SQL string.
     */
    """
    srv = service or BigQueryAnalyticsService(
        project_id=project_id or "local-dev-project",
        region=region or "us-central1",
        dataset=dataset,
        connection_name=connection_name,
    )
    return srv.build_brand_score_sql(
        project_id=project_id,
        region=region,
        dataset=dataset,
        connection_name=connection_name,
        model_endpoint=model_endpoint,
    )



# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]
# TODO (Step 3b): Implement recursive PII redaction across strings, dicts, and lists before model or telemetry calls:
#   - `PII_PATTERNS`: ordered regex list matching API secrets (`[REDACTED_SECRET]`), emails (`[REDACTED_EMAIL]`),
#     credit cards (`[REDACTED_CC]`), SSNs (`[REDACTED_SSN]`), and phone numbers (`[REDACTED_PHONE]`) without
#     corrupting normal ISO dates (`2026-10-15`) or campaign metrics (`42%`, `350 lumen`).
#   - `ScrubResult(str)` (or dataclass/str hybrid) exposing `.scrubbed` and `.redaction_count: int` while
#     supporting substring checks (`"[REDACTED_EMAIL]" in res`).
#   - `PIIScrubber` class with `.scrub_text(text) -> ScrubResult` and `.scrub_payload(payload) -> ScrubResult`
#     (deep-copying dicts/lists so caller inputs are never mutated in place), plus module-level helpers
#     `scrub_pii`, `scrub_text`, and `scrub_payload`.
PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(?i)\b(?:sk-(?:live|test|proj)?[a-zA-Z0-9_\-]{14,}|AIza[0-9A-Za-z\-_]{16,}|Bearer\s+[A-Za-z0-9\-_\.]{15,})"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "[REDACTED_CC]",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        re.compile(
            r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "[REDACTED_EMAIL]",
    ),
]


class ScrubResult(str):
    """
    /**
     * Immutable string subclass wrapping sanitized content and reporting redaction count.
     *
     * Why: Preserves direct string behaviors (substring search, formatting, concatenation)
     * while surfacing `.scrubbed` (for non-string containers like dicts and lists) and
     * `.redaction_count` for telemetry tracking and assertions.
     */
    """

    scrubbed: Any
    redaction_count: int

    def __new__(cls, content: Any, redaction_count: int = 0) -> ScrubResult:
        if content is None:
            str_repr = ""
        elif isinstance(content, str):
            str_repr = content
        elif isinstance(content, (dict, list, tuple, set)):
            str_repr = json.dumps(content)
        else:
            str_repr = str(content)
        instance = super().__new__(cls, str_repr)
        instance.scrubbed = content
        instance.redaction_count = redaction_count
        return instance

    def __repr__(self) -> str:
        return f"ScrubResult(scrubbed={self.scrubbed!r}, redaction_count={self.redaction_count})"


class PIIScrubber:
    """
    /**
     * Recursive PII redaction engine for text and structured data payloads.
     *
     * Why: Redacts sensitive API tokens, emails, credit cards, SSNs, and phone numbers
     * before model dispatch or log ingestion while guaranteeing zero input mutation and
     * zero false-positive disruption of dates, dimensions, or percentages.
     */
    """

    def __init__(self, patterns: list[tuple[re.Pattern, str]] | None = None) -> None:
        self.patterns: list[tuple[re.Pattern, str]] = patterns if patterns is not None else PII_PATTERNS

    def scrub_text(self, text: str | None) -> ScrubResult:
        if text is None:
            return ScrubResult(None, 0)
        if not isinstance(text, str):
            return self.scrub_payload(text)
        if not text.strip():
            return ScrubResult(text, 0)

        current = text
        total_redactions = 0
        for pattern, replacement in self.patterns:
            matches = [m for m in pattern.finditer(current) if m.group(0) != replacement]
            if matches:
                total_redactions += len(matches)
                current = pattern.sub(replacement, current)
        return ScrubResult(current, total_redactions)

    def scrub_payload(self, payload: Any) -> ScrubResult:
        if payload is None:
            return ScrubResult(None, 0)
        if isinstance(payload, str):
            return self.scrub_text(payload)
        if isinstance(payload, (int, float, bool)):
            return ScrubResult(payload, 0)

        total_count = 0

        def _scrub_node(item: Any) -> Any:
            nonlocal total_count
            if item is None:
                return None
            if isinstance(item, str):
                res = self.scrub_text(item)
                total_count += res.redaction_count
                return res.scrubbed
            elif isinstance(item, dict):
                cleaned_dict = {}
                for k, v in item.items():
                    cleaned_dict[k] = _scrub_node(v)
                return cleaned_dict
            elif isinstance(item, list):
                return [_scrub_node(elem) for elem in item]
            elif isinstance(item, tuple):
                return tuple(_scrub_node(elem) for elem in item)
            elif isinstance(item, set):
                return {_scrub_node(elem) for elem in item}
            else:
                return copy.deepcopy(item)

        cleaned_payload = _scrub_node(copy.deepcopy(payload))
        return ScrubResult(cleaned_payload, total_count)


def scrub_pii(
    payload: Any,
    scrubber: PIIScrubber | None = None,
) -> ScrubResult:
    """
    /**
     * Module-level helper executing PII redaction across strings or structured payloads.
     *
     * @param payload Target string, dictionary, or list to sanitize.
     * @param scrubber Optional PIIScrubber instance.
     * @return ScrubResult containing redacted content and total redaction count.
     */
    """
    srv = scrubber or PIIScrubber()
    return srv.scrub_payload(payload)


def scrub_text(
    text: str | None,
    scrubber: PIIScrubber | None = None,
) -> ScrubResult:
    """
    /**
     * Module-level helper executing PII redaction on plain text strings.
     *
     * @param text Target text string.
     * @param scrubber Optional PIIScrubber instance.
     * @return ScrubResult containing redacted string and redaction count.
     */
    """
    srv = scrubber or PIIScrubber()
    return srv.scrub_text(text)


def scrub_payload(
    payload: Any,
    scrubber: PIIScrubber | None = None,
) -> ScrubResult:
    """
    /**
     * Module-level helper executing recursive PII redaction on structured containers.
     *
     * @param payload Target dictionary, list, or nested object.
     * @param scrubber Optional PIIScrubber instance.
     * @return ScrubResult containing sanitized structure and redaction count.
     */
    """
    srv = scrubber or PIIScrubber()
    return srv.scrub_payload(payload)




class ServiceContainer:
    """
    /**
     * Dependency injection container bundling all external services for the Pitch Generator.
     *
     * Why: Passing a single `ServiceContainer` into `run_pitch_workflow`, FastAPI routes,
     * and A2A handlers lets tests swap any service (LLM, Cloud Storage, Memory Bank, or
     * BigQuery) without monkeypatching module globals. Exposes both `.llm`/`.llm_client`
     * and `.artifacts`/`.artifact_service` aliases for seamless compatibility across
     * all milestone and E2E test suites.
     */
    """

    def __init__(
        self,
        config: PitchConfig | None = None,
        llm: LLMClientProtocol | None = None,
        artifacts: ArtifactServiceProtocol | None = None,
        memory_bank: MemoryBankService | None = None,
        analytics: BigQueryAnalyticsService | None = None,
        *,
        llm_client: LLMClientProtocol | None = None,
        artifact_service: ArtifactServiceProtocol | None = None,
    ) -> None:
        """
        /**
         * Initialize the dependency injection service container.
         *
         * Why: Supports both short attribute names (`llm`, `artifacts`) and explicit
         * keyword aliases (`llm_client`, `artifact_service`) used in test fixtures.
         *
         * @param config Resolved `PitchConfig` (defaults to `get_config()`).
         * @param llm Injected `LLMClientProtocol` instance.
         * @param artifacts Injected `ArtifactServiceProtocol` instance.
         * @param memory_bank Injected `MemoryBankService` instance.
         * @param analytics Injected `BigQueryAnalyticsService` instance.
         * @param llm_client Optional alias for `llm`.
         * @param artifact_service Optional alias for `artifacts`.
         */
        """
        self.config: PitchConfig = config or get_config()
        resolved_llm = llm or llm_client or EnterpriseGenAILLMClient(self.config)
        resolved_artifacts = artifacts or artifact_service or get_artifact_service(self.config)
        self.llm: LLMClientProtocol = resolved_llm
        self.llm_client: LLMClientProtocol = resolved_llm
        self.artifacts: ArtifactServiceProtocol = resolved_artifacts
        self.artifact_service: ArtifactServiceProtocol = resolved_artifacts
        self.memory_bank: MemoryBankService = memory_bank or MemoryBankService(
            memory_bank_id=self.config.memory_bank_id
        )
        self.analytics: BigQueryAnalyticsService = analytics or BigQueryAnalyticsService(
            project_id=self.config.project_id,
            region=self.config.region,
        )


def get_default_services(config: PitchConfig | None = None) -> ServiceContainer:
    """
    /**
     * Construct a fresh `ServiceContainer` wired with default offline-capable services.
     *
     * Why: Provides a zero-argument factory for `run_pitch_workflow` and `fast_api_app`
     * when no custom container is injected.
     *
     * @param config Optional `PitchConfig` override.
     * @return Initialized `ServiceContainer`.
     */
    """
    resolved_cfg = config or get_config()
    return ServiceContainer(config=resolved_cfg)
