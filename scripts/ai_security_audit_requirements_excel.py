#!/usr/bin/env python3

from __future__ import annotations

import unicodedata
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl

try:
    from lib.ai_runtime import AIRuntime
except Exception:
    AIRuntime = None  # type: ignore

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = None  # type: ignore


FINGERPRINT_PATH = Path("/mnt/data/vision360_fingerprint.json")
REQUISITES_PATH = Path("/mnt/data/requisites.json")
OUTPUT_XLSX_PATH = Path("/mnt/data/security_audit_requirements.xlsx")

NEGATIVE_RISK_TOKENS = [
    "insecure", "unsafe", "weak", "debug", "debuggable", "cleartext", "allow_clear_text",
    "trust_all", "accept_all", "ignore_ssl", "skip_verification", "bypass", "hardcoded",
    "leak", "plaintext", "world_readable", "world_writable", "sha1", "md5",
    "debug_certificate", "janus", "v1_signature", "exported_true", "backup_enabled_true",
    "http_based",
]

APPLICABILITY_TOKENS = [
    "components", "present", "detected", "uses_", "is_used", "feature", "webview_components",
]

PASSWORD_HASHING_POSITIVE_IDS = {"has_password_hashing_uses_salts", "has_password_hashing_uses_kdf"}

MALWARE_REQ_TOKENS = [
    "malware", "adware", "virus", "trojan", "spyware", "ransomware", "malicious code", "malicious",
]
MALWARE_FLAG_TOKENS = ["malware", "adware", "virus", "trojan", "spyware", "ransomware", "malicious"]

OVERRIDE_SCOPE_FLAG_IDS = {
    "has_org_notifies_users_of_security_updates",
    "has_manifest_allow_clear_text_traffic_true",
    "has_uses_os_level_update_mechanisms",
    "has_android_dynamic_code_loading",
    "has_webview_remote_content",
    "has_soap_uses_mutual_tls",
    "has_defined_certificate_management_policy",
    "has_defined_identity_lifecycle_policy",
    "has_webview_components",
    "has_webview_javascript",
    "has_webview_file_scheme",
    "has_insecure_http_based_webview_communication",
    "has_webview_javascript_interface_limited_to_trusted_content",
    "has_soap_api_usage",
    "has_proper_ws_security_headers",
    "has_soap_message_level_encryption",
    "has_soap_message_level_signatures",
    "has_soap_prevents_replay_attacks",
    "has_saml_based_sso",
    "has_soap_validates_saml_token_expiry",
    "has_uses_xml_signatures",
    "has_uses_xml_encryption",
    "has_soap_uses_strict_schema_validation",
    "has_content_provider_actively_exposed",
    "has_manifest_custom_permission_defined",
}

GATE_FLAG_IDS = {
    "has_webview_components",
    "has_webview_remote_content",
    "has_android_dynamic_code_loading",
    "has_soap_api_usage",
    "has_saml_based_sso",
    "has_content_provider_actively_exposed",
    "has_manifest_custom_permission_defined",
    "has_uses_os_level_update_mechanisms",
    "has_org_notifies_users_of_security_updates",
    "has_defined_certificate_management_policy",
    "has_defined_identity_lifecycle_policy",
}


def _json_decode_error_details(e: json.JSONDecodeError) -> str:
    return f"{type(e).__name__}: {e.msg} (line {e.lineno}, col {e.colno})"


def load_json_with_one_repair(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error parsing {path.name}: {_json_decode_error_details(e)}", file=sys.stderr)
        patched: Optional[str] = None
        if "Extra data" in e.msg:
            patched = raw[: e.pos].rstrip()
        elif raw.lstrip().startswith("{{"):
            s = raw.lstrip()
            idx = raw.find(s)
            patched = raw[:idx] + s[1:]
        if patched is None:
            raise
        return json.loads(patched)


def normalize_requirements(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("requirements"), list):
        return data["requirements"]
    raise ValueError("requisites.json must be a JSON array of requirements (or an object with a 'requirements' array).")


def normalize_fingerprint_flags(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("flags"), list):
            return data["flags"]
        for k in ("results", "items"):
            if isinstance(data.get(k), list):
                return data[k]
    raise ValueError("vision360_fingerprint.json does not contain a recognizable list of flags.")


def is_prohibitive(req_desc: str) -> bool:
    s = req_desc.lower()
    return ("must not" in s) or ("shall not" in s) or ("should not" in s)


def is_conditional(req_desc: str) -> bool:
    s = req_desc.lower()
    return any(re.search(p, s) for p in [r"\bif\b", r"\bwhen\b", r"where applicable", r"on older versions"])


def req_mentions_malware(req_desc: str) -> bool:
    s = req_desc.lower()
    return any(t in s for t in MALWARE_REQ_TOKENS)


def classify_flag_for_requirement(flag_id: str, flag_title: str, req_desc: str) -> str:
    fid = (flag_id or "").lower()
    title = (flag_title or "").lower()
    if flag_id in PASSWORD_HASHING_POSITIVE_IDS:
        return "POSITIVE_CONTROL"
    if req_mentions_malware(req_desc) and any(t in fid or t in title for t in MALWARE_FLAG_TOKENS):
        return "NEGATIVE_RISK"
    if any(tok in fid or tok in title for tok in NEGATIVE_RISK_TOKENS):
        return "NEGATIVE_RISK"
    if any(tok in fid or tok in title for tok in APPLICABILITY_TOKENS):
        return "APPLICABILITY"
    return "POSITIVE_CONTROL"


def parse_summary_normalized(summary: Any) -> str:
    if not isinstance(summary, str) or not summary.strip():
        return "NA"
    s = summary.strip()
    v = s.split("=")[-1].strip() if "=" in s else s
    v_up = v.upper().replace(" ", "")
    if v_up in ("YES", "Y"):
        return "YES"
    if v_up in ("NO", "N"):
        return "NO"
    if v_up in ("NA", "N/A"):
        return "NA"
    m = re.search(r"\b(YES|NO|NA|N/A)\b", s.upper())
    if m:
        tok = m.group(1)
        return "NA" if tok in ("NA", "N/A") else tok
    return "NA"


def eval_against_expected(observed: str, expected: str) -> str:
    if observed == "NA":
        return "UNKNOWN"
    if expected == "YES":
        return "SUPPORT" if observed == "YES" else "CONTRADICT"
    if expected == "NO":
        return "SUPPORT" if observed == "NO" else "CONTRADICT"
    return "UNKNOWN"


@dataclass
class FlagEvidence:
    id: str
    title: str
    state: str
    summary: str
    summary_norm: str
    notes: str
    evidence_count: int
    classification: str
    expected: Optional[str]
    outcome: str


@dataclass
class RequirementAudit:
    puid: str
    description_en: str
    result: str
    flags_used: List[str]
    justification_en: str


def build_flag_evidence(flag_obj: Optional[Dict[str, Any]], flag_id: str, classification: str, expected: Optional[str]) -> FlagEvidence:
    if not flag_obj:
        return FlagEvidence(flag_id, "", "missing", "", "NA", "flag not present in fingerprint", 0, classification, expected, "MISSING")
    app_verdict = flag_obj.get("app_verdict") or {}
    summary = app_verdict.get("summary", "")
    summary_norm = parse_summary_normalized(summary)
    state = str(app_verdict.get("state", "") or "")
    notes = str(app_verdict.get("notes", "") or "")
    evidence_count = int(app_verdict.get("evidence_count", 0) or 0)
    if classification == "APPLICABILITY":
        outcome = "NA_OR_GATE"
    else:
        outcome = eval_against_expected(summary_norm, expected or "YES")
    return FlagEvidence(str(flag_obj.get("id") or flag_id), str(flag_obj.get("title") or ""), state, str(summary), summary_norm, notes, evidence_count, classification, expected, outcome)


def compute_override_scenario_activated(flag_evidences: List[FlagEvidence], flag_ids: List[str]) -> Tuple[bool, List[FlagEvidence]]:
    gate_evs = [fe for fe in flag_evidences if fe.id in flag_ids and fe.id in GATE_FLAG_IDS]
    return any(fe.summary_norm == "YES" for fe in gate_evs), gate_evs


def compute_conditional_scenario_activated(flag_evidences: List[FlagEvidence]) -> Optional[bool]:
    app_flags = [fe for fe in flag_evidences if fe.classification == "APPLICABILITY"]
    if not app_flags:
        return None
    return any(fe.summary_norm == "YES" for fe in app_flags)


def _split_flags_string(flags_str: str) -> List[str]:
    s = (flags_str or "").strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    return [p.strip() for p in re.split(r"[,\n;]+", s) if p.strip()]


def extract_req_fields(req_obj: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    puid = str(req_obj.get("PUID") or req_obj.get("id") or "").strip()
    desc = str(req_obj.get("Requirement description") or req_obj.get("Description") or req_obj.get("Descripcion") or "").strip()
    flags = req_obj.get("Flags", [])
    if isinstance(flags, str):
        flags_list = _split_flags_string(flags)
    elif isinstance(flags, list):
        flags_list = [str(x).strip() for x in flags if str(x).strip()]
    else:
        flags_list = []
    return puid, desc, flags_list


def audit_requirement(puid: str, desc: str, flag_ids: List[str], flags_by_id: Dict[str, Dict[str, Any]]) -> Tuple[str, List[FlagEvidence], Dict[str, Any]]:
    prohibitive = is_prohibitive(desc)
    conditional = is_conditional(desc)
    flag_evs: List[FlagEvidence] = []
    for fid in flag_ids:
        fobj = flags_by_id.get(fid)
        title = str((fobj or {}).get("title") or "")
        classification = classify_flag_for_requirement(fid, title, desc)
        expected: Optional[str] = None
        if classification == "POSITIVE_CONTROL":
            expected = "YES"
        elif classification == "NEGATIVE_RISK":
            expected = "NO"
        flag_evs.append(build_flag_evidence(fobj, fid, classification, expected))

    override_used = bool(flag_ids) and set(flag_ids).issubset(OVERRIDE_SCOPE_FLAG_IDS)
    override_scenario_activated: Optional[bool] = None
    gate_flags: List[FlagEvidence] = []
    conditional_scenario_activated: Optional[bool] = None

    has_negative_risk_yes = any((fe.classification == "NEGATIVE_RISK" and fe.summary_norm == "YES") for fe in flag_evs if fe.outcome != "MISSING")

    if override_used:
        override_scenario_activated, gate_flags = compute_override_scenario_activated(flag_evs, flag_ids)
        if override_scenario_activated is False:
            result = ("no" if has_negative_risk_yes else "yes") if prohibitive else ("no" if has_negative_risk_yes else "n/a")
            meta = dict(prohibitive=prohibitive, conditional=conditional, override_used=True, override_scenario_activated=override_scenario_activated, gate_flags=[ge.id for ge in gate_flags], conditional_scenario_activated=None)
            return result, flag_evs, meta

    if conditional:
        conditional_scenario_activated = compute_conditional_scenario_activated(flag_evs)

    non_app = [fe for fe in flag_evs if fe.classification != "APPLICABILITY"]
    any_contradict = any(fe.outcome == "CONTRADICT" for fe in non_app)
    all_support = (len(non_app) > 0) and all(fe.outcome == "SUPPORT" for fe in non_app)
    any_unknown = any(fe.outcome in ("UNKNOWN", "MISSING") for fe in non_app)

    if any_contradict:
        result = "no"
    elif all_support and not any_unknown:
        result = "yes"
    else:
        if conditional and (conditional_scenario_activated is False):
            result = "yes" if prohibitive else "n/a"
        else:
            result = "n/a"

    meta = dict(prohibitive=prohibitive, conditional=conditional, override_used=override_used, override_scenario_activated=override_scenario_activated, gate_flags=[ge.id for ge in gate_flags], conditional_scenario_activated=conditional_scenario_activated)
    return result, flag_evs, meta


class _ResponsesCompat:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def create(self, model: Optional[str] = None, input: Any = None, max_output_tokens: Optional[int] = None, reasoning: Optional[dict] = None):
        return self._runtime.create(input=input, model=model, max_output_tokens=max_output_tokens, reasoning=reasoning)

    def parse(self, model: Optional[str] = None, input: Any = None, text_format: Any = None, max_output_tokens: Optional[int] = None, reasoning: Optional[dict] = None):
        return self._runtime.parse(input=input, text_format=text_format, model=model, max_output_tokens=max_output_tokens, reasoning=reasoning)


class _ClientCompat:
    def __init__(self, runtime: Any) -> None:
        self.responses = _ResponsesCompat(runtime)


def openai_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or AIRuntime is None:
        return None
    task = os.getenv("AI_TASK", "").strip() or "ai_requirements_excel"
    return _ClientCompat(AIRuntime(task=task, api_key=api_key))


def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _normalize_typography(s: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"', "\u2013": "-", "\u2014": "-", "\u00A0": " "}
    return "".join(repl.get(ch, ch) for ch in s)


def looks_non_english(text: str) -> bool:
    if not text or not text.strip():
        return False
    s = _normalize_typography(text.strip())
    for ch in s:
        if ord(ch) <= 127:
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            return True
    spanish_markers = [" el ", " la ", " los ", " las ", " de ", " del ", " para ", " y ", " o ", " debe ", " cuando ", " donde ", " aplicacion ", " seguridad ", " requisito ", " descripcion ", " evidencias "]
    s_low = f" {s.lower()} "
    return any(m in s_low for m in spanish_markers)


def translate_texts_to_english_via_openai(items: List[Dict[str, str]]) -> Dict[str, str]:
    client = openai_client()
    if client is None or BaseModel is None:
        return {}

    model = os.getenv("OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    effort = os.getenv("OPENAI_REASONING_EFFORT", "medium").strip() or "medium"
    max_tokens = env_int("OPENAI_MAX_OUTPUT_TOKENS", 2000)
    supports_parse = hasattr(client.responses, "parse")

    class TranslationItem(BaseModel):
        id: str
        text_en: str

    class TranslationBatch(BaseModel):
        items: List[TranslationItem]

    system = (
        "You translate short requirement descriptions to English.\n"
        "Strict rules:\n"
        "- Output English only.\n"
        "- Preserve meaning. Do not add new requirements.\n"
        "- Keep the translation concise and professional.\n"
        "- Return ONLY JSON in the form: {\"items\": [{\"id\": \"...\", \"text_en\": \"...\"}, ...]}.\n"
    )
    user_payload = {"items": items}
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            if supports_parse:
                resp = client.responses.parse(
                    model=model,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                    text_format=TranslationBatch,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": effort},
                )
                parsed = resp.output_parsed
                return {it.id: it.text_en.strip() for it in parsed.items}

            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                max_output_tokens=max_tokens,
                reasoning={"effort": effort},
            )
            txt = (getattr(resp, "output_text", "") or "").strip()
            m = re.search(r"\{.*\}\s*$", txt, flags=re.DOTALL)
            if not m:
                raise ValueError("No valid JSON object found in model response.")
            obj = json.loads(m.group(0))
            out: Dict[str, str] = {}
            for it in obj.get("items", []):
                if isinstance(it, dict) and "id" in it and "text_en" in it:
                    out[str(it["id"])] = str(it.get("text_en") or "").strip()
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.6 * attempt)
    print(f"[WARN] AI translation failed after retries: {last_err}", file=sys.stderr)
    return {}


def generate_justifications_via_openai(batch_ctx: List[Dict[str, Any]]) -> Dict[str, str]:
    client = openai_client()
    if client is None or BaseModel is None:
        return {}

    model = os.getenv("OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    effort = os.getenv("OPENAI_REASONING_EFFORT", "medium").strip() or "medium"
    max_tokens = env_int("OPENAI_MAX_OUTPUT_TOKENS", 2000)
    supports_parse = hasattr(client.responses, "parse")

    class JustificationItem(BaseModel):
        id: str
        justification: str

    class JustificationBatch(BaseModel):
        items: List[JustificationItem]

    system = (
        "You draft audit justifications in English for security requirement outcomes.\n"
        "Strict rules:\n"
        "- Output English only.\n"
        "- If any provided notes contain non-English text, paraphrase them into English and do not quote them verbatim.\n"
        "- Do not invent evidence or flags.\n"
        "- Use only the provided context.\n"
        "- Keep 1 to 3 sentences per requirement.\n"
        "- Explicitly mention: app_verdict.state, normalized app_verdict.summary (YES/NO/NA), notes (include 'Fallback verdict' if present), and evidence_count.\n"
        "- If a flag is not present in the fingerprint, state: 'flag not present in fingerprint'.\n"
        "- Do not change the precomputed result.\n"
        "- Return ONLY JSON in the form: {\"items\": [{\"id\": \"...\", \"justification\": \"...\"}, ...]}.\n"
    )
    user_payload = {"batch": batch_ctx}
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            if supports_parse:
                resp = client.responses.parse(
                    model=model,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                    text_format=JustificationBatch,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": effort},
                )
                parsed = getattr(resp, "output_parsed", None)
                if parsed is not None:
                    return {it.id: it.justification.strip() for it in parsed.items}
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                max_output_tokens=max_tokens,
                reasoning={"effort": effort},
            )
            txt = (getattr(resp, "output_text", "") or "").strip()
            m = re.search(r"\{.*\}\s*$", txt, flags=re.DOTALL)
            if not m:
                raise ValueError("No valid JSON object found in model response.")
            obj = json.loads(m.group(0))
            out: Dict[str, str] = {}
            for it in obj.get("items", []):
                if isinstance(it, dict) and "id" in it and "justification" in it:
                    out[str(it["id"])] = str(it.get("justification") or "").strip()
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.6 * attempt)
    print(f"[WARN] AI justification generation failed after retries: {last_err}", file=sys.stderr)
    return {}


def deterministic_justification(req: RequirementAudit, flag_evidences: List[FlagEvidence], meta: Dict[str, Any]) -> str:
    def _note_hint(fe: FlagEvidence) -> str:
        return " (Fallback verdict)" if "fallback verdict" in (fe.notes or "").lower() else ""

    def _brief(fe: FlagEvidence) -> str:
        exp = f", expected={fe.expected}" if fe.expected in ("YES", "NO") else ""
        return f"{fe.id}={fe.summary_norm} (state={fe.state}, evidence_count={fe.evidence_count}{_note_hint(fe)}{exp})"

    result = req.result
    present = [fe for fe in flag_evidences if fe.outcome != "MISSING"]
    missing = [fe for fe in flag_evidences if fe.outcome == "MISSING"]
    contradicts = [fe for fe in present if fe.outcome == "CONTRADICT"]
    supports = [fe for fe in present if fe.outcome == "SUPPORT"]
    unknowns = [fe for fe in present if fe.outcome == "UNKNOWN"]

    s1 = f"Result: {result} for requirement {req.puid}."
    if result == "yes":
        s2 = "Based on the fingerprint and the flags mapped to this requirement, the application is compliant because the evaluated signals support the expected secure posture and no contradicting signals were observed among the mapped flags."
    elif result == "no":
        s2 = "Based on the fingerprint and the flags mapped to this requirement, the application is not compliant because at least one mapped signal contradicts the expected outcome."
    else:
        s2 = "Based on the fingerprint and the flags mapped to this requirement, the result is n/a because the available signals are insufficient, not applicable under the detected scenario, or contain unknown/NA coverage for the mapped flags."

    if result == "no" and contradicts:
        s3 = f"Contradicting signals: {'; '.join(_brief(fe) for fe in contradicts[:3])}."
    elif supports:
        s3 = f"Key supporting signals: {'; '.join(_brief(fe) for fe in supports[:3])}."
    else:
        any_present = present[:3]
        s3 = f"Observed mapped signals: {'; '.join(_brief(fe) for fe in any_present) if any_present else 'none'}."

    gaps = []
    if missing:
        gaps.append(f"{len(missing)} flag(s) were not present in the fingerprint (flag not present in fingerprint).")
    if unknowns:
        gaps.append("Some mapped flag summaries were NA/unknown, limiting certainty to fingerprint coverage.")
    if not gaps:
        gaps.append("This justification is derived solely from the fingerprint summaries, states, notes, and evidence counts for the mapped flags.")
    sentences = [s1, s2, s3, " ".join(gaps)]
    if meta.get("override_used") and meta.get("override_scenario_activated") is not None:
        sentences.append("Feature-gating (override scope) was applied; scenario_activated=%s based on the gate flags included in the requirement flag list." % ("TRUE" if meta.get("override_scenario_activated") else "FALSE"))
    if meta.get("conditional") and meta.get("conditional_scenario_activated") is not None:
        sentences.append("Conditional applicability was evaluated using the mapped applicability signals; scenario_activated=%s." % ("TRUE" if meta.get("conditional_scenario_activated") else "FALSE"))
    return " ".join(sentences[:6])


def main() -> None:
    strict_english = env_bool("STRICT_ENGLISH_OUTPUT", True)
    if not FINGERPRINT_PATH.exists():
        raise SystemExit(f"Missing required file: {FINGERPRINT_PATH}.")
    if not REQUISITES_PATH.exists():
        raise SystemExit(f"Missing required file: {REQUISITES_PATH}.")

    fingerprint_data = load_json_with_one_repair(FINGERPRINT_PATH)
    requisites_data = load_json_with_one_repair(REQUISITES_PATH)
    requirements = normalize_requirements(requisites_data)
    flags_list = normalize_fingerprint_flags(fingerprint_data)

    flags_by_id: Dict[str, Dict[str, Any]] = {}
    for f in flags_list:
        if isinstance(f, dict) and f.get("id"):
            flags_by_id[str(f["id"])] = f

    raw_desc_by_puid: Dict[str, str] = {}
    to_translate: List[Dict[str, str]] = []
    for req_obj in requirements:
        if not isinstance(req_obj, dict):
            continue
        puid, desc, _ = extract_req_fields(req_obj)
        if not puid:
            continue
        if not desc:
            desc = "(missing description)"
        raw_desc_by_puid[puid] = desc
        if looks_non_english(desc):
            to_translate.append({"id": puid, "text": desc})

    translations: Dict[str, str] = {}
    if to_translate:
        client = openai_client()
        if client is None:
            if strict_english:
                raise SystemExit("STRICT_ENGLISH_OUTPUT is enabled, but non-English requirement descriptions were detected and the AI runtime is unavailable. Provide an English requisites.json or expose OPENAI_API_KEY.")
            print("[WARN] Non-English descriptions detected; translation skipped because AI runtime is unavailable.", file=sys.stderr)
        else:
            print(f"[AI] Translating {len(to_translate)} requirement description(s) to English...")
            translations = translate_texts_to_english_via_openai(to_translate)
            missing = [it["id"] for it in to_translate if it["id"] not in translations or not translations[it["id"]].strip()]
            if missing and strict_english:
                raise SystemExit("STRICT_ENGLISH_OUTPUT is enabled, but translation failed for one or more items: " + ", ".join(missing))
            if missing:
                print(f"[WARN] Translation missing for {len(missing)} item(s); originals may remain.", file=sys.stderr)

    desc_en_by_puid: Dict[str, str] = {}
    for puid, desc in raw_desc_by_puid.items():
        desc_en = translations.get(puid, "").strip() or desc
        if strict_english and looks_non_english(desc_en):
            raise SystemExit(f"STRICT_ENGLISH_OUTPUT is enabled, but the final description for PUID={puid} is not English.")
        desc_en_by_puid[puid] = desc_en

    batch_size = env_int("OPENAI_BATCH_SIZE", 25)
    batch_size = 25 if batch_size <= 0 else batch_size
    audits: List[RequirementAudit] = []
    counts = {"yes": 0, "no": 0, "n/a": 0}
    total = len(requirements)
    n_batches = (total + batch_size - 1) // batch_size if total else 0

    for b in range(n_batches):
        start = b * batch_size
        end = min(total, start + batch_size)
        req_slice = requirements[start:end]
        print(f"[AI] Batch {b + 1}/{n_batches}: {len(req_slice)} requirements")
        batch_ctx: List[Dict[str, Any]] = []
        batch_results: List[Tuple[RequirementAudit, List[FlagEvidence], Dict[str, Any]]] = []

        for req_obj in req_slice:
            if not isinstance(req_obj, dict):
                continue
            puid, _desc_raw, flag_ids = extract_req_fields(req_obj)
            if not puid:
                continue
            desc_en = desc_en_by_puid.get(puid, "(missing description)")
            result, flag_evs, meta = audit_requirement(puid, desc_en, flag_ids, flags_by_id)
            req_audit = RequirementAudit(puid=puid, description_en=desc_en, result=result, flags_used=flag_ids, justification_en="")
            batch_results.append((req_audit, flag_evs, meta))

            flags_ctx = []
            for fe in flag_evs:
                note_hint = "Fallback verdict" if "fallback verdict" in (fe.notes or "").lower() else ""
                notes_trim = (fe.notes or "")[:800]
                if strict_english and looks_non_english(notes_trim):
                    notes_trim = "Non-English notes detected in fingerprint. Do not quote verbatim; provide an English paraphrase."
                flags_ctx.append({
                    "id": fe.id,
                    "classification": fe.classification,
                    "expected": fe.expected,
                    "observed_summary_norm": fe.summary_norm,
                    "app_verdict": {"state": fe.state, "summary": fe.summary, "notes": notes_trim, "evidence_count": fe.evidence_count, "note_hint": note_hint},
                    "outcome": fe.outcome,
                })
            batch_ctx.append({"id": puid, "description_en": desc_en, "result": result, "flags_used": flag_ids, "meta": meta, "flags": flags_ctx})

        use_openai_just = env_bool("USE_OPENAI_JUSTIFICATIONS", False)
        just_map = generate_justifications_via_openai(batch_ctx) if use_openai_just else {}

        for req_audit, flag_evs, meta in batch_results:
            just = (just_map.get(req_audit.puid, "") or "").strip() or deterministic_justification(req_audit, flag_evs, meta)
            if strict_english and looks_non_english(just):
                just = deterministic_justification(req_audit, flag_evs, meta)
                if strict_english and looks_non_english(just):
                    raise SystemExit(f"STRICT_ENGLISH_OUTPUT is enabled, but the generated justification for PUID={req_audit.puid} is not English.")
            req_audit.justification_en = just
            audits.append(req_audit)
            counts[req_audit.result] += 1

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "audit"
    ws.append(["id (PUID)", "Description (EN)", "Result", "Justification (EN)", "Flags used"])
    for a in audits:
        ws.append([a.puid, a.description_en, a.result, a.justification_en, ", ".join(a.flags_used)])

    OUTPUT_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_XLSX_PATH)
    print(f"[OK] Excel generated: {OUTPUT_XLSX_PATH}")
    print(f"[SUMMARY] total={len(audits)} yes={counts['yes']} no={counts['no']} n/a={counts['n/a']}")


if __name__ == "__main__":
    try:
        main()
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {_json_decode_error_details(e)}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
