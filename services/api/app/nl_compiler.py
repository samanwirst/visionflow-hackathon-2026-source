"""Deterministic natural-language -> workflow-DSL compiler (M2-AC1).

This is a disclosed, rule-based compiler. It does NOT call a text model and
requires no credentials; it maps a constrained set of English phrasings onto the
validated :class:`WorkflowDSL`. This is an honest fallback per the task brief:
no generated code is ever produced or executed — only a structured draft.

Contract:

- Output is always a *draft* (``requires_approval=True``, saved disabled). The
  user edits and explicitly approves/enables it; the compiler never arms a rule.
- The grammar is intentionally narrow. A request that names more than one object
  class, more than one trigger, an upper-bound count, a negated or unrecognised
  visual condition, an unsupported appearance attribute, or an unsupported action
  is **rejected** with a clear error rather than silently reduced to a partial
  workflow (M2-001). Assumptions the compiler had to make surface as warnings.
- A visual predicate (a question the geometric rule cannot answer, e.g. "the
  hood is open") is attached only when explicitly requested, exactly once and
  positively; it routes to the Teacher at runtime (M2-AC2), never at compile time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import SUPPORTED_CLASSES

# Class synonyms -> canonical supported class. "vehicle"/"car" are deliberately
# NOT merged silently when ambiguous; see below.
_CLASS_SYNONYMS = {
    "car": "car", "cars": "car", "sedan": "car", "automobile": "car",
    "truck": "truck", "trucks": "truck", "lorry": "truck",
    "bus": "bus", "buses": "bus",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "motorcycles": "motorcycle",
    "bicycle": "bicycle", "bike": "bicycle", "bicycles": "bicycle",
    "person": "person", "people": "person", "pedestrian": "person", "worker": "person",
}

_ENTER_WORDS = ("enter", "enters", "arrive", "arrives", "comes into", "pulls into", "drives into")
_EXIT_WORDS = ("exit", "exits", "leaves", "leave", "departs", "drives out", "pulls out")
_DWELL_WORDS = ("stays", "stay", "remains", "remain", "dwell", "parked", "idles", "waits")

# Supported visual conditions (whitelist). A visual condition is a question the
# geometric rule cannot answer; only these recognised conditions are preserved as
# a structured, non-executable predicate. An unrecognised condition is REJECTED
# (never silently dropped). The prompt/labels are fixed server vocabulary.
_VISUAL_CONDITIONS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:hood|bonnet)\b.*\b(?:open|up|raised|popped)\b|\b(?:open|raised|popped)\s+(?:hood|bonnet)\b"),
     "the vehicle hood is open"),
    (re.compile(r"\b(?:trunk|boot|tailgate)\b.*\bopen\b|\bopen\s+(?:trunk|boot|tailgate)\b"),
     "the vehicle trunk is open"),
    (re.compile(r"\bdoor\b.*\bopen\b|\bopen\s+door\b"),
     "a vehicle door is open"),
    (re.compile(r"\b(?:head)?lights?\b.*\b(?:on|flashing|lit)\b"),
     "the vehicle lights are on"),
)

# Connectives that introduce a condition clause. Their presence means the user
# asked for an extra check; if it is not a recognised visual condition the
# request is rejected rather than compiled without it (M2-AC1).
_CONDITION_MARKERS = (
    "verify", "check whether", "check if", "check that",
    "confirm that", "confirm whether",
    "with its", "with their", "with the", "while its", "while the",
    "whose", "having its", "having their", "that has", "that is",
    "if its", "if their", "if the", "if it", "provided that", "as long as",
)

# Negation adjacent to a visual state word ("hood not open", "lights not on") or
# an explicit closed/off state. A negated visual condition is rejected: the
# Teacher confirms a *positive* predicate, so a negation would silently invert
# the semantics (M2-001). "no" alone is omitted to avoid false positives.
_NEGATION_NEAR_STATE = re.compile(
    r"\b(?:not|never|without|isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t|won['’]?t)\s+"
    r"(?:\w+\s+){0,3}?(?:open|up|raised|popped|on|lit|flashing)\b"
)
_CLOSED_STATE = re.compile(
    r"\b(?:hood|bonnet|trunk|boot|tailgate|door|doors|lights?|headlights?)\s+"
    r"(?:\w+\s+){0,3}?(?:closed|shut|down|lowered)\b|\b(?:lights?|headlights?)\s+off\b"
)

# Unsupported appearance / attribute clauses (colour, tint, condition). These are
# not something the geometric rule or the whitelisted visual predicates can
# answer, so a request that constrains on them is rejected rather than dropped.
_UNSUPPORTED_ATTR = re.compile(
    r"\b(red|blue|green|black|white|silver|grey|gray|yellow|orange|brown|gold|"
    r"tinted|dented|damaged|scratched|dirty|rusty|muddy|wet)\b"
)

# Unsupported actions: physical/control actuation or external dispatch. Only the
# in-app create_notification is supported, so these are rejected (never silently
# downgraded to a notification) (M2-001).
_UNSUPPORTED_ACTION = re.compile(
    r"\b(unlock|re-?lock|lock|activate|actuate|deactivate|disarm|arm the|"
    r"honk|deploy|dispatch(?:\s+(?:security|a|the))?|sound the|"
    r"(?:open|close|raise|lower|shut)\s+the\s+(?:gate|barrier|garage|door|arm|bay\s*door)|"
    r"turn\s+(?:on|off)|switch\s+(?:on|off)|"
    r"move\s+the\s+(?:robot|arm)|start\s+the\s+(?:engine|robot|motor)|"
    r"stop\s+the\s+(?:engine|robot|motor))\b"
)

# External delivery / code execution (kept from the prior pass).
_EXTERNAL_DELIVERY = re.compile(
    r"\b(webhook|https?|post to|call the api|slack|telegram|e-?mails?|sms|"
    r"text message|phone|call the)\b"
)

# Data-management actions (delete/erase recordings, evidence, events, ...). These
# are not supported and must NOT be silently downgraded to a notification (the
# "delete the video" reproduction, M2-001).
_DATA_ACTION = re.compile(
    r"\b(delete|remove|erase|wipe|purge|destroy|drop|discard|clear)\b"
    r"(?:\s+\w+){0,3}?\s+"
    r"\b(video|videos|recording|recordings|footage|clip|clips|evidence|event|"
    r"events|file|files|data|notification|notifications|workflow|workflows|source|frame|frames)\b"
)

# Negated trigger: "does not enter", "never exits", "doesn't stay", ... The engine
# fires on the positive occurrence of a trigger, so a negated trigger cannot be
# expressed and must be rejected rather than silently turned into the positive
# rule ("does not enter" -> entry) (M2-001).
_TRIGGER_WORD_STEMS = sorted(
    {w.split()[0] for w in (_ENTER_WORDS + _EXIT_WORDS + _DWELL_WORDS)},
    key=len, reverse=True,
)
_NEG_TRIGGER = re.compile(
    r"\b(?:not|never|without|does\s+not|do\s+not|did\s+not|doesn['’]?t|don['’]?t|"
    r"didn['’]?t|isn['’]?t|aren['’]?t|won['’]?t|no\s+longer)\b"
    r"(?:\s+\w+){0,2}?\s+"
    r"\b(?:" + "|".join(re.escape(s) for s in _TRIGGER_WORD_STEMS) + r")\b"
)

_TIME_UNITS = r"(?:second|sec|minute|min|hour|hr)s?"

# Upper-bound / non-lower-bound duration comparison for a dwell rule. The dwell
# engine only expresses a MINIMUM dwell ("for N" / "longer than N"); an upper
# bound ("less than N minutes", "within N minutes") cannot be represented and is
# rejected rather than silently turned into a >= rule (M2-001).
_UPPER_BOUND_DURATION = re.compile(
    r"\b(?:less than|under|within|at most|no more than|fewer than|shorter than|"
    r"up to|below|no longer than|maximum of|max)\b(?:\s+\w+){0,2}?\s+"
    r"\d+(?:\.\d+)?\s*" + _TIME_UNITS
)

# Comparator vocabulary used to strip consumed count/duration phrases from the
# residual-token check below.
_COMPARATORS = (
    r"(?:more than|greater than|at least|at most|no more than|fewer than|"
    r"less than|over|exceeds?|up to|under|within|minimum of|maximum of|"
    r"longer than|shorter than|no longer than|below)"
)

# Scaffolding / command vocabulary that carries no unsupported semantics. Any
# token that is NOT consumed by a recognised grammar element AND is not in this
# set makes the request fall outside the supported grammar and is rejected
# (M2-001: reject every unconsumed token rather than compiling a partial rule).
# Deliberately excludes semantic words (not/never/without/no/and-of-conditions),
# which must surface as rejections.
_FILLER = {
    "a", "an", "the", "me", "my", "our", "we", "us", "i", "you", "your",
    "when", "whenever", "if", "once", "while", "as", "then", "there", "here",
    "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "in", "into", "to", "at", "on", "of", "for", "from", "by", "with", "within",
    "and", "or", "than",
    "please", "kindly", "just", "also", "still", "now",
    "do", "does", "did", "should", "shall", "will", "would", "can", "could",
    "may", "must",
    "it", "its", "their", "them", "they",
    "create", "creates", "make", "makes", "raise", "raises", "generate",
    "send", "sends", "give", "show", "display", "log", "logs",
    "alert", "alerts", "alerting", "notify", "notifies", "notification",
    "notifications", "warn", "warns", "warning",
    "flag", "flags", "tell", "tells", "let", "know", "report", "reports",
    "trigger", "triggers", "fire", "fires",
    "something", "anything", "somebody", "someone",
}


def _residual_tokens(low: str, zone_name: str) -> list[str]:
    """Return meaningful tokens NOT consumed by any recognised grammar element.

    Everything the compiler understands (class, trigger, zone, duration/count,
    the supported visual condition, condition markers, scaffolding) is blanked
    out; whatever survives and is not scaffolding means the request used phrasing
    outside the supported grammar and must be rejected (M2-001)."""
    work = list(low)

    def blank(a: int, b: int) -> None:
        for i in range(a, b):
            work[i] = " "

    def blank_re(pattern: str) -> None:
        for m in re.finditer(pattern, low):
            blank(*m.span())

    if zone_name:
        start = low.find(zone_name)
        while start != -1:
            blank(start, start + len(zone_name))
            start = low.find(zone_name, start + len(zone_name))
    blank_re(r"\bvehicles?\b")
    for w in _CLASS_SYNONYMS:
        blank_re(rf"\b{re.escape(w)}\b")
    for w in _ENTER_WORDS + _EXIT_WORDS + _DWELL_WORDS:
        blank_re(rf"\b{re.escape(w)}\b")
    blank_re(rf"\b{_COMPARATORS}\s+\d+(?:\.\d+)?")
    blank_re(rf"\b{_COMPARATORS}\b")
    blank_re(rf"\d+(?:\.\d+)?\s*{_TIME_UNITS}")
    blank_re(r"\b\d+\s*(?:or more|or above)\b")
    blank_re(r"\b(?:number of|count of|how many)\b")
    blank_re(r"\d+\s*\+")
    for pat, _ in _VISUAL_CONDITIONS:
        blank_re(pat.pattern)
    for marker in _CONDITION_MARKERS:
        blank_re(re.escape(marker))
    # Vehicle-state vocabulary (used by the visual conditions and negation checks).
    blank_re(r"\b(?:hood|bonnet|trunk|boot|tailgate|door|doors|lights?|headlights?)\b")
    blank_re(r"\b(?:open|up|raised|popped|closed|shut|down|lowered|lit|flashing)\b")
    blank_re(r"\b(?:on|off)\b")
    text = "".join(work)
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _FILLER]


def _match_visual_conditions(text: str) -> list[str]:
    """Return every recognised visual-condition description present in ``text``."""
    return [desc for pat, desc in _VISUAL_CONDITIONS if pat.search(text)]


def _build_predicate(desc: str) -> dict:
    return {
        "prompt": f"Does the image clearly show that {desc}?",
        "labels": ["yes", "no"],
        "expected_label": "yes",
        "description": desc,
    }


@dataclass
class CompileResult:
    ok: bool
    draft: dict | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanation: str = ""


def _find_class(text: str) -> tuple[str | None, list[str], list[str]]:
    """Return (object_class, warnings, errors).

    Rejects the ambiguous word 'vehicle' and rejects requests that name more than
    one supported class (e.g. "car or truck") rather than silently picking one.
    """
    warnings: list[str] = []
    # "vehicle" is ambiguous across car/truck/bus; ask the user instead of guessing.
    if re.search(r"\bvehicles?\b", text) and not any(
        re.search(rf"\b{re.escape(w)}\b", text) for w in _CLASS_SYNONYMS
    ):
        return None, [], [
            "the word 'vehicle' is ambiguous — specify car, truck, bus, motorcycle or bicycle"
        ]
    found: list[str] = []
    for word, canon in _CLASS_SYNONYMS.items():
        if re.search(rf"\b{re.escape(word)}\b", text) and canon not in found:
            found.append(canon)
    if not found:
        return None, warnings, []
    if len(found) > 1:
        return None, [], [
            "more than one object class was requested "
            f"({', '.join(sorted(found))}); specify exactly one supported class"
        ]
    return found[0], warnings, []


def _has_count(text: str) -> bool:
    """True when the text expresses an object-count comparison (not a duration)."""
    for m in re.finditer(
        r"\b(?:more than|at least|at most|no more than|fewer than|less than|over|"
        r"exceeds?|up to|minimum of|maximum of)\s+(\d+)\s*(\w+)?",
        text,
    ):
        unit = m.group(2) or ""
        if not re.match(r"(second|sec|minute|min|hour|hr)", unit):
            return True
    if re.search(r"\b(number of|count of|how many)\b", text):
        return True
    if re.search(r"\b\d+\s*(?:or more|or above|\+)\b", text):
        return True
    return False


def _trigger_families(text: str) -> list[str]:
    """Return the distinct trigger families referenced in ``text`` (M2-001). More
    than one means the request is ambiguous and is rejected by the caller."""
    fams: list[str] = []
    if any(re.search(rf"\b{re.escape(w)}\b", text) for w in _ENTER_WORDS):
        fams.append("object_entered_zone")
    if any(re.search(rf"\b{re.escape(w)}\b", text) for w in _EXIT_WORDS):
        fams.append("object_exited_zone")
    dwell = any(re.search(rf"\b{re.escape(w)}\b", text) for w in _DWELL_WORDS) or bool(
        re.search(r"\b(longer|more|greater) than\b.*\b(second|sec|minute|min|hour|hr)s?\b", text)
    )
    if dwell:
        fams.append("dwell_in_zone")
    if _has_count(text):
        fams.append("object_count_exceeded")
    return fams


def _parse_duration_seconds(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr)s?", text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit.startswith("min"):
        return value * 60.0
    if unit.startswith("h"):
        return value * 3600.0
    return value


def _parse_count(text: str) -> tuple[int | None, list[str], list[str]]:
    """Return (threshold, warnings, errors) for a count comparison.

    The engine fires on ``inside_count > threshold`` (strictly greater), so:
    - "more than N" / "exceeds N" / "over N"  -> threshold = N (exclusive)
    - "at least N" / "N or more"              -> threshold = N-1 (inclusive)
    - "at most N" / "fewer than N" / ...       -> REJECTED (no upper bound)
    An inclusive bound of 1 ("at least 1") cannot be expressed (threshold must be
    >= 1) and is rejected explicitly (M2-001)."""
    if re.search(r"\b(at most|no more than|fewer than|less than|up to|maximum of)\s+\d+", text):
        return None, [], [
            "only a lower-bound count is supported (e.g. 'more than N' or "
            "'at least N'), not an upper bound"
        ]
    m = re.search(r"\b(?:at least|minimum of)\s+(\d+)\b", text) or re.search(
        r"\b(\d+)\s*(?:or more|or above|\+)\b", text
    )
    if m:
        n = int(m.group(1))
        thr = n - 1
        if thr < 1:
            return None, [], [
                f"count threshold '{n}' is too low to express; use 'at least 2' or more"
            ]
        return thr, [
            f"interpreting 'at least {n}' as more than {thr} (inclusive lower bound)"
        ], []
    m = re.search(
        r"\b(?:more than|greater than|over|exceeds?)\s+(\d+)\b(?!\s*(?:second|sec|minute|min|hour|hr))",
        text,
    )
    if m:
        n = int(m.group(1))
        if n < 1:
            return None, [], ["count threshold must be at least 1"]
        return n, [], []
    return None, [], []


def _match_zone(text: str, zones: list[dict]) -> tuple[int | None, str, list[str], list[str]]:
    """Return (zone_id, matched_name, warnings, errors). zones: [{'id','name'}].

    More than one named zone (e.g. "Service Bay A or Service Bay B") is rejected
    rather than silently reduced to the first match (M2-001)."""
    if not zones:
        return None, "", [], ["no zones exist for this source — draw a zone first"]
    present = [z for z in zones if z["name"].lower() and z["name"].lower() in text]
    if len(present) > 1:
        names = ", ".join(f"'{z['name']}'" for z in present)
        return None, "", [], [
            f"more than one zone was named ({names}); specify a single zone"
        ]
    if len(present) == 1:
        return present[0]["id"], present[0]["name"].lower(), [], []
    if len(zones) == 1:
        return zones[0]["id"], zones[0]["name"].lower(), [
            f"assuming zone '{zones[0]['name']}' (the only zone)"
        ], []
    names = ", ".join(f"'{z['name']}'" for z in zones)
    return None, "", [], [f"zone is ambiguous — name one of: {names}"]


def compile_text(text: str, source_id: int, zones: list[dict]) -> CompileResult:
    """Compile a natural-language instruction into a WorkflowDSL draft dict.

    ``zones`` is the list of existing zones for the source: [{'id','name'}].
    """
    raw = (text or "").strip()
    if len(raw) < 3:
        return CompileResult(False, errors=["instruction is empty or too short"])
    if len(raw) > 2000:
        return CompileResult(False, errors=["instruction is too long (max 2000 chars)"])
    low = raw.lower()

    errors: list[str] = []
    warnings: list[str] = []

    object_class, cls_warn, cls_err = _find_class(low)
    warnings += cls_warn
    errors += cls_err
    if object_class is None and not cls_err:
        errors.append(
            "could not identify a supported object class "
            f"(one of {sorted(SUPPORTED_CLASSES)})"
        )

    # A negated trigger ("does not enter", "never exits") cannot be expressed and
    # must be rejected, never silently compiled into the positive rule (M2-001).
    if _NEG_TRIGGER.search(low):
        errors.append(
            "a negated trigger (e.g. 'does not enter') is not supported; describe "
            "the positive event to detect (enter, exit, dwell, or count)"
        )

    # Trigger: exactly one family is allowed. Alternatives/conjunctions such as
    # "enters or exits" are rejected rather than silently reduced to one (M2-001).
    families = _trigger_families(low)
    trigger_type: str | None = None
    if not families:
        errors.append(
            "could not identify a supported trigger — use words like 'enters', "
            "'exits', 'stays for N minutes', or 'more than N'"
        )
    elif len(families) > 1:
        errors.append(
            "more than one trigger type was requested "
            f"({', '.join(sorted(families))}); specify a single trigger "
            "(enter, exit, dwell, or count)"
        )
    else:
        trigger_type = families[0]

    zone_id, zone_name, zone_warn, zone_err = _match_zone(low, zones)
    warnings += zone_warn
    errors += zone_err

    # Text with the matched zone name removed, so leftover-attribute checks do not
    # trip on words inside a zone name.
    residual = low.replace(zone_name, " ") if zone_name else low

    duration_seconds = None
    threshold = None
    if trigger_type == "dwell_in_zone":
        # An upper-bound duration ("less than 10 minutes") cannot be expressed by
        # the minimum-dwell engine and must be rejected, not turned into a >= rule.
        if _UPPER_BOUND_DURATION.search(low):
            errors.append(
                "only a minimum dwell duration is supported (e.g. 'for 10 minutes' "
                "or 'longer than 10 minutes'), not an upper bound"
            )
        duration_seconds = _parse_duration_seconds(low)
        if duration_seconds is None:
            errors.append("dwell rule needs a duration, e.g. 'for 10 minutes'")
    if trigger_type == "object_count_exceeded":
        threshold, count_warn, count_err = _parse_count(low)
        warnings += count_warn
        errors += count_err
        if threshold is None and not count_err:
            errors.append("count rule needs a threshold, e.g. 'more than 3 cars'")

    # Visual condition handling (M2-001): preserve a single positive recognised
    # condition; reject negation, multiple conditions, and unrecognised clauses.
    conditions = _match_visual_conditions(low)
    had_condition_marker = any(m in low for m in _CONDITION_MARKERS)
    negated = bool(_NEGATION_NEAR_STATE.search(low) or _CLOSED_STATE.search(low))
    visual_predicate: dict | None = None
    if negated and (conditions or had_condition_marker):
        errors.append(
            "negated visual conditions (e.g. 'hood not open') are not supported; "
            "state the positive condition to verify"
        )
    elif len(conditions) > 1:
        errors.append(
            "more than one visual condition was requested "
            f"({'; '.join(conditions)}); specify a single visual condition"
        )
    elif len(conditions) == 1:
        visual_predicate = _build_predicate(conditions[0])
    elif had_condition_marker:
        # A requested condition clause that is not a supported visual condition is
        # rejected rather than silently dropped (M2-AC1).
        errors.append(
            "an unsupported or unrecognized visual condition was requested; "
            "supported visual conditions: hood/trunk/door open, lights on"
        )

    # Unsupported appearance/attribute clause (colour, tint, damage, ...). Checked
    # on the zone-stripped text so a zone name cannot trigger it.
    attr = _UNSUPPORTED_ATTR.search(residual)
    if attr:
        errors.append(
            f"an unsupported attribute filter ('{attr.group(1)}') was requested; "
            "VisionFlow cannot filter on colour or appearance"
        )

    # Action: only in-app create_notification is supported. External delivery,
    # physical actuation, and code/command execution are rejected rather than
    # silently downgraded to a local notification.
    if _EXTERNAL_DELIVERY.search(low):
        errors.append(
            "external delivery (webhook/HTTP/email/SMS/phone/Slack/Telegram) is "
            "not available; only in-app notifications are supported"
        )
    if _UNSUPPORTED_ACTION.search(low):
        errors.append(
            "that action is not available (physical actuation / external dispatch); "
            "only in-app notifications are supported"
        )
    if _DATA_ACTION.search(low):
        errors.append(
            "deleting or modifying stored videos, recordings or evidence is not a "
            "supported action; only in-app notifications are supported"
        )
    if (
        re.search(r"\b(python|javascript|shell|bash|source code|exec|eval)\b", low)
        or (re.search(r"\b(run|execute|invoke)\b", low)
            and re.search(r"\b(code|script|command|program)\b", low))
    ):
        errors.append(
            "executing code, scripts, or shell commands is not supported"
        )

    # Final backstop (M2-001): after extracting every supported element, no
    # meaningful token may remain. Anything left means the request used phrasing
    # outside the supported grammar (an extra condition, an unknown clause, ...)
    # and must be rejected rather than compiled into a partial rule. Skipped when
    # the class/zone could not be resolved (the specific errors above are clearer).
    if object_class is not None and not zone_err:
        leftover = _residual_tokens(low, zone_name)
        if leftover:
            errors.append(
                "unsupported or unrecognized phrasing: "
                + ", ".join(f"'{t}'" for t in dict.fromkeys(leftover))
                + "; rephrase using the supported grammar "
                "(class, enter/exit/dwell/count, zone, optional single visual condition)"
            )

    if errors:
        return CompileResult(
            False, errors=errors, warnings=warnings,
            explanation="Could not compile a complete, unambiguous workflow.",
        )

    trigger: dict = {"type": trigger_type, "zone_id": zone_id}
    if duration_seconds is not None:
        trigger["duration_seconds"] = duration_seconds
    if threshold is not None:
        trigger["threshold"] = threshold

    name = _suggest_name(object_class, trigger_type, visual_predicate)
    draft = {
        "name": name,
        "source_id": source_id,
        "object_class": object_class,
        "trigger": trigger,
        "action": {"type": "create_notification", "config": {"title": name}},
        "cooldown_seconds": 10.0,
        "requires_approval": True,
    }
    if visual_predicate is not None:
        draft["visual_predicate"] = visual_predicate
        warnings.append(
            "a visual predicate was detected; it requires a configured Teacher "
            "VLM at runtime and is checked only on candidate events"
        )

    explanation = (
        f"{object_class} · {trigger_type}"
        + (f" ≥{duration_seconds:.0f}s" if duration_seconds else "")
        + (f" >{threshold}" if threshold is not None else "")
        + (f" · visual: {visual_predicate['description']}" if visual_predicate else "")
        + " → create_notification (draft, disabled until you approve)"
    )
    return CompileResult(True, draft=draft, warnings=warnings, explanation=explanation)


def _suggest_name(object_class: str, trigger_type: str, visual: dict | None) -> str:
    base = {
        "object_entered_zone": f"{object_class} entered zone",
        "object_exited_zone": f"{object_class} exited zone",
        "dwell_in_zone": f"{object_class} dwell in zone",
        "object_count_exceeded": f"{object_class} count exceeded",
    }.get(trigger_type, f"{object_class} rule")
    if visual:
        base += f" + verify {visual['description']}"
    return base[:255]
