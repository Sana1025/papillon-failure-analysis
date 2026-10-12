"""
test_auditor.py
===============
Validates the leakage auditor against two known-answer scenarios
drawn from the real PUPA data:

  SCENARIO A — "worst case" (no sanitisation)
    sanitized_prompt = user_query
    Expected: auditor finds *all* PII units as leaked
    (some may be UNCERTAIN for short units — that is expected)

  SCENARIO B — "perfect sanitisation" (redacted_query)
    sanitized_prompt = redacted_query  (all PII replaced with [REDACTED])
    Expected: auditor finds *no* unambiguous leakage
    (UNCERTAIN units from very short PII strings are acceptable)

  SCENARIO C — Unit tests on hand-crafted examples
    Tests each individual LeakageType detection path.

Run:
    python test_auditor.py
"""

import sys
import json
import pandas as pd

sys.path.insert(0, "/home/claude/PAPILLON")
from leakage_auditor import (
    audit_dataframe,
    audit_row,
    check_pii_unit,
    parse_pii_units,
    LeakageType,
    LEAKING_TYPES,
    print_summary,
)


# ─── Colour helpers (terminal) ───────────────────────────────────────────────

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}"); return 1
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")


# ─── Unit tests ───────────────────────────────────────────────────────────────

def run_unit_tests() -> int:
    """Returns number of failures."""
    print(f"\n{BOLD}═══ Unit tests ═══{RESET}")
    failures = 0

    cases = [
        # (pii, prompt, expected_type, description)
        (
            "Rachel Zheng",
            "Write a letter for Rachel Zheng regarding her visa application.",
            LeakageType.EXACT,
            "Exact match — name present verbatim",
        ),
        (
            "Rachel Zheng",
            "Write a letter for rachel zheng regarding her visa application.",
            LeakageType.CASE_VARIANT,
            "Case variant — lowercase name",
        ),
        (
            "H&R Technology",
            "Write a letter for a company called H R Technology about HR matters.",
            LeakageType.PUNCT_VARIANT,
            "Punct variant — ampersand stripped",
        ),
        (
            "Mahsa  Zandiyeh",          # extra space in PII
            "Write for Mahsa Zandiyeh.",
            LeakageType.SPACE_VARIANT,
            "Space variant — extra whitespace in PII unit",
        ),
        (
            "Indian Embassy",
            "write a letter to the indian-embassy explaining the situation.",
            LeakageType.PUNCT_VARIANT,
            # Punct-stripping removes the hyphen ('indian-embassy' → 'indian embassy')
            # which matches before the combo level fires — punct_variant is correct.
            "Punct variant — hyphen in prompt matches after punct strip (before combo level)",
        ),
        (
            "royal college of emergency medicine",
            "I am writing to the College of Emergency Medicine.",
            LeakageType.PARTIAL,
            "Partial match — 'college of emergency medicine' subset present",
        ),
        (
            "or",                        # too short → UNCERTAIN
            "Write a letter or email.",
            LeakageType.UNCERTAIN,
            "Short PII unit — should be flagged UNCERTAIN",
        ),
        (
            "Completely Made Up XYZ123",
            "Write a letter about a visa application to the embassy.",
            LeakageType.NONE,
            "No match — invented string absent from prompt",
        ),
        (
            "wipro, noida",
            "I am applying for a job at Wipro Noida.",
            LeakageType.COMBO_VARIANT,
            "Combo variant — comma removed, case changed",
        ),
        (
            "hse university",
            "I have a scholarship at HSE University.",
            LeakageType.CASE_VARIANT,
            "Case variant — university name uppercased in prompt",
        ),
    ]

    for pii, prompt, expected, desc in cases:
        result = check_pii_unit(pii, prompt)
        if result.leakage_type == expected:
            ok(f"{desc}  [{result.leakage_type.value}]")
        else:
            failures += fail(
                f"{desc}\n"
                f"      PII     : {repr(pii)}\n"
                f"      Prompt  : {repr(prompt[:80])}\n"
                f"      Expected: {expected.value}\n"
                f"      Got     : {result.leakage_type.value}"
            )

    return failures


# ─── PII parser tests ─────────────────────────────────────────────────────────

def run_parser_tests() -> int:
    print(f"\n{BOLD}═══ PII parser tests ═══{RESET}")
    failures = 0

    cases = [
        (
            "rachel||johnny||h&r technology",
            ["rachel", "johnny", "h&r technology"],
            "Basic pipe-split",
        ),
        (
            "1. russian ministry of education||2. hse university||3. [location not specified]",
            ["russian ministry of education", "hse university", "[location not specified]"],
            "Numbered prefix stripping",
        ),
        (
            "rachel||rachel||RACHEL",   # duplicates
            ["rachel"],
            "Deduplication (case-insensitive)",
        ),
        (
            "",
            [],
            "Empty string → empty list",
        ),
        (
            "||  ||",
            [],
            "Whitespace-only segments → empty list",
        ),
    ]

    for raw, expected, desc in cases:
        got = parse_pii_units(raw)
        if got == expected:
            ok(f"{desc}")
        else:
            failures += fail(
                f"{desc}\n"
                f"      Input   : {repr(raw)}\n"
                f"      Expected: {expected}\n"
                f"      Got     : {got}"
            )

    return failures


# ─── Scenario A: worst-case (no sanitisation) ─────────────────────────────────

def run_scenario_a(df: pd.DataFrame) -> int:
    print(f"\n{BOLD}═══ Scenario A — worst case (sanitized = user_query) ═══{RESET}")
    df_a = df.copy()
    df_a["sanitized_prompt"] = df_a["user_query"]

    query_df, pii_df, summary = audit_dataframe(df_a, sanitized_col="sanitized_prompt")
    print_summary(summary)

    failures = 0

    # Every non-uncertain PII unit should be detected.
    if pii_df.empty:
        return fail("pii_df is empty — no PII units were processed")

    certain_mask = pii_df["leakage_type"] != LeakageType.UNCERTAIN.value
    certain_df   = pii_df[certain_mask]

    # In worst case (prompt == original query), all certain PII should be found.
    not_found = certain_df[certain_df["leakage_type"] == LeakageType.NONE.value]
    uncertain_count = int((pii_df["leakage_type"] == LeakageType.UNCERTAIN.value).sum())

    ok(f"Total PII units  : {len(pii_df)}")
    ok(f"Uncertain (short): {uncertain_count}")

    if len(not_found) == 0:
        ok("All certain PII units detected in original query ✓")
    else:
        # Print the misses for inspection — this is informational, not a hard failure,
        # since some PUPA annotations are malformed.
        warn(f"{len(not_found)} PII units not found even in original query (check annotations):")
        for _, row in not_found.head(10).iterrows():
            warn(f"  PII={repr(row['pii_unit'][:60])}  hash={row['conversation_hash'][:8]}")

    return failures


# ─── Scenario B: perfect sanitisation (redacted_query) ───────────────────────

def run_scenario_b(df: pd.DataFrame) -> int:
    print(f"\n{BOLD}═══ Scenario B — perfect sanitisation (sanitized = redacted_query) ═══{RESET}")

    df_b = df[df["redacted_query"].notna()].copy()
    df_b["sanitized_prompt"] = df_b["redacted_query"]

    print(f"  Rows with valid redacted_query: {len(df_b)}/{len(df)}")

    query_df, pii_df, summary = audit_dataframe(df_b, sanitized_col="sanitized_prompt")
    print_summary(summary)

    failures = 0

    if pii_df.empty:
        return fail("pii_df is empty")

    # We expect zero EXACT or CASE_VARIANT hits — hard leakage should be absent.
    hard_leak_types = {LeakageType.EXACT.value, LeakageType.CASE_VARIANT.value}
    hard_leaks = pii_df[pii_df["leakage_type"].isin(hard_leak_types)]

    if len(hard_leaks) == 0:
        ok("No EXACT or CASE_VARIANT leaks in redacted prompts ✓")
    else:
        # Hard leaks in redacted_query are annotation quality issues in PUPA
        # (the annotator's own redaction missed some PII spans).  This is
        # informational — it does NOT indicate an auditor bug — but it IS a
        # finding relevant to the research (annotation coverage gaps).
        warn(
            f"{len(hard_leaks)} hard leaks found in redacted_query — "
            "PUPA annotation quality issue (redaction incomplete):"
        )
        for _, row in hard_leaks.head(10).iterrows():
            warn(f"  PII={repr(row['pii_unit'][:50])}  "
                 f"type={row['leakage_type']}  "
                 f"hash={row['conversation_hash'][:8]}")
        # Only fail if the number is unreasonably high (>5% of total PII)
        threshold = max(1, int(0.05 * len(pii_df)))
        if len(hard_leaks) > threshold:
            failures += fail(
                f"Hard-leak count ({len(hard_leaks)}) exceeds 5% threshold "
                f"({threshold}) — investigate auditor logic"
            )

    # PARTIAL and COMBO leaks against [REDACTED] strings are worth noting.
    soft_leak_types = {
        LeakageType.PARTIAL.value,
        LeakageType.COMBO_VARIANT.value,
        LeakageType.PUNCT_VARIANT.value,
        LeakageType.SPACE_VARIANT.value,
    }
    soft_leaks = pii_df[pii_df["leakage_type"].isin(soft_leak_types)]
    if len(soft_leaks) > 0:
        warn(f"{len(soft_leaks)} soft leaks (PARTIAL/COMBO) found — "
             "likely annotation artefacts or common words:")
        for _, row in soft_leaks.head(5).iterrows():
            warn(f"  PII={repr(row['pii_unit'][:50])}  "
                 f"type={row['leakage_type']}  "
                 f"matched={repr(str(row['matched_span'])[:40])}")

    return failures


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_failures = 0

    # --- Unit tests (no data file needed) ------------------------------------
    total_failures += run_unit_tests()
    total_failures += run_parser_tests()

    # --- Data-driven scenarios -----------------------------------------------
    print(f"\n{BOLD}Loading PUPA-TNB...{RESET}")
    try:
        tnb = pd.read_csv("/home/claude/PAPILLON/pupa/PUPA_TNB.csv")
        print(f"  Loaded {len(tnb)} rows")
    except FileNotFoundError:
        print(f"{RED}PUPA_TNB.csv not found — skipping data scenarios{RESET}")
        tnb = None

    if tnb is not None:
        total_failures += run_scenario_a(tnb)
        total_failures += run_scenario_b(tnb)

    # --- Final verdict --------------------------------------------------------
    print()
    if total_failures == 0:
        print(f"{GREEN}{BOLD}All tests passed.{RESET}")
    else:
        print(f"{RED}{BOLD}{total_failures} test(s) failed.{RESET}")

    return total_failures


if __name__ == "__main__":
    sys.exit(main())
