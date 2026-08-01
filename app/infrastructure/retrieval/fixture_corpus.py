from typing import Any, NamedTuple


class FixtureDocument(NamedTuple):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


# Sample logbook entries standing in for a real ingested corpus. Once a document
# ingestion pipeline exists (Bronze -> Silver -> Iceberg Golden per CLAUDE.md), the
# keyword retriever reads from that index instead and this fixture is retired.
DEFAULT_FIXTURE_CORPUS: list[FixtureDocument] = [
    FixtureDocument(
        chunk_id="log-001",
        document_id="doc-log-001",
        text=(
            "Morning shift equipment check completed on Line 2 conveyor belt. No anomalies "
            "detected. All safety guards in place and lubrication levels nominal."
        ),
        metadata={"source_title": "Morning Shift Equipment Log"},
    ),
    FixtureDocument(
        chunk_id="log-002",
        document_id="doc-log-002",
        text=(
            "Incident report: minor slip near the loading dock at 14:32. No injuries "
            "reported. Area cordoned off and wet floor signage placed pending cleanup."
        ),
        metadata={"source_title": "Incident Report - Loading Dock"},
    ),
    FixtureDocument(
        chunk_id="log-003",
        document_id="doc-log-003",
        text=(
            "Shift handover notes from night crew to day crew: Boiler 3 pressure reading "
            "stable at 145 psi. No outstanding maintenance requests."
        ),
        metadata={"source_title": "Shift Handover Notes"},
    ),
    FixtureDocument(
        chunk_id="log-004",
        document_id="doc-log-004",
        text=(
            "Safety walk conducted across warehouse zone B. Fire extinguishers inspected "
            "and tagged current. Emergency exit routes clear of obstructions."
        ),
        metadata={"source_title": "Safety Walk Record - Zone B"},
    ),
    FixtureDocument(
        chunk_id="log-005",
        document_id="doc-log-005",
        text=(
            "Scheduled maintenance log: replaced worn drive belt on packaging machine 4. "
            "Machine tested and returned to service at 09:15."
        ),
        metadata={"source_title": "Scheduled Maintenance Log"},
    ),
    FixtureDocument(
        chunk_id="log-006",
        document_id="doc-log-006",
        text=(
            "Fire alarm triggered at 02:47 during the night shift due to smoke detected "
            "near the server room. Fire brigade notified; false alarm confirmed after "
            "inspection, caused by dust buildup."
        ),
        metadata={"source_title": "Night Shift Alarm Report"},
    ),
    FixtureDocument(
        chunk_id="log-007",
        document_id="doc-log-007",
        text=(
            "Visitor log entry: contractor team arrived for HVAC inspection at 10:00, "
            "escorted by facilities staff, departed 11:30 after completing filter "
            "replacement."
        ),
        metadata={"source_title": "Visitor and Contractor Log"},
    ),
    FixtureDocument(
        chunk_id="log-008",
        document_id="doc-log-008",
        text=(
            "Near-miss report: forklift operator narrowly avoided collision with a "
            "pedestrian in aisle 7. Recommend additional mirror installation and "
            "refresher training."
        ),
        metadata={"source_title": "Near-Miss Report - Aisle 7"},
    ),
]
