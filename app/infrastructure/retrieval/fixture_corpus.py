from typing import Any, NamedTuple


class FixtureDocument(NamedTuple):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    # Declared last with a default so existing positional construction keeps working
    # (NamedTuple also forbids a defaulted field before a non-defaulted one, and
    # `metadata` has no default).
    language: str = "en"


# Sample logbook entries standing in for a real ingested corpus. Once a document
# ingestion pipeline exists (Bronze -> Silver -> Iceberg Golden per CLAUDE.md), the
# keyword retriever reads from that index instead and this fixture is retired.
ENGLISH_FIXTURE_CORPUS: list[FixtureDocument] = [
    FixtureDocument(
        chunk_id="log-001",
        document_id="doc-log-001",
        text=(
            "Morning shift equipment check completed on Line 2 conveyor belt. No anomalies "
            "detected. All safety guards in place and lubrication levels nominal."
        ),
        metadata={"source_title": "Morning Shift Equipment Log"},
        language="en",
    ),
    FixtureDocument(
        chunk_id="log-002",
        document_id="doc-log-002",
        text=(
            "Incident report: minor slip near the loading dock at 14:32. No injuries "
            "reported. Area cordoned off and wet floor signage placed pending cleanup."
        ),
        metadata={"source_title": "Incident Report - Loading Dock"},
        language="en",
    ),
    FixtureDocument(
        chunk_id="log-003",
        document_id="doc-log-003",
        text=(
            "Shift handover notes from night crew to day crew: Boiler 3 pressure reading "
            "stable at 145 psi. No outstanding maintenance requests."
        ),
        metadata={"source_title": "Shift Handover Notes"},
        language="en",
    ),
    FixtureDocument(
        chunk_id="log-004",
        document_id="doc-log-004",
        text=(
            "Safety walk conducted across warehouse zone B. Fire extinguishers inspected "
            "and tagged current. Emergency exit routes clear of obstructions."
        ),
        metadata={"source_title": "Safety Walk Record - Zone B"},
        language="en",
    ),
    FixtureDocument(
        chunk_id="log-005",
        document_id="doc-log-005",
        text=(
            "Scheduled maintenance log: replaced worn drive belt on packaging machine 4. "
            "Machine tested and returned to service at 09:15."
        ),
        metadata={"source_title": "Scheduled Maintenance Log"},
        language="en",
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
        language="en",
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
        language="en",
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
        language="en",
    ),
]


# Arabic counterparts, one per English entry, so retrieval accuracy can be measured
# independently per language over comparable content. Written as natural Modern Standard
# Arabic rather than pre-normalized text: diacritics, hamza variants and ta marbuta in
# the source are what prove the normalizer earns its place. Digits stay ASCII and
# technical acronyms (HVAC, psi) stay Latin, as a technician would actually write them.
ARABIC_FIXTURE_CORPUS: list[FixtureDocument] = [
    FixtureDocument(
        chunk_id="log-ar-001",
        document_id="doc-log-ar-001",
        text=(
            "اكتمل فحص المعدات في وردية الصباح على الحزام الناقل في الخط الثاني. لم يتم "
            "رصد أي أعطال. جميع حواجز السلامة في مكانها ومستويات التزييت طبيعية."
        ),
        metadata={"source_title": "سجل معدات وردية الصباح"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-002",
        document_id="doc-log-ar-002",
        text=(
            "تقرير حادث: انزلاق بسيط قرب رصيف التحميل في الساعة 14:32. لم تسجل أي "
            "إصابات. تم تطويق المنطقة ووضع لافتات تحذير من الأرضية المبللة بانتظار التنظيف."
        ),
        metadata={"source_title": "تقرير حادث - رصيف التحميل"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-003",
        document_id="doc-log-ar-003",
        text=(
            "ملاحظات تسليم الوردية من الطاقم الليلي إلى الطاقم النهاري: قراءة ضغط الغلاية "
            "رقم 3 مستقرة عند 145 psi. لا توجد طلبات صيانة معلقة."
        ),
        metadata={"source_title": "ملاحظات تسليم الوردية"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-004",
        document_id="doc-log-ar-004",
        text=(
            "جولة سلامة في منطقة المستودع ب. تم فحص طفايات الحريق ووسمها بتاريخ ساري. "
            "مسارات مخارج الطوارئ خالية من العوائق."
        ),
        metadata={"source_title": "سجل جولة السلامة - المنطقة ب"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-005",
        document_id="doc-log-ar-005",
        text=(
            "سجل صيانة مجدولة: تم استبدال حزام الإدارة المهترئ في ماكينة التغليف رقم 4. "
            "تم اختبار الماكينة وإعادتها إلى الخدمة في الساعة 09:15."
        ),
        metadata={"source_title": "سجل الصيانة المجدولة"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-006",
        document_id="doc-log-ar-006",
        text=(
            "انطلق إنذار الحَرِيق في الساعة 02:47 أثناء الوردية الليلية بسبب دخان تم رصده "
            "قرب غرفة الخوادم. تم إبلاغ فرقة الإطفاء، وتأكد أنه إنذار كاذب بعد المعاينة، "
            "وسببه تراكم الغبار."
        ),
        metadata={"source_title": "تقرير إنذار الوردية الليلية"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-007",
        document_id="doc-log-ar-007",
        text=(
            "قيد في سجل الزوار: وصل فريق المقاولين لفحص HVAC في الساعة 10:00، برفقة موظفي "
            "المرافق، وغادر في الساعة 11:30 بعد إتمام استبدال الفلاتر."
        ),
        metadata={"source_title": "سجل الزوار والمقاولين"},
        language="ar",
    ),
    FixtureDocument(
        chunk_id="log-ar-008",
        document_id="doc-log-ar-008",
        text=(
            "تقرير وشك وقوع حادث: تفادى سائق الرافعة الشوكية بصعوبة الاصطدام بأحد المشاة "
            "في الممر رقم 7. يوصى بتركيب مرايا إضافية وإقامة تدريب تنشيطي."
        ),
        metadata={"source_title": "تقرير وشك وقوع حادث - الممر 7"},
        language="ar",
    ),
]


# The index the service actually loads. Kept as one list because a single BM25 index is
# what OpenSearch will do; the two source lists stay separately importable so tests can
# build an English-only index and pin the English ranking baseline against it.
DEFAULT_FIXTURE_CORPUS: list[FixtureDocument] = [
    *ENGLISH_FIXTURE_CORPUS,
    *ARABIC_FIXTURE_CORPUS,
]
