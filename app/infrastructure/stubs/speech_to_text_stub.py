from app.domain.models import AudioRequest, Transcript

# Canned questions, one per supported language, each phrased the way a technician would
# actually ask about the night-shift alarm entry in the fixture corpus (log-006 /
# log-ar-006). Picking real questions rather than placeholder text is what keeps the whole
# downstream path -- language detection, retrieval, citation validation -- genuinely
# exercisable from an audio upload.
_CANNED_TRANSCRIPTS: dict[str, str] = {
    "en": "What caused the fire alarm during the night shift?",
    "ar": "ما سبب إنذار الحريق في الوردية الليلية؟",
}

_DEFAULT_LANGUAGE = "en"

_MODEL_NAME = "faster-whisper-large-v3-turbo-stub"

# High but not certain, matching what a real ASR engine reports for clean speech.
_STUB_CONFIDENCE = 0.9


class SpeechToTextStub:
    """Returns a canned transcript. Stands in for faster-whisper-large-v3-turbo served
    via Cloudera AI Inference.

    It does not decode the audio at all -- the bytes are ignored. The language hint is
    the only thing it reads, and only to choose which canned question to hand back, so an
    Arabic upload still flows end-to-end into Arabic retrieval.

    `duration_seconds` stays None: deriving it from the byte count would mean assuming a
    PCM sample rate that is wrong for every compressed format. An absent number is
    honest; the real adapter reports the decoder's own figure.
    """

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        hinted = audio.language_hint.code if audio.language_hint else _DEFAULT_LANGUAGE
        language = hinted if hinted in _CANNED_TRANSCRIPTS else _DEFAULT_LANGUAGE
        return Transcript(
            text=_CANNED_TRANSCRIPTS[language],
            language=language,
            confidence=_STUB_CONFIDENCE,
            duration_seconds=None,
            model=_MODEL_NAME,
        )
