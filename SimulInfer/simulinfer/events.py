"""The public translation protocol; inference diagnostics stay in the session."""
_FIELDS = {
    'ready': {'model_type', 'direction', 'source_text_enabled', 'tts_enabled',
              'response_mode', 'sample_rate', 'first_chunk_ms', 'chunk_ms',
              'audio_transport', 'input_audio', 'tts', 'demo_only'},
    'chunk': {'index', 'start_ms', 'end_ms', 'action', 'text', 'source_text', 'source_text_full', 'demo_only'},
    'flush': {'start_ms', 'end_ms', 'action', 'text', 'source_text', 'source_text_full', 'demo_only'},
    'done': {'model_type', 'direction', 'response_mode', 'full_text', 'source_text',
             'duration_ms', 'num_chunks', 'tts_enabled', 'tts_audio', 'input_audio',
             'unresolved_audio_ms', 'skipped_untranslated_audio_ms', 'demo_only'},
    'audio': {'seq', 'sample_rate', 'format', 'channels', 'data', 'bytes', 'encoding'},
    'tts_start': {'seq', 'text', 'sample_rate', 'format', 'channels'},
    'tts_end': {'seq', 'audio_bytes'},
    'tts_done': {'audio_bytes', 'sample_rate', 'format', 'channels', 'complete', 'failed_segments'},
    'tts_error': {'seq', 'partial', 'audio_bytes', 'detail'},
    'voice_clone': {'revision', 'reference_seconds', 'source_end_ms'},
    'tts_unavailable': {'detail'},
    'input_progress': {'received_ms'},
    'error': {'detail'},
}
_TTS_FIELDS = {'sample_rate', 'format', 'channels', 'clone_source',
               'continuity', 'history_window_s'}


def public_event(event):
    kind = event.get('event')
    if kind not in _FIELDS:
        return None
    result = {'event': kind}
    result.update((key, value) for key, value in event.items() if key in _FIELDS[kind])
    if isinstance(result.get('tts'), dict):
        result['tts'] = {k: v for k, v in result['tts'].items() if k in _TTS_FIELDS}
    if kind == 'done':
        result['complete'] = event.get('complete',True) is not False and not any(event.get(k) for k in (
            'unresolved_audio_ms', 'skipped_untranslated_audio_ms', 'empty_st_reset_count'))
    return result
