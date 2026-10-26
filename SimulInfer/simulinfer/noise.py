"""Compact VAD-selected source spans while retaining source-time metadata."""
import numpy as np


def mask_spans(mask,start_sample=0):
    mask=np.asarray(mask,dtype=np.bool_)
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    return [(start_sample+int(a),start_sample+int(b)) for a,b in zip(edges[::2],edges[1::2])]


def merge_spans(spans):
    merged=[]
    for start,end in sorted(spans):
        if end<=start:continue
        if merged and start<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else:merged.append((start,end))
    return merged


def padded_st_audio(blocks,pad_samples,trailing_samples=None):
    """Pad detected source segments across contiguous chunk boundaries.

    Each block retains its original waveform and raw VAD spans. Never insert
    unavailable samples across source gaps, nor duplicate overlapping padding.
    """
    trailing_samples=pad_samples if trailing_samples is None else trailing_samples
    groups=[]
    for block in blocks:
        if not groups or groups[-1][-1][0]+groups[-1][-1][2]!=block[0]:groups.append([])
        groups[-1].append(block)
    audios=[];kept=[]
    for group in groups:
        start=group[0][0];end=group[-1][0]+group[-1][2]
        original=np.concatenate([b[5]['original'] for b in group])
        if len(original)!=end-start:raise ValueError('ST原始音频与源时间轴长度不一致')
        spans=merge_spans([span for b in group for span in b[5]['speech_spans']])
        expanded=merge_spans([(max(start,a-pad_samples),min(end,b+trailing_samples)) for a,b in spans])
        kept.extend(expanded)
        audios.extend(original[a-start:b-start] for a,b in expanded)
    return (np.concatenate(audios) if audios else np.empty(0,dtype=np.float32)),kept


async def detect_st_blocks(blocks,detector):
    """Re-detect original ST audio, without joining across source-time gaps."""
    import asyncio
    groups=[]
    for block in blocks:
        if not groups or groups[-1][-1][0]+groups[-1][-1][2]!=block[0]:groups.append([])
        groups[-1].append(block)
    detected=[];reports=[]
    for group in groups:
        start=group[0][0]
        pieces=[b[5]['original'] if len(b)>5 else b[1][:b[2]] for b in group]
        audio=np.concatenate(pieces)
        if len(audio)!=sum(b[2] for b in group):raise ValueError('ST original audio length mismatch')
        result=await asyncio.to_thread(detector.analyze,audio)
        reports.append(result)
        spans=merge_spans([(start+max(0,min(len(audio),round(a*16000))),
                            start+max(0,min(len(audio),round(b*16000))))
                           for a,b in result['st_vad_segments']])
        for block,piece in zip(group,pieces):
            left=block[0];right=left+block[2]
            selected=[(max(a,left),min(b,right)) for a,b in spans if b>left and a<right]
            detected.append((*block[:4],block[4] if len(block)>4 else None,
                             {'original':piece,'speech_spans':selected}))
    stats=speech_stats(detected)
    stats.update(st_vad_mode='firered_nonstream',st_vad_accepted=bool(stats['st_vad_segments']),
                 st_vad_audio_ms=sum(b[2] for b in blocks)/16,
                 st_vad_threshold=reports[0].get('st_vad_threshold') if reports else None,
                 st_vad_probability_max=max((r.get('st_vad_probability_max',0) for r in reports),default=0),
                 st_vad_seconds=sum(r.get('st_vad_seconds',0) for r in reports))
    return detected,stats


def short_speech_policy(stats,direction,has_long_chunk=False,trigger='eof'):
    """Direction and continuity-aware policy applied only to new target text."""
    speech=float(stats['st_vad_speech_ms']);longest=float(stats['st_vad_longest_speech_ms'])
    ratio=longest/speech if speech else 0
    reliable=has_long_chunk or longest>=1000
    applies=not reliable and trigger in {'audible_wait','trailing_silence','eof','unknown'}
    effective=longest+.5*max(0,speech-longest)
    limit=None;bonus=0
    if applies and effective<=1500:
        index=0 if effective<=300 else 1 if effective<=500 else 2 if effective<=800 else 3 if effective<=1100 else 4
        limit=((4,4),(5,6),(7,7),(8,9),(12,9))[index][direction=='zh-en']
        if longest>=800 and ratio>=.70:bonus=3 if direction=='zh-en' else 2
        elif longest>=500 and ratio>=.80:bonus=2 if direction=='zh-en' else 1
        limit+=bonus
    return {'st_short_speech_guard':True,'st_short_speech_guard_applied':applies,
            'st_has_long_speech_chunk':has_long_chunk,'st_has_reliable_continuous_speech':reliable,
            'st_speech_continuity_ratio':round(ratio,6),'st_short_speech_segments':len(stats['st_vad_segments']),
            'st_short_speech_effective_ms':round(effective,3),
            'st_short_speech_base_token_limit':limit-bonus if limit is not None else None,
            'st_short_speech_continuity_bonus':bonus,'st_short_speech_token_limit':limit}


def speech_stats(blocks):
    spans=merge_spans([span for b in blocks for span in b[5]['speech_spans']])
    lengths=[(b-a)/16 for a,b in spans]
    return {'st_vad_speech_ms':sum(lengths),'st_vad_longest_speech_ms':max(lengths,default=0),
            'st_vad_segments':[[a/16000,b/16000] for a,b in spans]}


def speech_audio(samples, mask, *, start_sample, leading_samples=0):
    samples=np.asarray(samples,dtype=np.float32)
    mask=np.asarray(mask,dtype=np.bool_).copy()
    if mask.shape!=samples.shape:
        raise ValueError('VAD mask与真实输入采样长度不一致')
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    for start in starts:
        mask[max(0,int(start)-leading_samples):start]=True
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    spans=[(start_sample+int(a),start_sample+int(b)) for a,b in zip(edges[::2],edges[1::2])]
    return samples[mask].copy(),spans
