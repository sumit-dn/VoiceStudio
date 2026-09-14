/**
 * TranscriptionDropzone — drop (or pick) an audio file, get a transcript.
 *
 * The Transcriptions page could only ever be filled by the microphone: you
 * held the dictation hotkey and spoke. A file already on disk — a recording, a
 * voice note, an interview — had no way in, even though the backend's
 * job-free `POST /transcribe` has always accepted exactly that.
 *
 * Files are posted one at a time on purpose. Transcription is GPU-pool work
 * (see `run_transcribe_guarded`), so firing a whole dropped folder at once
 * would queue behind itself and starve TTS; a serial queue also lets the user
 * watch progress file by file and keep partial results when one fails.
 *
 * Each success is written through `onTranscribed` into the same localStorage
 * store the dictation path uses, so drops and dictations share one history,
 * one search box and one export.
 */
import React, { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Upload, Loader2, FileAudio } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { API, apiFetch } from '../api/client';
import { asrMissingPayload, toastAsrModelMissing } from '../utils/asrModelMissing';
import { Button } from '../ui';

/** Containers the backend can decode. The sidecars normalise to 16 kHz mono
 *  themselves (via ffmpeg when libsndfile can't read the container), so this
 *  list only has to keep obviously-wrong files out of a slow round trip. */
const ACCEPT = '.wav,.mp3,.m4a,.mp4,.aac,.flac,.ogg,.opus,.webm,.wma,.aiff,.aif';

const AUDIO_EXT = new Set(
  ACCEPT.split(',').map((e) => e.trim().replace(/^\./, '').toLowerCase()),
);

/** True when the browser calls it audio/video, or the extension is one we list.
 *
 * The MIME type alone is not enough: a file dragged from some file managers
 * arrives with an empty `type`, and `.opus`/`.m4a` are commonly untyped. */
export function isProbablyAudio(file) {
  if (!file) return false;
  const type = (file.type || '').toLowerCase();
  if (type.startsWith('audio/') || type.startsWith('video/')) return true;
  const ext = (file.name || '').split('.').pop()?.toLowerCase();
  return !!ext && AUDIO_EXT.has(ext);
}

export default function TranscriptionDropzone({ onTranscribed, disabled = false }) {
  const { t } = useTranslation();
  const [dragging, setDragging] = useState(false);
  const [queue, setQueue] = useState({ done: 0, total: 0, name: '' });
  const inputRef = useRef(null);
  // Drag events fire on every child element, so a naive boolean flickers as the
  // pointer crosses the icon and the label. Counting enter/leave pairs keeps
  // the highlight steady for as long as the pointer is anywhere inside.
  const dragDepth = useRef(0);

  const busy = queue.total > 0;
  const inert = disabled || busy;

  const transcribeOne = useCallback(async (file) => {
    const fd = new FormData();
    fd.append('audio', file, file.name);
    // No `mode`: the default ('fast') routes to the capture backend, i.e. the
    // engine the user actually selected for dictation. 'accurate' looked like
    // a free upgrade — word timings for a file you read rather than paste —
    // but it ignores that selection entirely and resolves the global offline
    // engine instead. On a machine set to IndicConformer that silently
    // transcribed with Whisper large-v3, loading ~6 GB of weights in-process
    // to do it. A dropped file goes through the same engine as the mic.
    const res = await apiFetch(`${API}/transcribe`, { method: 'POST', body: fd });
    return res.json();
  }, []);

  const handleFiles = useCallback(
    async (fileList) => {
      const files = Array.from(fileList || []).filter(isProbablyAudio);
      if (!files.length) {
        toast.error(t('transcriptions.drop_not_audio'));
        return;
      }

      let ok = 0;
      for (let i = 0; i < files.length; i += 1) {
        const file = files[i];
        setQueue({ done: i, total: files.length, name: file.name });
        try {
          const json = await transcribeOne(file);
          const text = (json?.text || '').trim();
          if (!text) {
            // The engine ran and returned nothing. That is a real outcome for
            // silence or for a language the model does not cover, so name the
            // file and keep going rather than failing the whole drop.
            toast.error(t('transcriptions.drop_no_speech', { name: file.name }));
            continue;
          }
          onTranscribed?.({
            text,
            language: json.language || 'unknown',
            duration_s: json.duration_s || 0,
            segments: json.segments || [],
            source: file.name,
          });
          ok += 1;
        } catch (e) {
          // A TTS-only install answers 409 with a typed payload — render the
          // download CTA instead of a raw "409 Conflict".
          const missing = asrMissingPayload(e);
          if (missing) {
            toastAsrModelMissing(missing);
            break; // every remaining file would hit the same wall
          }
          toast.error(
            t('transcriptions.drop_failed', {
              name: file.name,
              error: e?.message || String(e),
            }),
          );
        }
      }

      setQueue({ done: 0, total: 0, name: '' });
      if (ok > 0) toast.success(t('transcriptions.drop_done', { count: ok }));
    },
    [onTranscribed, t, transcribeOne],
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      if (inert) return;
      handleFiles(e.dataTransfer?.files);
    },
    [handleFiles, inert],
  );

  const onDragEnter = useCallback(
    (e) => {
      e.preventDefault();
      if (inert) return;
      dragDepth.current += 1;
      setDragging(true);
    },
    [inert],
  );

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  }, []);

  const openPicker = useCallback(() => {
    if (!inert) inputRef.current?.click();
  }, [inert]);

  return (
    <div
      className={`flex flex-col items-center justify-center gap-[8px] rounded-[var(--radius-lg)] px-[20px] py-[18px] text-center [transition:border-color_0.15s,background-color_0.15s] ${
        dragging
          ? '[border:1px_dashed_var(--color-brand)] bg-bg-elev-1'
          : '[border:1px_dashed_var(--color-border)] bg-transparent'
      } ${inert ? 'opacity-60' : ''}`}
      onDrop={onDrop}
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      role="button"
      tabIndex={inert ? -1 : 0}
      aria-disabled={inert}
      aria-label={t('transcriptions.drop_title')}
      onClick={openPicker}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openPicker();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          handleFiles(e.target.files);
          // Clear it so picking the SAME file again still fires onChange.
          e.target.value = '';
        }}
      />

      {busy ? (
        <>
          <Loader2 size={22} className="animate-spin text-fg-muted" aria-hidden="true" />
          <p className="m-0 text-[var(--text-sm)] font-medium text-fg" role="status" aria-live="polite">
            {t('transcriptions.drop_progress', {
              done: queue.done + 1,
              total: queue.total,
              name: queue.name,
            })}
          </p>
        </>
      ) : (
        <>
          {dragging ? (
            <FileAudio size={22} className="text-brand" aria-hidden="true" />
          ) : (
            <Upload size={22} className="text-fg-subtle opacity-60" aria-hidden="true" />
          )}
          <p className="m-0 text-[var(--text-sm)] font-medium text-fg">
            {t('transcriptions.drop_title')}
          </p>
          <p className="m-0 max-w-[320px] text-[var(--text-xs)] leading-[1.6] text-fg-muted">
            {t('transcriptions.drop_desc')}
          </p>
          <Button size="sm" variant="ghost" disabled={inert} onClick={(e) => {
            e.stopPropagation();
            openPicker();
          }}>
            {t('transcriptions.drop_browse')}
          </Button>
        </>
      )}
    </div>
  );
}
