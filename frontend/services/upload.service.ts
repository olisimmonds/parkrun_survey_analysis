import type { Dataset } from '@/types/dataset';
import { defaultConfig } from '@/types/config';
import { generateId, sleep } from '@/lib/utils';

export interface UploadedFile {
  id: string;
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'complete' | 'error';
  stage?: string;
  dataset?: Dataset;
  error?: string;
}

interface UploadOut {
  jobId: string;
  surveyId: string;
  message: string;
}

interface JobStatusOut {
  jobId: string;
  surveyId: string;
  stage: string;
  status: string;
  progress: number;
  error?: string | null;
}

function inferType(name: string): Dataset['type'] {
  const lower = name.toLowerCase();
  if (lower.includes('survey') || lower.includes('feedback')) return 'Survey';
  if (lower.includes('report') || lower.includes('doc')) return 'Document';
  return 'Operational';
}

async function pollJobStatus(
  jobId: string,
  onProgress: (progress: number, stage: string) => void,
  signal: AbortSignal,
): Promise<JobStatusOut> {
  const baseUrl = defaultConfig.apiBaseUrl;
  const maxAttempts = 180; // 15 minutes at 5s intervals

  for (let i = 0; i < maxAttempts; i++) {
    if (signal.aborted) throw new Error('Upload cancelled.');

    await sleep(5000);
    if (signal.aborted) throw new Error('Upload cancelled.');

    const res = await fetch(`${baseUrl}/api/ingest/status/${jobId}`, { signal });
    if (!res.ok) throw new Error(`Status poll failed: ${res.status}`);

    const status: JobStatusOut = await res.json();
    onProgress(status.progress, status.stage);

    if (status.status === 'done' || status.stage === 'done') return status;
    if (status.status === 'failed') {
      throw new Error(status.error ?? 'Processing failed.');
    }
  }

  throw new Error('Processing timed out after 15 minutes.');
}

export const uploadService = {
  async uploadFile(
    file: File,
    onProgress: (progress: number, stage?: string) => void,
  ): Promise<Dataset> {
    if (defaultConfig.mockMode) {
      // Mock path retained for development without a running backend.
      for (let p = 10; p <= 90; p += 20) {
        await sleep(300);
        onProgress(p, 'processing');
      }
      await sleep(400);
      onProgress(100, 'done');
      return {
        id: `ds_${generateId()}`,
        name: file.name.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' '),
        type: inferType(file.name),
        uploadedAt: new Date().toISOString(),
        size: file.size,
        rowCount: Math.floor(Math.random() * 2000) + 100,
        columnCount: Math.floor(Math.random() * 20) + 5,
        description: '',
        tags: [],
        status: 'ready',
        fileName: file.name,
      };
    }

    // ── Real upload ───────────────────────────────────────────────────────
    const baseUrl = defaultConfig.apiBaseUrl;
    const abortController = new AbortController();

    // Stage 1: POST file
    onProgress(5);
    const form = new FormData();
    form.append('file', file);
    form.append('survey_type', inferType(file.name).toLowerCase());

    let uploadRes: Response;
    try {
      uploadRes = await fetch(`${baseUrl}/api/ingest/upload`, {
        method: 'POST',
        body: form,
        signal: abortController.signal,
      });
    } catch {
      throw new Error(
        `Cannot reach the backend at ${baseUrl}. ` +
        `Make sure the API server is running: cd backend && uvicorn app.main:app --port 8000`
      );
    }

    if (!uploadRes.ok) {
      const detail = await uploadRes.json().catch(() => ({}));
      throw new Error((detail as { detail?: string }).detail ?? `Upload failed: ${uploadRes.status}`);
    }

    const { jobId, surveyId }: UploadOut = await uploadRes.json();
    onProgress(15);

    // Stage 2: Poll job to completion
    await pollJobStatus(jobId, (progress, stage) => onProgress(progress, stage), abortController.signal);

    // Stage 3: Fetch the completed dataset record
    const datasetRes = await fetch(`${baseUrl}/api/datasets/${surveyId}`);
    if (!datasetRes.ok) throw new Error('Failed to fetch completed dataset.');
    const dataset: Dataset = await datasetRes.json();

    onProgress(100);
    return dataset;
  },
};
