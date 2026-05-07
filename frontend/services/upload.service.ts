import type { Dataset } from '@/types/dataset';
import { generateId, sleep } from '@/lib/utils';

export interface UploadedFile {
  id: string;
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'complete' | 'error';
  dataset?: Dataset;
  error?: string;
}

function inferType(name: string): Dataset['type'] {
  const lower = name.toLowerCase();
  if (lower.includes('survey') || lower.includes('feedback')) return 'Survey';
  if (lower.includes('report') || lower.includes('doc')) return 'Document';
  return 'Operational';
}

export const uploadService = {
  async uploadFile(
    file: File,
    onProgress: (progress: number) => void,
  ): Promise<Dataset> {
    // Simulate chunked upload progress
    for (let p = 10; p <= 90; p += 20) {
      await sleep(300);
      onProgress(p);
    }
    await sleep(400);
    onProgress(100);

    const dataset: Dataset = {
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

    return dataset;
  },
};
