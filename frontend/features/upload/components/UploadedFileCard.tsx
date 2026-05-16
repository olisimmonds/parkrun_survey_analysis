'use client';

import { CheckCircle, XCircle, FileText, X, Loader2 } from 'lucide-react';
import { cn, formatBytes } from '@/lib/utils';
import type { UploadedFile } from '@/services/upload.service';
import { useUploadStore } from '@/store/uploadStore';

interface Props {
  upload: UploadedFile;
}

const STAGE_LABEL: Record<string, string> = {
  classify: 'Classifying questions…',
  store: 'Storing responses…',
  embed: 'Computing embeddings…',
  cluster: 'Clustering themes…',
  wiki_update: 'Building knowledge base…',
  done: 'Complete',
  processing: 'Processing…',
};

const statusConfig = {
  pending: { label: 'Queued', color: 'text-gray-400' },
  uploading: { label: 'Uploading…', color: 'text-parkrun-dark' },
  complete: { label: 'Complete', color: 'text-green-600' },
  error: { label: 'Error', color: 'text-red-500' },
};

export default function UploadedFileCard({ upload }: Props) {
  const removeUpload = useUploadStore((s) => s.removeUpload);
  const { file, progress, status, stage, error } = upload;
  const baseLabel = statusConfig[status].label;
  const { color } = statusConfig[status];
  const label = status === 'uploading' && stage
    ? (STAGE_LABEL[stage] ?? baseLabel)
    : baseLabel;

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-gray-50">
          {status === 'uploading' ? (
            <Loader2 size={20} className="animate-spin text-parkrun-dark" />
          ) : status === 'complete' ? (
            <CheckCircle size={20} className="text-green-500" />
          ) : status === 'error' ? (
            <XCircle size={20} className="text-red-500" />
          ) : (
            <FileText size={20} className="text-gray-400" />
          )}
        </div>

        {/* File info */}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-gray-800">
            {file.name}
          </p>
          <p className="mt-0.5 text-xs text-gray-400">{formatBytes(file.size)}</p>

          {/* Progress bar */}
          {(status === 'uploading' || status === 'complete') && (
            <div className="mt-2">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-300',
                    status === 'complete'
                      ? 'bg-green-500'
                      : 'bg-parkrun-bright',
                  )}
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {error && (
            <p className="mt-1 text-xs text-red-500">{error}</p>
          )}
        </div>

        {/* Status + dismiss */}
        <div className="flex flex-shrink-0 flex-col items-end gap-1">
          <span className={cn('text-xs font-medium', color)}>{label}</span>
          {status !== 'uploading' && (
            <button
              onClick={() => removeUpload(upload.id)}
              className="text-gray-300 hover:text-gray-500"
              aria-label="Remove"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
