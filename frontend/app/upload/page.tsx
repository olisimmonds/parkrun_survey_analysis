'use client';

import { useUploadStore } from '@/store/uploadStore';
import FileDropzone from '@/features/upload/components/FileDropzone';
import UploadedFileCard from '@/features/upload/components/UploadedFileCard';
import { Trash2 } from 'lucide-react';

export default function UploadPage() {
  const { uploads, clearCompleted } = useUploadStore();
  const hasCompleted = uploads.some((u) => u.status === 'complete');

  return (
    <div className="mx-auto max-w-2xl px-4 py-10 sm:px-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Data</h1>
        <p className="mt-1 text-sm text-gray-500">
          Add survey exports, operational data, or document files to your
          insights library.
        </p>
      </div>

      {/* Dropzone */}
      <FileDropzone />

      {/* Upload queue */}
      {uploads.length > 0 && (
        <div className="mt-6">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-semibold text-gray-700">
              Uploads ({uploads.length})
            </p>
            {hasCompleted && (
              <button
                onClick={clearCompleted}
                className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600"
              >
                <Trash2 size={12} />
                Clear completed
              </button>
            )}
          </div>
          <div className="flex flex-col gap-2">
            {uploads.map((upload) => (
              <UploadedFileCard key={upload.id} upload={upload} />
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
