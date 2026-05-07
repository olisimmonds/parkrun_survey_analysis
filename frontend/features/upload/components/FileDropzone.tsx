'use client';

import { useCallback, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUploadStore } from '@/store/uploadStore';

const ACCEPTED = ['.csv', '.xlsx', '.xls', '.json', '.pdf'];

export default function FileDropzone() {
  const addFiles = useUploadStore((s) => s.addFiles);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files) return;
      addFiles(Array.from(files));
    },
    [addFiles],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    handleFiles(e.target.files);

  return (
    <label
      htmlFor="file-upload"
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={cn(
        'flex cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-8 py-16 text-center transition-all',
        isDragging
          ? 'border-parkrun-bright bg-parkrun-bright/5 scale-[1.01]'
          : 'border-gray-200 bg-white hover:border-parkrun-dark/40 hover:bg-gray-50',
      )}
    >
      <div
        className={cn(
          'flex h-16 w-16 items-center justify-center rounded-full transition-colors',
          isDragging ? 'bg-parkrun-bright/15' : 'bg-gray-100',
        )}
      >
        <UploadCloud
          size={32}
          className={isDragging ? 'text-parkrun-bright' : 'text-gray-400'}
        />
      </div>

      <div>
        <p className="text-base font-semibold text-gray-800">
          {isDragging ? 'Drop your files here' : 'Drag & drop files to upload'}
        </p>
        <p className="mt-1 text-sm text-gray-500">
          or{' '}
          <span className="font-medium text-parkrun-dark underline-offset-2 hover:underline">
            browse your computer
          </span>
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-2">
        {ACCEPTED.map((ext) => (
          <span
            key={ext}
            className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-500"
          >
            {ext.toUpperCase()}
          </span>
        ))}
      </div>

      <p className="text-xs text-gray-400">
        Multiple files supported · Max 50 MB per file
      </p>

      <input
        id="file-upload"
        type="file"
        multiple
        accept={ACCEPTED.join(',')}
        className="sr-only"
        onChange={onInputChange}
      />
    </label>
  );
}
