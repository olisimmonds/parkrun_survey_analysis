'use client';

import { X } from 'lucide-react';
import type { Dataset } from '@/types/dataset';
import { formatBytes, formatDate } from '@/lib/utils';

interface Props {
  dataset: Dataset;
  onClose: () => void;
}

const Row = ({ label, value }: { label: string; value?: string | number | null }) =>
  value != null ? (
    <div className="flex justify-between gap-4 py-2.5">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-right text-sm font-medium text-gray-800">
        {value}
      </span>
    </div>
  ) : null;

export default function MetadataModal({ dataset, onClose }: Props) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="w-full max-w-md rounded-2xl bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-gray-100 p-6">
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              {dataset.name}
            </h2>
            <p className="mt-0.5 text-sm text-gray-400">{dataset.fileName}</p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 flex-shrink-0 rounded-lg p-1.5 text-gray-400 hover:bg-gray-100"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="divide-y divide-gray-50 px-6">
          <Row label="Type" value={dataset.type} />
          <Row label="Uploaded" value={formatDate(dataset.uploadedAt)} />
          <Row label="File size" value={formatBytes(dataset.size)} />
          {dataset.rowCount != null && (
            <Row label="Rows" value={dataset.rowCount.toLocaleString()} />
          )}
          {dataset.columnCount != null && (
            <Row label="Columns" value={dataset.columnCount} />
          )}
          <Row label="Status" value={dataset.status} />
        </div>

        {/* Description */}
        {dataset.description && (
          <div className="px-6 pb-4 pt-3">
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Description
            </p>
            <p className="text-sm text-gray-600">{dataset.description}</p>
          </div>
        )}

        {/* Tags */}
        {dataset.tags && dataset.tags.length > 0 && (
          <div className="px-6 pb-6">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Tags
            </p>
            <div className="flex flex-wrap gap-2">
              {dataset.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full bg-gray-100 px-3 py-1 text-xs text-gray-600"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="border-t border-gray-100 px-6 py-4">
          <button
            onClick={onClose}
            className="w-full rounded-xl bg-parkrun-dark py-2.5 text-sm font-semibold text-white hover:bg-parkrun-dark/90 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
