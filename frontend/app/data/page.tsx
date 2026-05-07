'use client';

import { useEffect } from 'react';
import { useDatasetStore } from '@/store/datasetStore';
import DatasetTable from '@/features/datasets/components/DatasetTable';
import { Database, Loader2 } from 'lucide-react';
import Link from 'next/link';

export default function DataPage() {
  const { fetchDatasets, isLoading, error, datasets } = useDatasetStore();

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Data</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your uploaded datasets ready for AI analysis.
          </p>
        </div>
        <Link
          href="/upload"
          className="rounded-xl bg-parkrun-dark px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-parkrun-dark/90 transition-colors"
        >
          + Upload new
        </Link>
      </div>

      {/* Summary cards */}
      {!isLoading && datasets.length > 0 && (
        <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {(
            [
              {
                label: 'Total datasets',
                value: datasets.length,
                colour: 'text-parkrun-dark',
              },
              {
                label: 'Survey files',
                value: datasets.filter((d) => d.type === 'Survey').length,
                colour: 'text-green-600',
              },
              {
                label: 'Operational',
                value: datasets.filter((d) => d.type === 'Operational').length,
                colour: 'text-blue-600',
              },
              {
                label: 'Documents',
                value: datasets.filter((d) => d.type === 'Document').length,
                colour: 'text-amber-600',
              },
            ] as const
          ).map(({ label, value, colour }) => (
            <div
              key={label}
              className="rounded-2xl border border-gray-100 bg-white p-4 shadow-sm"
            >
              <p className={`text-2xl font-bold ${colour}`}>{value}</p>
              <p className="mt-0.5 text-xs text-gray-400">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* States */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16 text-gray-400">
          <Loader2 size={24} className="animate-spin" />
        </div>
      ) : error ? (
        <div className="rounded-2xl border border-red-100 bg-red-50 p-6 text-center text-sm text-red-600">
          {error}
        </div>
      ) : datasets.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed border-gray-200 py-16 text-center">
          <Database size={32} className="text-gray-300" />
          <div>
            <p className="font-semibold text-gray-700">No datasets yet</p>
            <p className="mt-1 text-sm text-gray-400">
              Upload your first file to get started.
            </p>
          </div>
          <Link
            href="/upload"
            className="rounded-xl bg-parkrun-dark px-5 py-2.5 text-sm font-semibold text-white hover:bg-parkrun-dark/90 transition-colors"
          >
            Upload data
          </Link>
        </div>
      ) : (
        <DatasetTable />
      )}
    </div>
  );
}
