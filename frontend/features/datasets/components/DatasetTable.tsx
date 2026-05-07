'use client';

import { useState } from 'react';
import { Trash2, Info, Search, ChevronUp, ChevronDown } from 'lucide-react';
import { formatBytes, formatDate, cn } from '@/lib/utils';
import {
  useDatasetStore,
  useFilteredDatasets,
} from '@/store/datasetStore';
import type { Dataset } from '@/types/dataset';
import MetadataModal from './MetadataModal';

const TYPE_COLOURS: Record<Dataset['type'], string> = {
  Survey: 'bg-parkrun-bright/15 text-green-800',
  Operational: 'bg-blue-50 text-blue-700',
  Document: 'bg-amber-50 text-amber-700',
};

type SortKey = 'name' | 'type' | 'uploadedAt' | 'size';

export default function DatasetTable() {
  const { filter, setFilter, removeDataset } = useDatasetStore();
  const datasets = useFilteredDatasets();
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('uploadedAt');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sorted = [...datasets].sort((a, b) => {
    let cmp = 0;
    if (sortKey === 'name') cmp = a.name.localeCompare(b.name);
    else if (sortKey === 'type') cmp = a.type.localeCompare(b.type);
    else if (sortKey === 'uploadedAt')
      cmp = a.uploadedAt.localeCompare(b.uploadedAt);
    else if (sortKey === 'size') cmp = a.size - b.size;
    return sortDir === 'asc' ? cmp : -cmp;
  });

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k ? (
      sortDir === 'asc' ? (
        <ChevronUp size={14} />
      ) : (
        <ChevronDown size={14} />
      )
    ) : (
      <ChevronDown size={14} className="opacity-20" />
    );

  return (
    <>
      {/* Filters */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative max-w-sm flex-1">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
          />
          <input
            type="text"
            placeholder="Search datasets…"
            value={filter.search}
            onChange={(e) => setFilter({ search: e.target.value })}
            className="w-full rounded-xl border border-gray-200 bg-white py-2 pl-9 pr-4 text-sm text-gray-800 shadow-sm placeholder:text-gray-400 focus:border-parkrun-dark focus:outline-none focus:ring-2 focus:ring-parkrun-dark/10"
          />
        </div>

        <div className="flex gap-2">
          {(['All', 'Survey', 'Operational', 'Document'] as const).map(
            (type) => (
              <button
                key={type}
                onClick={() => setFilter({ type })}
                className={cn(
                  'rounded-full px-4 py-1.5 text-xs font-medium transition-colors',
                  filter.type === type
                    ? 'bg-parkrun-dark text-white'
                    : 'bg-white text-gray-500 hover:bg-gray-100',
                )}
              >
                {type}
              </button>
            ),
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50 text-xs font-semibold uppercase tracking-wider text-gray-500">
                {(
                  [
                    ['name', 'Dataset Name'],
                    ['type', 'Type'],
                    ['uploadedAt', 'Uploaded'],
                    ['size', 'Size'],
                  ] as [SortKey, string][]
                ).map(([key, label]) => (
                  <th
                    key={key}
                    onClick={() => handleSort(key)}
                    className="cursor-pointer px-6 py-3 text-left"
                  >
                    <span className="flex items-center gap-1">
                      {label}
                      <SortIcon k={key} />
                    </span>
                  </th>
                ))}
                <th className="px-6 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {sorted.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    className="px-6 py-12 text-center text-sm text-gray-400"
                  >
                    No datasets found.{' '}
                    <a
                      href="/upload"
                      className="text-parkrun-dark underline"
                    >
                      Upload one
                    </a>{' '}
                    to get started.
                  </td>
                </tr>
              ) : (
                sorted.map((ds) => (
                  <tr
                    key={ds.id}
                    className="transition-colors hover:bg-gray-50/50"
                  >
                    <td className="px-6 py-4">
                      <p className="font-medium text-gray-800">{ds.name}</p>
                      <p className="mt-0.5 text-xs text-gray-400">
                        {ds.fileName}
                      </p>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={cn(
                          'inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium',
                          TYPE_COLOURS[ds.type],
                        )}
                      >
                        {ds.type}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-500">
                      {formatDate(ds.uploadedAt)}
                    </td>
                    <td className="px-6 py-4 text-gray-500">
                      {formatBytes(ds.size)}
                      {ds.rowCount && (
                        <span className="ml-2 text-gray-300">
                          · {ds.rowCount.toLocaleString()} rows
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => setSelected(ds)}
                          className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                          aria-label="View metadata"
                        >
                          <Info size={16} />
                        </button>
                        <button
                          onClick={() => removeDataset(ds.id)}
                          className="rounded-lg p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500"
                          aria-label="Delete dataset"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <MetadataModal dataset={selected} onClose={() => setSelected(null)} />
      )}
    </>
  );
}
