'use client';

import { useMemo } from 'react';
import { create } from 'zustand';
import type { Dataset, DatasetFilter } from '@/types/dataset';
import { datasetsService } from '@/services/datasets.service';

interface DatasetState {
  datasets: Dataset[];
  filter: DatasetFilter;
  isLoading: boolean;
  error: string | null;
  fetchDatasets: () => Promise<void>;
  addDataset: (dataset: Dataset) => void;
  removeDataset: (id: string) => Promise<void>;
  setFilter: (filter: Partial<DatasetFilter>) => void;
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  datasets: [],
  filter: { search: '', type: 'All' },
  isLoading: false,
  error: null,

  fetchDatasets: async () => {
    set({ isLoading: true, error: null });
    try {
      const datasets = await datasetsService.getAll();
      set({ datasets, isLoading: false });
    } catch {
      set({ error: 'Failed to load datasets', isLoading: false });
    }
  },

  addDataset: (dataset) =>
    set((state) => ({ datasets: [dataset, ...state.datasets] })),

  removeDataset: async (id) => {
    await datasetsService.delete(id);
    set((state) => ({ datasets: state.datasets.filter((d) => d.id !== id) }));
  },

  setFilter: (filter) =>
    set((state) => ({ filter: { ...state.filter, ...filter } })),
}));

export function useFilteredDatasets() {
  const datasets = useDatasetStore((state) => state.datasets);
  const filter = useDatasetStore((state) => state.filter);

  return useMemo(
    () =>
      datasets.filter((d) => {
        const matchesSearch =
          !filter.search ||
          d.name.toLowerCase().includes(filter.search.toLowerCase()) ||
          d.fileName.toLowerCase().includes(filter.search.toLowerCase());
        const matchesType = filter.type === 'All' || d.type === filter.type;
        return matchesSearch && matchesType;
      }),
    [datasets, filter],
  );
}
