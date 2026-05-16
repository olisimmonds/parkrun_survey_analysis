import type { Dataset } from '@/types/dataset';
import { defaultConfig } from '@/types/config';
import mockData from '@/data/mock/datasets.json';
import { sleep } from '@/lib/utils';
import { apiFetch } from './api';

const mock = mockData as Dataset[];

export const datasetsService = {
  async getAll(): Promise<Dataset[]> {
    if (defaultConfig.mockMode) {
      await sleep(400);
      return mock;
    }
    return apiFetch<Dataset[]>('/api/datasets');
  },

  async getById(id: string): Promise<Dataset | undefined> {
    if (defaultConfig.mockMode) {
      await sleep(200);
      return mock.find((d) => d.id === id);
    }
    return apiFetch<Dataset>(`/api/datasets/${id}`);
  },

  async delete(id: string): Promise<void> {
    if (defaultConfig.mockMode) {
      await sleep(300);
      return;
    }
    await apiFetch<void>(`/api/datasets/${id}`, { method: 'DELETE' });
  },
};
