import type { Dataset } from '@/types/dataset';
import mockData from '@/data/mock/datasets.json';
import { sleep } from '@/lib/utils';

const mock = mockData as Dataset[];

export const datasetsService = {
  async getAll(): Promise<Dataset[]> {
    await sleep(400);
    return mock;
  },

  async getById(id: string): Promise<Dataset | undefined> {
    await sleep(200);
    return mock.find((d) => d.id === id);
  },

  async delete(id: string): Promise<void> {
    await sleep(300);
    // In production: DELETE /datasets/:id
  },
};
