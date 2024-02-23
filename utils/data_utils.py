import torch
from torch.utils.data import DataLoader, RandomSampler

def make_loader(dataset,batch_size=16, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, 
                      sampler=RandomSampler(dataset, generator=generator),
                      collate_fn = lambda s: dataset.collate_fn(s),
                      batch_size=batch_size,
                      )