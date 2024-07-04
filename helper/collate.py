import torch

def collate_dataset_gen(batch):
    return batch


def collate_nn(batch):
    window = torch.concat([w for w, _, _ in batch])
    onset = torch.concat([o for _, o, _ in batch])
    ids = torch.concat([id for _, _, id in batch])
    return (window, onset, ids)