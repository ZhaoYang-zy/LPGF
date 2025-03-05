from torch.utils.data.dataloader import DataLoader


def getDataLoader(batch_size, dataset, shuffle=True, num_workers=1, drop_last=True):
    dataLoader = DataLoader(batch_size=batch_size, dataset=dataset, shuffle=shuffle, num_workers=num_workers,
                            drop_last=drop_last)
    return dataLoader

