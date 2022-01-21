

def dict_to_device(sample,device):
    for key in sample.keys():
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].to(device,non_blocking=True)

