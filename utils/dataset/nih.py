from torch.utils.data import Dataset

class NIHChestDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]