import pandas as pd

LAGGED_DAYS = 60

#Inserts a mask equal to LAGGED_DAYS to distinguish nan values from 0
def mask_nan(frame):
    mask = frame.isna()
    #Input frame is arranged as follows: 'hours','prov_id',recurrance block,other block
    for i in range(1,LAGGED_DAYS+1):
        #frame.insert(1+LAGGED_DAYS+i,f"mask_{i}",mask[f"hours_l{i}"].astype(int).astype('float32'))
        frame.insert(1+LAGGED_DAYS+i,f"mask_{i}",mask[f"L{i}_hours"].astype(int).astype('float32'))
    frame = frame.fillna(0)
    return frame


class PBJ_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        include_fields = ['hours','prov_id','day_of_week','avg_employees_7days']
        for i in range(1,LAGGED_DAYS+1):
            include_fields.insert(i+1,f"L{i}_hours")
        
        self.frame = pd.read_csv(csv_file,usecols=include_fields)
        self.frame = mask_nan(self.frame.reindex(columns=include_fields))

        one_hot = pd.get_dummies(self.frame.day_of_week)
        self.frame = self.frame.drop(['day_of_week'],axis=1)
        self.frame = pd.concat([self.frame,one_hot],axis=1)


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = np.array(self.frame.iloc[idx, 1:])
        labels = np.array(self.frame.iloc[idx, 0])

        series = np.flip(features[:,1:1+LAGGED_DAYS])
        mask = np.flip(features[:,1+LAGGED_DAYS:1+2*LAGGED_DAYS])
        time_series = np.concatenate([series,mask],axis=0)

        additional_descriptors = features[:,1+2*LAGGED_DAYS:]

        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample