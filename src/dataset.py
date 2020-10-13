import configr
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = configr.TOKENIZER
        self.max_len = configr.MAX_LEN

    
    def __len__(self):
        return len(self.review)
    

    def __getitem__(self, item):
        review = self.review[item]
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )


        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        padding_len = self.max_len  - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids" : torch.tensor(ids, dtype = torch.long),
            "mask" : torch.tensor(mask, dtype = torch.long),
            "token_type_ids" : torch.tensor(token_type_ids, dtype = torch.long),
            "target" : torch.tensor(self.target[item], dtype = torch.float)
        }

