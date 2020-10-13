import torch
from tqdm import tqdm
import torch.nn as nn

def loss_fn(outputs, target):
    return nn.BCEWithLogitsLoss()(outputs, target.view(-1, 1))



def train_fn(data_loader, model, optimizer, accumulation_steps, device):
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        target = d['target']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        target = target.to(device, dtype = torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids = ids,
            mask = mask,
            token_type_ids = token_type_ids
        )

        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids']
            mask = d['mask']
            token_type_ids = d['token_type_ids']
            target = d['target']

            ids = ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            target = target.to(device, dtype = torch.float)

            outputs = model(
                ids = ids,
                mask = mask,
                token_type_ids = token_type_ids
            )
            fin_targets.extend(target.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
