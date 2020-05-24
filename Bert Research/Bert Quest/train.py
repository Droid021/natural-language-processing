import torch.nn as nn
import transformers 
import torch
from scipy import stats
import numpy as np
import pandas as pd
from torch.optim import Adam
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import wandb
#https://app.wandb.ai/droid/torch-xla-colab
wandb.init(project="torch-xla-colab", reinit=True)

class BertBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 30)
    
    def forward(self,ids, mask, token_type_ids):
        _ , pooled_output = self.model(ids, attention_mask = mask,
                                    token_type_ids = token_type_ids)
        out = self.dropout(pooled_output)
        return self.linear(out)
# BertBaseUncased()

class BertDataset:
    def __init__(self, qtitle, qbody, answer,targets, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.targets = targets

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer = str(self.answer[item])
        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body,
            answer,
            add_special_tokens=True,
            max_length = self.max_len,
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = self.max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }
 
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train(dataloader, model, optimizer, device, scheduler=None):
    wandb.watch(model)
    model.train()
    for batch_idx, d in enumerate(dataloader):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        xm.optimizer_step(optimizer)

        if scheduler:
            scheduler.step()
        if batch_idx % 100 == 0:
            wandb.log({"Batch index": batch_idx, "Train loss": loss})
            print(f"Batch index : {batch_idx} Train loss : {loss}")

def eval(dataloader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    for batch_index, data in enumerate(dataloader):
        wandb.init(project="torch-xla-colab", reinit=True)
        ids = data['ids'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['targets'].to(device)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        print(f"Test Loss {loss}")
        wandb.log({"Test Loss": loss})

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())
    wandb.log({"Final Outputs": fin_outputs, "Final Targets": fin_tagets})
    return np.vstack(fin_outputs), np.vstack(fin_targets)

def run(index):
    MAX_LEN = 512
    BATCH_SIZE = 4
    EPOCHS = 4
    
    dfx = pd.read_csv('train.csv').fillna('none')
    df_train, df_valid = train_test_split(dfx, random_state=42, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    sample = pd.read_csv('sample_submission.csv')
    target_cols = list(sample.drop('qa_id', axis=1).columns)

    train_targets = df_train[target_cols].values
    valid_targets = df_valid[target_cols].values

    train_dataset = BertDataset(
        qtitle = df_train.question_title.values,
        qbody = df_train.question_body.values,
        answer = df_train.answer.values,
        targets = train_targets,
        max_len = MAX_LEN,
    )

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas = xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle = True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        sampler=train_sampler
    )

    valid_dataset = BertDataset(
        qtitle = df_valid.question_title.values,
        qbody = df_valid.question_body.values,
        answer = df_valid.answer.values,
        targets = valid_targets,
        max_len = MAX_LEN,
    )
    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset,
        num_replicas = xm.xrt_world_size(),
        rank=xm.get_ordinal()
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        sampler = valid_sampler
    )

    device = xm.xla_device()
    lr = 3e-5 * xm.xrt_world_size()
    num_training_steps = int(len(train_dataset)/BATCH_SIZE/xm.xrt_world_size() * EPOCHS)
    model = BertBaseUncased().to(device)
    optimizer = Adam(model.parameters(),lr=lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(EPOCHS):
        train_parallel_loader = pl.ParallelLoader(train_loader, [device])
        train(dataloader=train_parallel_loader.per_device_loader(device), 
              model=model, device=device, optimizer=optimizer,scheduler=scheduler)
        
        valid_parallel_loader = pl.ParallelLoader(valid_loader, [device])
        outputs, targets = eval(dataloader=valid_parallel_loader.per_device_loader(device), 
                                model=model, device=device)
        
        spear=[]
        for jj in range(targets.shape[1]):
            p1 = list(targets[:, jj])
            p2 = list(outputs[:, jj])
            coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            spear.append(coef)
        spear = np.mean(spear)
        wandb.log({"Epoch": epoch+1, "Mean spearman coefficient": spear})
        print(f'spearman = {spear}')
        # Save model to wandb
    xm.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
if __name__ == '__main__':
    xmp.spawn(run, nprocs=8)
