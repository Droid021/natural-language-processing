import torch
import utils
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def loss_fn(start_logits, end_logits, target_start, target_end):
    l1 = nn.BCEWithLogitsLoss()(start_logits, target_start)
    l2 = nn.BCEWithLogitsLoss()(end_logits, target_end)
    return l1+l2

def train(dataloader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(dataloader, total=len(dataloader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1,o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(o1,o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

def eval(dataloader, model, optimizer, device, scheduler):
    model.eval()

    fin_output_start, fin_output_end, fin_padding_len = [],[],[]
    fin_tweet_tokens, fin_orig_sentiment, fin_orig_selected= [],[],[]
    fin_orig_tweet = []
    for bi, d in enumerate(tqdm(dataloader, total=len(dataloader))):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        tweet_tokens = d['tweet_tokens']
        padding_len = d['padding_len']
        orig_sentiment = d['orig_sentiment'] 
        orig_selected = d['orig_selected']
        orig_tweet = d['orig_tweet']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1,o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_len.extend(padding_len.cpu().detach().numpy().toList())
        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    threshold = 0.2
    jaccards = []
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_len[j]
        orig_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_output_start[j, :][:-padding_len] >= threshold
            mask_end = fin_output_end[j, :][:-padding_len] >= threshold
        else: 
            mask_start = fin_output_start[j, :] >= threshold
            mask_end = fin_output_end[j, :] >= threshold
        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]

        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else: 
                idx_end = idx_start
        else:
            idx_start = 0 
            idx_end = 0

        for mj in range(idx_start, idx_end+1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]

        final_output = ''
        for ot in output_tokens:
            if ot.startswith('##'):
                final_output = final_output + ot[2:]
            elif len(ot) == 1 and ot in string.punctuation:
                final_output = final_output + ot
            else:
                final_output = final_output + ' ' + ot
        final_output = final_output.split()
        if sentiment == 'neutral' or len(orig_tweet.split()) < 4:
            final_output = orig_tweet

        jac = utils.jaccard(target_string.split(), final_output.strip())
        jaccards.append(jac)
    mean_jac = np.mean(jaccards)

    return mean_jac
        