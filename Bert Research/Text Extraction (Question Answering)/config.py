import tokenizers
import urllib
import os
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 20
ACCUMULATION = 2
BERT_MODEL = "bert-base-uncased"
MODEL_PATH = "model.pt"
TRAINING_FILE = "../data/tweet-sentiment-extraction/train.csv"

def download_vocab_files_for_tokenizer(tokenizer, model_type, output_path):
    vocab_files_map = tokenizer.pretrained_vocab_files_map
    vocab_files = {}
    for resource in vocab_files_map.keys():
        download_location = vocab_files_map[resource][model_type]
        f_path = os.path.join(output_path, os.path.basename(download_location))
        urllib.request.urlretrieve(download_location, f_path)
        vocab_files[resource] = f_path
    return vocab_files

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
vocab_files = download_vocab_files_for_tokenizer(tokenizer, BERT_MODEL, 'vocab_files')
TOKENIZER = BertWordPieceTokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'))

# print(TOKENIZER)
