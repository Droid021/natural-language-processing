import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
ACCUMULATION = 2
BERT_PATH = "../data/bert-uncased/"
MODEL_PATH = "model.pt"
TRAINING_FILE = "../data/imdb-50k/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 
#TOKENIZER = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-cased')
