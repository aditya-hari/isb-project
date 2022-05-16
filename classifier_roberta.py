import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

RobertaTokenizer = AutoTokenizer.from_pretrained("roberta-base")
RobertaModel = AutoModel.from_pretrained("roberta-base")

device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv('train.csv', sep=',', names=['sentence','type_id'])
test_df = pd.read_csv('test.csv', sep=',', names=['sentence','type_id'])

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.00001
drop_out = 0.3
tokenizer = RobertaTokenizer

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.data.sentence[index])
        sentence = " ".join(sentence.split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.type_id[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

train_dataset=df
testing_dataset=test_df

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_out)
        self.classifier = torch.nn.Linear(768, 10)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


#['business process and analysis', 'project management', 'cybersecurity', 'business strategy', 'product management', 'web development', 'risk management', 'data analysis', 'machine learning', 'mobile development']

model = RobertaClass()
model.to(device)
weights = [ 0.43462199,  0.33015468,  1.72547666,  0.67633068,  2.30255584, 1.20498865,  1.90361369,  1.56629337,  8.36936811, 15.45197013]
class_weights = torch.FloatTensor(weights).to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}\n")
    print(f"Training Accuracy Epoch: {epoch_accu}\n")
    print("\n")
    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; tr_loss = 0
    nb_tr_steps =0
    nb_tr_examples =0
    pred = []
    act = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            pred += big_idx.tolist()
            act += targets.tolist()
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}\n")
    print(f"Validation Accuracy Epoch: {epoch_accu}\n")
    mf1 = f1_score(act, pred, average='macro')
    print(f"Validation Macro F1: {mf1}\n")
    return mf1

bacc = 0
for epoch in range(EPOCHS):
    train(epoch)
    acc = valid(model, testing_loader)
    if acc > bacc:
        torch.save(model, 'roberta_classifier.pt')
    print("\n")
    print("\n")
