import yaml
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from dataset import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class HIVTrainer():
    def __init__(self, train_data, model_name='Rostlab/prot_bert', training_args_path='training_args.yaml'):

        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(model_name)
        
        with open(training_args_path, 'r') as f:
            yaml_file = yaml.load(f, Loader=yaml.FullLoader)

        training_args = TrainingArguments(**yaml_file)

        self.model_name = model_name
        self.train_dataset = self.build_dataset(train_data, split='train', tokenizer_name=model_name)
        self.val_dataset = self.build_dataset(train_data, split='test', tokenizer_name=model_name)

        self.trainer = Trainer(
            model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
            args=training_args,                   # training arguments, defined above
            train_dataset=self.train_dataset,          # training dataset
            eval_dataset=self.val_dataset,             # evaluation dataset
            compute_metrics = compute_metrics,    # evaluation metrics
        )

    def train(self):
        self.trainer.train()

    def test(self, test_data):
        test_dataset = self.build_dataset(test_data, split='test', tokenizer_name=self.model_name)
        predictions, label_ids, metrics = self.trainer.predict(test_dataset)
        return predictions, label_ids, metrics
    
    def build_dataset(self, *args, **kwargs):
        return HIVDataset(*args, **kwargs)

class HIVTrainer_Albert(HIVTrainer):
    def build_dataset(self, *args, **kwargs):
        return HIVDataset_Albert(*args, **kwargs)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auc
    }