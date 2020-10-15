import numpy as np
import torch
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import datetime
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class Classifier:

    def __init__(self, load=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=10,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = None
        self.epochs = 4
        self.path = 'data/saved_weights.pt'

        # Load weights for evaluation
        if load:
            self.model.load_state_dict(torch.load(self.path, map_location=self.device))

    def train(self, train_dataloader, val_dataloader):

        # Total number of training steps
        total_steps = len(train_dataloader) * self.epochs

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=total_steps)

        best_val_acc = 0.0
        for epoch in range(0, self.epochs):

            # Train
            self.__train_epoch(train_dataloader)

            # Validation
            avg_val_accuracy, _, _ = self.evaluate(val_dataloader)

            # Save best model
            if avg_val_accuracy > best_val_acc:
                best_val_acc = avg_val_accuracy
                torch.save(self.model.state_dict(), self.path)

        return best_val_acc

    def __flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def __format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        #  hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def __train_epoch(self, dataloader):

        self.model.train()

        for step, batch in enumerate(dataloader):
            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch

            # Clear gradients
            self.model.zero_grad()

            # Forward pass
            outputs = self.model(sent_id,
                                 attention_mask=mask,
                                 labels=labels,
                                 return_dict=True)

            # Get loss on CrossEntropyLoss
            loss = outputs.loss

            loss.backward()

            # Clip the the gradients to 1.0 (exploding gradient)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Update lr
            self.scheduler.step()

    def evaluate(self, dataloader):

        # Eval mode
        self.model.eval()

        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []

        for batch in dataloader:
            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch

            with torch.no_grad():
                outputs = self.model(sent_id,
                                     attention_mask=mask,
                                     labels=labels,
                                     return_dict=True)

            # Logits (before softmax)
            logits = outputs.logits

            # Put values to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Predictions and true labels
            predictions.extend(logits)
            true_labels.extend(label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += self.__flat_accuracy(logits, label_ids)

        # Average accuracy
        avg_val_accuracy = eval_accuracy / len(dataloader)

        return avg_val_accuracy, predictions, true_labels

    def prepare_data(self, df, batch_size=2):
        df.reset_index(inplace=True)

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokens = tokenizer.batch_encode_plus(df['text'].tolist(), truncation=True, padding=True)

        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        y = torch.tensor(df['label'].tolist())

        data = TensorDataset(seq, mask, y)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader
