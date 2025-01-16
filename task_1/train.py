import numpy as np
import polars as pl
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

label_names = ["O", "B-MNT", "I-MNT"]
id2label = dict(enumerate(label_names))
label2id = dict(zip(id2label.values(), id2label.keys()))

model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def bert_transform(record):
    tokens, labels = record["tokens"], record["labels"]
    # Tokenize the input words. This will break words into subtokens if necessary.
    # For instance, "ChatGPT" might become ["Chat", "##G", "##PT"].
    tokenized = tokenizer(tokens, truncation=True, is_split_into_words=True)

    # Get the word IDs corresponding to each token. This tells us to which original word each token corresponds.
    word_ids = tokenized.word_ids()

    previous_word_id = None
    new_labels = []

    # For each token, determine which label it should get.
    for wid in word_ids:
        # If the token does not correspond to any word (e.g., it's a special token), set its tag to -100.
        if wid is None:
            new_labels.append(-100)
        # If the token corresponds to a new word, use the tag for that word.
        elif wid != previous_word_id:
            new_labels.append(labels[wid])
        # If the token is a subtoken (i.e., part of a word we've already tagged), set its tag to -100.
        else:
            new_labels.append(-100)
        previous_word_id = wid

    tokenized["labels"] = new_labels
    return tokenized

def compute_metrics(p):
    seqeval = evaluate.load("seqeval")
    # p is the results containing a list of predictions and a list of labels
    # Unpack the predictions and true labels from the input tuple 'p'.
    predictions_list, labels_list = p

    # Convert the raw prediction scores into tag indices by selecting the tag with the highest score for each token.
    predictions_list = np.argmax(predictions_list, axis=2)

    # Filter out the '-100' labels that were used to ignore certain tokens (like sub-tokens or special tokens).
    # Convert the numeric tags in 'predictions' and 'labels' back to their string representation using 'tag_names'.
    # Only consider tokens that have tags different from '-100'.
    true_predictions = [
        [label_names[p] for (p, l) in zip(predictions, labels) if l != -100]
        for predictions, labels in zip(predictions_list, labels_list)
    ]
    true_tags = [
        [label_names[l] for (p, l) in zip(predictions, labels) if l != -100]
        for predictions, labels in zip(predictions_list, labels_list)
    ]

    # Evaluate the predictions using the 'seqeval' library, which is commonly used for sequence labeling tasks like NER.
    # This provides metrics like precision, recall, and F1 score for sequence labeling tasks.
    results = seqeval.compute(predictions=true_predictions, references=true_tags)

    # Return the evaluated metrics as a dictionary.
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


train, test = pl.read_ndjson("./train.ndjson").to_dicts(), pl.read_ndjson("./test.ndjson").to_dicts()
train, test = list(map(bert_transform, train)), list(map(bert_transform, test))
train_ds = Dataset.from_list(train)
test_ds = Dataset.from_list(test)

data_collator = DataCollatorForTokenClassification(tokenizer)
args = TrainingArguments(
    "output",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print(trainer.train())
