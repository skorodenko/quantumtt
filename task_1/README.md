# NER for mountain names

The task is about recognizing mountain names in text corpus.

To solve NER task - Bert model was finetuned on data generated by Llama3.1.


Pretrained model: https://huggingface.co/rinkuro/bert-mount-ner


## How to install:

Runs with Python 3.11 (can be used with >3.11, but may require tweaks in requirements.txt)

Preferable way is to use poetry.

Otherwise use provided requirements.txt file.

```{bash}
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## How to run:

1. Dataset generation - dataset.ipynb
2. Training process - train.py
3. Demo - demo.ipynb
4. Inference (same code as demo, with cli for file input)
```{bash}
python inference.py -i text_file.txt
```


## Dataset:

- Mountain.csv - contains names of mountains
- train.ndjson - contains train data in ndjson format
- test.ndjson - contains test data in ndjson format


## Evaluation metrics:
- Overall accuracy: 99.17%
- Overall recall: 83.26%
- Overall f1 score: 79.62%


## Ways to improve:

1. This project utilizes Llama3.1 for dataset generation. Such data is known to be not ideal to train models.
To reach top notch performance realworld data is a must have.
2. This project uses Bert (base), which is used to tokenize and classify text. There are models which would perform better. 