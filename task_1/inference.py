import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


model = AutoModelForTokenClassification.from_pretrained("rinkuro/bert-mount-ner")
tokenizer = AutoTokenizer.from_pretrained("rinkuro/bert-mount-ner")
classifier = pipeline("ner", model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that detects mountain names in file"
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Input filename")
    args = parser.parse_args()
    input_filename = args.input

    with open(input_filename, "r") as fp:
        text = fp.read()

    res = classifier(text)
    print("Discovered mountain names:")
    for e in res:
        print(e)
