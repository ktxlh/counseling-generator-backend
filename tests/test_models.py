import sys

# setting path
sys.path.append('../counseling-generator-backend')

import pandas as pd
from models import Generator, Predictor
from transformers import AutoModelForCausalLM, BertForSequenceClassification

CODES = ['AF', 'SUP', 'PR', 'QUC', 'RF', 'QUO', 'INT', 'GR']

model_dir = '/data/shsu70/testing/predictors/'
model_path = '/data/shsu70/testing/generator'

def save_dummy_models():
    print("save_dummy_models()")
    model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                output_attentions = False,
                output_hidden_states = False,
                return_dict = True,
    )
    model.resize_token_embeddings(model.config.vocab_size + 2)
    for code, name in Predictor.MODEL_NAMES.items():
        model.save_pretrained(model_dir + name)

    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/DialoGPT-small',
        output_attentions = False,
        output_hidden_states = False,
        return_dict = True,
    )
    model.resize_token_embeddings(model.config.vocab_size + 2 + len(CODES))
    model.save_pretrained(model_path)


def test_predictor():
    print("test_predictor()")
    df = pd.DataFrame.from_dict({
        'user_id':["a", "b", "a", "b", "a","b"], 
        'is_listener':[0,1,0,1,0,1], 
        "time":["t" for _ in range(6)], 
        'utterance':["Hi", "how r u?", "good, u?", "good", "why r u here", "i wanna discuss some issue"],
        'predictor_input_ids': [[] for _ in range(6)],
        'generator_input_ids': [[] for _ in range(6)],
    })
    predictor = Predictor(model_dir)
    for i in range(len(df)):
        print(i)
        scores = predictor.predict(df[:i+1])
        print(scores)


def test_generator():
    print("test_generator()")
    df = pd.DataFrame.from_dict({
        'user_id':["a", "b", "a", "b", "a","b"], 
        'is_listener':[0,1,0,1,0,1], 
        "time":["t" for _ in range(6)], 
        'utterance':["Hi", "How are you?", "good, thx. how abt u?", "better than i deserve!", "why brings you here", "i wanna discuss some issue about my family"],
        'predictor_input_ids': [[] for _ in range(6)],
        'generator_input_ids': [[] for _ in range(6)],
    })
    generator = Generator(model_path)
    for i in range(len(df)):
        print(i)
        utterances = generator.predict(df[:i+1], CODES if i - 5 + 1 >= 0 else [])
        print(utterances)


if __name__ == "__main__":
    # save_dummy_models()
    # test_predictor()
    test_generator()
