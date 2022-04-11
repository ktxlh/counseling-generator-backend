import os
from random import random
from typing import List, Tuple

import pandas as pd
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizerFast)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

LISTENER_TOKEN, CLIENT_TOKEN = '<|listener|>', '<|client|>'
CONTEXT_LEN = 5

class Predictor:
    MAX_LEN = 128
    MODEL_NAMES = {
        "AF": "MI_label_Affirm",
        "QUC": "MI_label_ClosedQuestion",
        "GR": "MI_label_Grounding",
        "INT": "MI_label_Introduction",
        "QUO": "MI_label_OpenQuestion",
        "PR": "MI_label_Persuade",
        "RF": "MI_label_Reflection",
        "SUP": "MI_label_Support",
    }

    def __init__(self, model_dir: str):
        """Load tokenizer and all classifiers

        Args:
            model_dir (str): path to the dir storing all classifiers
        """

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer.add_tokens([LISTENER_TOKEN, CLIENT_TOKEN])
        self.CLS_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

        self.models = {
            code: BertForSequenceClassification.from_pretrained(
                os.path.join(model_dir, model_name),
                output_attentions = False,
                output_hidden_states = False,
                return_dict = True,
            ) for code, model_name in Predictor.MODEL_NAMES.items()
        }
        for model in self.models.values():
            model.eval()
            model.cuda()
        

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Predict the next MITI code used if context is long enough

        Args:
            df (pd.DataFrame): speaker-prefixed latest utterances 

        Returns:
            List[Tuple[str, float]]: code and its predicted score pairs
        """
        index = len(df) - 1
        speaker_token = (LISTENER_TOKEN if df.at[index, 'is_listener'] else CLIENT_TOKEN)
        df.at[index, 'utterance'] = speaker_token + df.at[index, 'utterance']
        df.at[index, 'predictor_input_ids'] = self.tokenizer(
            df.at[index, 'utterance'], 
            add_special_tokens = False,  # Avoid adding special tokens '[CLS]' and '[SEP]'
            max_length = Predictor.MAX_LEN,
            truncation = True,
            padding = False,
            return_token_type_ids = False,
            return_attention_mask = False,
        )["input_ids"]
        
        scores = []
        if index - CONTEXT_LEN + 1 >= 0:
            input_ids = [self.CLS_TOKEN_ID,] + [y for x in df.iloc[index - CONTEXT_LEN + 1:]["predictor_input_ids"].tolist() for y in x]
            input_ids = torch.tensor(input_ids).view(1, -1)  # torch.LongTensor of shape (batch_size, sequence_length)
            for code in Predictor.MODEL_NAMES.keys():
                logits = self.models[code](input_ids.cuda()).logits[0, :]
                score = torch.nn.functional.softmax(logits, dim=0)[1].item()
                score = score * random() * 2 # DEBUG TODO remove this
                scores.append((code, score))
        return scores
        

class Generator:
    MAX_LEN = 48
    CODE_TOKENS = [f"<|{code}|>" for code in Predictor.MODEL_NAMES.keys()]

    def __init__(self, model_path: str):
        """Load tokenizer and generator

        Args:
            model_path (str): path to trained generator
        """

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        self.tokenizer.add_tokens([LISTENER_TOKEN, CLIENT_TOKEN])
        self.tokenizer.add_tokens(Generator.CODE_TOKENS)
        self.CODE_TOKEN_IDS = dict(zip(
            Predictor.MODEL_NAMES.keys(), 
            self.tokenizer.convert_tokens_to_ids(Generator.CODE_TOKENS)
        ))

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = True,
        )
        self.model.eval()
        self.model.cuda()
        

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, codes: List[str]) -> List[Tuple[str, str]]:
        """Predict the next utterance if given codes

        Args:
            df (pd.DataFrame): speaker-prefixed latest utterances 
            codes (List[str]): MITI codes to use, one for each generations

        Returns:
            List[Tuple[str, str]]: codes and corresponding generated utterances
        """
        index = len(df) - 1
        speaker_token = (LISTENER_TOKEN if df.at[index, 'is_listener'] else CLIENT_TOKEN)
        if not df.at[index, 'utterance'].startswith(speaker_token):
            df.at[index, 'utterance'] = speaker_token + df.at[index, 'utterance']
        df.at[index, 'generator_input_ids'] = self.tokenizer(
            df.at[index, 'utterance'], 
            max_length = Generator.MAX_LEN,
            truncation = True,
            padding = False,
            return_attention_mask = False,
        )["input_ids"]

        utterances = []
        if len(codes) > 0:
            input_ids = [y for x in df.iloc[index - CONTEXT_LEN + 1:]["generator_input_ids"].tolist() for y in x] + [-1,]  # placeholder for a code
            input_ids = torch.tensor(input_ids).view(1, -1)  # torch.LongTensor of shape (batch_size, sequence_length)
            for code in codes:
                input_ids[0, -1] = self.CODE_TOKEN_IDS[code]
                # Decoding methods: https://huggingface.co/blog/how-to-generate
                sample_output = self.model.generate(
                    input_ids.cuda(), 
                    do_sample=True, 
                    max_length=50, 
                    top_p=0.95, 
                    top_k=50
                )
                utterance = self.tokenizer.decode(sample_output[0, input_ids.shape[1]:], skip_special_tokens=True)
                utterances.append((code, utterance))
        return utterances
