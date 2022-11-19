import os
from typing import List, Tuple

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BertForSequenceClassification, BertTokenizerFast)

# Comment this line to test locally
assert torch.cuda.is_available()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Run it only for the first time.
# nltk.download("punkt")

LISTENER_TOKEN, CLIENT_TOKEN = '<|listener|>', '<|client|>'
CONTEXT_LEN = 5

# Avoid slow first GPU call
if torch.cuda.is_available():
    for i in range(3):
        device = torch.device(f"cuda:{i}")
        torch.ones(1).to(device)
    print("Moved dummy tensor(s) to cuda")

class Predictor:
    PRED_THRESHOLD = 0.6
    MAX_NUM_PREDS = 3
    NO_INT_THRESHOLD = 4
    START_PRED_THRESHOLD = 5

    MAX_LEN = 128
    MODEL_NAMES = {
        "AF": "AF/best_model",
        "QUC": "QUC/best_model",
        "GR": "GR/best_model",
        "INT": "INT/best_model",
        "QUO": "QUO/best_model",
        "PR": "PR/best_model",
        "RF": "RF/best_model",
        "SUP": "SUP/best_model",
    }

    # This is the only list sorted! .keys() are not sorted!
    CODES = ["AF", "SUP", "PR", "QUC", "RF", "QUO", "INT", "GR"]

    def __init__(self, model_dir: str):
        """Load tokenizer and all classifiers

        Args:
            model_dir (str): path to the dir storing all classifiers
        """

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer.add_tokens([LISTENER_TOKEN, CLIENT_TOKEN])
        self.CLS_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            model.to(self.device)
        

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Predict the next MITI code used if context is long enough

        Args:
            df (pd.DataFrame): speaker-prefixed latest utterances 

        Returns:
            List[Tuple(str, float)]: predicted codes and corresponding predicted scores
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
        
        # Predict only if the dialog history (context) is long enough
        if index < Predictor.START_PRED_THRESHOLD - 1:
            return []
        
        code_scores = []
        context_start_index = max(0, index - CONTEXT_LEN + 1)
        input_ids = [self.CLS_TOKEN_ID,] + [y for x in df.iloc[context_start_index:]["predictor_input_ids"].tolist() for y in x]
        input_ids = torch.tensor(input_ids).view(1, -1)  # torch.LongTensor of shape (batch_size, sequence_length)
        for code in Predictor.CODES:
            logits = self.models[code](input_ids.to(self.device)).logits[0, :]
            score = torch.nn.functional.softmax(logits, dim=0)[1].item()
            code_scores.append((code, score))

        # Rule: Don't suggest Introduction/Greetings when index >= NO_INT_THRESHOLD
        if index >= Predictor.NO_INT_THRESHOLD:
            code_scores = list(filter(lambda code_score: code_score[0] != "INT", code_scores))

        # Sort
        code_scores.sort(key=lambda code_score: -code_score[1])
        # Filter
        code_scores = list(filter(lambda code_score: code_score[1] > Predictor.PRED_THRESHOLD, code_scores))
        # Truncate
        del code_scores[Predictor.MAX_NUM_PREDS:]

        return code_scores
        

class Generator:
    MAX_LEN = 64
    MAX_NEW_LEN = 32  # Extra variable to control the desired new utterance length
    MAX_NUM_SENTS = 2

    CODE_TOKENS = [f"<|{code}|>" for code in Predictor.CODES]

    def __init__(self, model_path: str):
        """Load tokenizer and generator

        Args:
            model_path (str): path to trained generator
        """

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        self.tokenizer.add_tokens([LISTENER_TOKEN, CLIENT_TOKEN])
        self.tokenizer.add_tokens(Generator.CODE_TOKENS)
        bad_words = [
            "male", "female", "man", "woman", "girl", "boy", "boyfriend", 
            "girlfriend", "husband", "wife", "bf", "gf", "age", "old",
            "son", "daughter", "parents", 
        ]
        self.bad_words_ids = [self.tokenizer.encode(w) for w in bad_words]
        self.CODE_TOKEN_IDS = dict(zip(
            Predictor.CODES, 
            self.tokenizer.convert_tokens_to_ids(Generator.CODE_TOKENS)
        ))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = True,
        )
        self.model.eval()
        self.model.to(self.device)
        

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, code_scores: List[Tuple[str, float]]) -> List[str]:
        """Predict the next utterance if given codes

        Args:
            df (pd.DataFrame): speaker-prefixed latest utterances 
            code_scores (List[Tuple[str, float]]): MITI codes to use, one for each generations, and their scores

        Returns:
            List[str]: generated utterances corresponding to codes
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
        if len(code_scores) > 0:
            context_start_index = max(0, index - CONTEXT_LEN + 1)
            input_ids = [y for x in df.iloc[context_start_index:]["generator_input_ids"].tolist() for y in x] + [-1,]  # placeholder for a code
            input_ids = torch.tensor(input_ids).view(1, -1)  # torch.LongTensor of shape (batch_size, sequence_length)
            input_ids = input_ids.repeat(len(code_scores), 1)
            input_ids[:, -1] = torch.LongTensor([self.CODE_TOKEN_IDS[code] for (code, _) in code_scores])
            # Decoding methods: https://huggingface.co/blog/how-to-generate
            outputs = self.model.generate(
                input_ids.to(self.device), 
                do_sample=True, 
                max_length=input_ids.shape[1] + Generator.MAX_NEW_LEN,
                top_p=0.95, 
                top_k=50,
                length_penalty=0.5,
                temperature=0.4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                forced_eos_token_id=self.tokenizer.eos_token_id,
            )
            utterances = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
        return utterances


class Guard:
    def __init__(self, model_dir: str):
        model_name = "finetuned_mh_BERT_epoch_0_IP_hatebert_4.model"
        self.tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", do_lower_case=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            "GroNLP/hateBERT",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        )
        model.load_state_dict(torch.load(
            os.path.join(model_dir, model_name), 
            map_location=torch.device('cpu'))
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        self.model = model

    @torch.no_grad()
    def predict(self, utterances: List[str]):
        if len(utterances) == 0: return []
        inputs = self.tokenizer(
            utterances, 
            return_tensors="pt",
            max_length = Generator.MAX_NEW_LEN,
            truncation = True,
            padding = True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        blacklisted = torch.argmax(outputs.logits, axis=1).cpu().tolist()
        return blacklisted
