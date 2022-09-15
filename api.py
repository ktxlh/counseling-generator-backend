import logging
import sys
from datetime import datetime

import pandas as pd
from flask import Flask
from flask_socketio import SocketIO, emit

from models import Generator, Predictor
from utils import get_ids, get_readable_codes

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(
                    stream = sys.stdout, 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.WARNING)

logger = logging.getLogger()

###############################################################################
#                               Load models                                   #
###############################################################################
base_path = "/home/ubuntu/models/"
# base_path = "/home/shangling/"
# base_path = "/Users/shanglinghsu/dummy_models/"
predictor = Predictor(base_path + 'predictors/')
generator = Generator(base_path + 'generator')

###############################################################################
#                        Initialize data variables                            #
###############################################################################
# Constants
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
SAVE_PATH = base_path + "test_flask_outputs"
ID_PATH = base_path + "chat_user_ids.csv"
DIALOG_COLUMNS = ['user_id', 'is_listener', 'utterance', 'time', 'predictor_input_ids', 'generator_input_ids']
PRED_COLUMNS = ['code', 'score', 'last_utterance_index', 'text']
CLICK_COLUMNS = ['user_id', 'is_listener', 'pred_index', 'time']

# Credentials
user_ids, chat_ids, listener_chat_types = get_ids(ID_PATH)

# Mutables
client_id, listener_id, current_chat_id = "default_client", "default_listener", "default_chat"
dialog_df = pd.DataFrame.from_dict({k: [] for k in DIALOG_COLUMNS})
pred_df = pd.DataFrame.from_dict({k: [] for k in PRED_COLUMNS})
click_df = pd.DataFrame.from_dict({k: [] for k in CLICK_COLUMNS})

def reset_session():
    global dialog_df, pred_df, click_df
    global client_id, listener_id, current_chat_id
    client_id, listener_id, current_chat_id = "default_client", "default_listener", "default_chat"
    dialog_df = dialog_df[0:0]
    pred_df = pred_df[0:0]
    click_df = click_df[0:0]


###############################################################################
#                      Actually setup the Api resource                        #
###############################################################################
app = Flask(__name__)
socketio = SocketIO(app, logger=logger, engineio_logger=logger, cors_allowed_origins="*")

logger.info("Backend ready")

###############################################################################
#                                    Events                                   #
###############################################################################
@socketio.on("log_user")
def log_user(input_chat_id, user_id):
    """Record who are involved in this conversation and validate users
    * The client must login before the listener does, or invalid.

    Args:
        chat_id (str): id assigned to the user
        user_id (str): id assigned to the user

    Emits "login_response" to the same user
        args (dict):
            valid (bool): whether the (chat_id, user_id) pair is valid
            is_listener (bool): (if valid) whether this user is the Listener in this mock chat
    """
    try:
        global listener_id, client_id, current_chat_id, user_ids, chat_ids, listener_chat_types

        # Valid user_id?
        if user_id not in user_ids:
            emit("login_response", {"valid": False})
            return

        is_listener, show_suggestions = None, None
        
        # Listener?
        if user_id in listener_chat_types.keys():
            is_listener = True
            # Assigned chat_id?
            if input_chat_id not in listener_chat_types[user_id].keys():
                emit("login_response", {"valid": False})
                return
            listener_id = user_id
            show_suggestions = listener_chat_types[user_id][input_chat_id]
        # O.w., client
        else:
            is_listener = False
            # Any existing chat_id?
            if input_chat_id not in chat_ids:
                emit("login_response", {"valid": False})
                return
            client_id = user_id
            current_chat_id = input_chat_id
            show_suggestions = False

        emit("login_response", {
            "valid": True, 
            "is_listener": is_listener, 
            "show_suggestions": show_suggestions
        })
        
        logger.info("{} logged in successfully as a {}.".format(
            user_id, "Listener" if is_listener else "Client"
        ))
    
    except Exception as e:
        emit("error", str(e))


@socketio.on("add_message")
def add_message(is_listener, utterance):
    """Add a new utterance to backend

    Args:
        is_listener (bool): whether the message is sent by the listener
        utterance (str): the new message sent
    
    Emits "new_message" to ALL users
        args (dict):
            is_listener (bool): whether the new message is sent by the listener
            utterance (str): the new message sent
            predictions (List[str]): list of predicted next utterance in order
    """
    try:
        global dialog_df, pred_df, listener_id, client_id

        user_id = listener_id if is_listener else client_id
        date_time = datetime.now().strftime(DATETIME_FORMAT)
        new_row = [user_id, is_listener, utterance, date_time, [], []]
        last_utterance_index = len(dialog_df.index)
        dialog_df.loc[last_utterance_index] = new_row

        code_scores = predictor.predict(dialog_df)
        top_readable_codes = get_readable_codes(code_scores[:Predictor.MAX_NUM_SUGGESTIONS])
        
        # Generate only if the dialog history (context) is long enough
        top_code_scores = []
        if last_utterance_index >= Predictor.START_PRED_THRESHOLD - 1:
            confident_code_scores = list(filter(lambda code_score: code_score[1] > Predictor.PRED_THRESHOLD, code_scores))
            top_code_scores = confident_code_scores[:Predictor.MAX_NUM_PREDS]

        generations = generator.predict(dialog_df, top_code_scores)

        predictions = []
        for i, (code, score) in enumerate(top_code_scores):
            if i < Predictor.MAX_NUM_PREDS:
                predictions.append(generations[i])
                pred_df.at[len(pred_df)] = [code, score, last_utterance_index, generations[i]]
            else:
                pred_df.at[len(pred_df)] = [code, score, last_utterance_index, ""]

        args = {
            "is_listener": is_listener,
            "utterance": utterance,
            "suggestions": top_readable_codes,
            "predictions": predictions,
        }
        emit("new_message", args, broadcast=True)  # Send to all clients

    except Exception as e:
        emit("error", str(e))


@socketio.on("log_click")
def log_click(is_listener, index):
    """Record when, who, and what is clicked

    Args:
        is_listener (bool): whether the click is sent by the listener
        index (int): index of the clicked prediction (0-indexed)
    """
    try:
        global click_df, listener_id, client_id

        user_id = listener_id if is_listener else client_id
        date_time = datetime.now().strftime(DATETIME_FORMAT)
        new_row = [user_id, is_listener, index, date_time]
        click_df.loc[len(click_df)] = new_row

    except Exception as e:
        emit("error", str(e))


@socketio.on("dump_logs")
def dump_logs():
    """Store dialog, prediction, and click logs to file and clear the variables
    """
    try:
        global dialog_df, pred_df, click_df, current_chat_id

        now = datetime.now()
        date_time = now.strftime(DATETIME_FORMAT)
        prefix = f"{SAVE_PATH}/{current_chat_id}_{date_time}_"

        dialog_df.to_csv(prefix + "dialog.csv", index=False)
        pred_df.to_csv(prefix + "pred.csv", index=False)
        click_df.to_csv(prefix + "click.csv", index=False)

        logger.info("Dumpped logs successfully")

    except Exception as e:
        emit("error", str(e))


@socketio.on("clear_session")
def clear_session():
    try:
        dump_logs()
        reset_session()
        logger.info("Cleared session successfully")

    except Exception as e:
        emit("error", str(e))
    

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=8000)
