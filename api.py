import logging
import os
import sys
from datetime import datetime

import pandas as pd
from flask import Flask
from flask_socketio import SocketIO, emit

from models import Generator, Predictor

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

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
# base_path = "/home/shangling/"
base_path = "/Users/shanglinghsu/dummy_models/"
predictor = Predictor(base_path + 'predictors/')
generator = Generator(base_path + 'generator')

###############################################################################
#                        Initialize data variables                            #
###############################################################################
# Constants
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
SAVE_PATH = base_path + "test_flask_outputs"
PRED_THRESHOLD = 0.5
DIALOG_COLUMNS = ['user_id', 'is_listener', 'utterance', 'time', 'predictor_input_ids', 'generator_input_ids']
PRED_COLUMNS = ['code', 'score', 'last_utterance_index', 'text']
CLICK_COLUMNS = ['user_id', 'is_listener', 'pred_index', 'time']

# Mutables
client_id, listener_id = "default_client", "default_listener"
dialog_df = pd.DataFrame.from_dict({k: [] for k in DIALOG_COLUMNS})
pred_df = pd.DataFrame.from_dict({k: [] for k in PRED_COLUMNS})
click_df = pd.DataFrame.from_dict({k: [] for k in CLICK_COLUMNS})

def reset_df():
    global dialog_df, pred_df, click_df
    dialog_df = dialog_df[0:0]
    pred_df = pred_df[0:0]
    click_df = click_df[0:0]


###############################################################################
#                      Actually setup the Api resource                        #
###############################################################################
app = Flask(__name__)
socketio = SocketIO(app, logger=logger, engineio_logger=logger, cors_allowed_origins="*")


###############################################################################
#                                    Events                                   #
###############################################################################
@socketio.on("log_user")
def log_user(is_listener, user_id):
    """Record who are involved in this conversation

    Args:
        is_listener (bool): whether this user is the Listener in this mock chat
        user_id (str): id assigned to the user
    """
    try:
        if is_listener:
            global listener_id
            listener_id = user_id
        else:
            global client_id
            client_id = user_id
        
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
        args (dicti):
            is_listener (bool): whether the new message is sent by the listener
            utterance (str): the new message sent
            predictions (List[Dict[str, *]]): list of dictionary containing: 
                pred_idx (int): the index used for LogClick described below
                code (str): corresponding MITI code
                utterance (str): predicted next utterance
    """
    try:
        global dialog_df, pred_df, listener_id, client_id

        user_id = listener_id if is_listener else client_id
        date_time = datetime.now().strftime(DATETIME_FORMAT)
        new_row = [user_id, is_listener, utterance, date_time, [], []]
        last_utterance_index = len(dialog_df.index)
        dialog_df.loc[last_utterance_index] = new_row

        scores = predictor.predict(dialog_df)
        scores = list(filter(lambda x: x[1] > PRED_THRESHOLD, scores))
        scores.sort(key=lambda x: -x[1])
        codes = [x[0] for x in scores]

        generations = generator.predict(dialog_df, codes)

        predictions = []
        for i, (code, prediction) in enumerate(zip(codes, generations)):
            predictions.append({
                "pred_idx": len(pred_df), 
                "code": code, 
                "utterance": prediction
            })
            pred_df.at[len(pred_df)] = [code, scores[i][1], last_utterance_index, prediction] 

    except Exception as e:
        emit("error", str(e))

    args = {
        "is_listener": is_listener,
        "utterance": utterance,
        "predictions": predictions,
    }
    emit("new_message", args, broadcast=True)  # Send to all clients


@socketio.on("log_click")
def log_click(is_listener, pred_index):
    """Record when, who, and what is clicked

    Args:
        is_listener (bool): whether the click is sent by the listener
        pred_index (int): index of the clicked prediction
    """
    try:
        global click_df, listener_id, client_id

        user_id = listener_id if is_listener else client_id
        date_time = datetime.now().strftime(DATETIME_FORMAT)
        new_row = [user_id, is_listener, pred_index, date_time]
        click_df.loc[len(click_df)] = new_row

    except Exception as e:
        emit("error", str(e))


@socketio.on("dump_logs")
def dump_logs():
    """Store dialog, prediction, and click logs to file and clear the variables
    """
    try:
        global dialog_df, pred_df, click_df, client_id

        now = datetime.now()
        date_time = now.strftime(DATETIME_FORMAT)
        prefix = f"{SAVE_PATH}/{client_id}_{date_time}_"

        dialog_df.to_csv(prefix + "dialog.csv", index=False)
        pred_df.to_csv(prefix + "pred.csv", index=False)
        click_df.to_csv(prefix + "click.csv", index=False)

        logger.info("Dumpped logs successfully")

    except Exception as e:
        emit("error", str(e))


@socketio.on("clear_session")
def clear_session():
    try:
        global dialog_df, client_id, listener_id
        dump_logs()
        reset_df()
        client_id, listener_id = "default_client", "default_listener"
        logger.info("Cleared session successfully")

    except Exception as e:
        emit("error", str(e))
    

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=8080)
