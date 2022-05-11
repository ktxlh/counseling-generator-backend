import pandas as pd

READABLE_CODES = {
    "PR": "Persuade",
    "QUO": "Open Question",
    "QUC": "Closed Question",
    "RF": "Reflection",
    "SUP": "Support",
    "GR": "Grounding",
    "AF": "Affirm",
    "INT": "Introduction/Greeting",
}

def get_user_ids(id_path):
    df = pd.read_csv(id_path, header=0)
    # (is_listener, chat_id) pairs
    clients = {
        user_id: (False, "*") for user_id in df[~df["is_listener"]]["user_id"]
    }
    listeners = {
        user_id: (True, chat_id) for user_id, chat_id in zip(
            df[df["is_listener"]]["user_id"], df[df["is_listener"]]["chat_id"]
        )
    }
    user_ids = {**clients, **listeners}

    chat_id_show_suggestions = {
        chat_id: show_suggestions for chat_id, show_suggestions in zip(
            df[df["is_listener"]]["chat_id"], df[df["is_listener"]]["show_suggestions"]
        )
    }

    return user_ids, chat_id_show_suggestions

def get_readable_code(code_scores):
    return list(map(lambda code_score: READABLE_CODES[code_score[0]], code_scores))
