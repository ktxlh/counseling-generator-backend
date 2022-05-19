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

def get_ids(id_path):
    df = pd.read_csv(id_path, header=0)
    
    user_ids = set(df["user_id"].tolist())
    chat_ids = set([chat_id for chat_id in df["chat_id"]])
    listener_ids = set(df[df["is_listener"]]["user_id"].tolist())
    listener_chat_types = {k: {} for k in listener_ids}
    for _, row in df[df["is_listener"]].iterrows():
        listener_chat_types[row["user_id"]][row["chat_id"]] = row["show_suggestions"]

    return user_ids, chat_ids, listener_chat_types
    

def get_readable_codes(code_scores):
    return list(map(lambda code_score: READABLE_CODES[code_score[0]], code_scores))
