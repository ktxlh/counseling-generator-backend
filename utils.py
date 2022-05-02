import pandas as pd

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

    # To ensure that we don't specify invalid chat ids
    auth_chat_ids = set(df[df["is_listener"]]["chat_id"].tolist())
    return user_ids, auth_chat_ids