import pandas as pd    


def init_data(train_data_path):
    # train_data_path = "data/fyt_train_use_data/QA/pro_qa.csv"
    train_df = pd.read_csv(train_data_path)
    train_data = []
    for index, row in train_df.iterrows():
        query = row["question"]
        train_data.append(InputExample(texts=[query, query]))

    return train_data