"""
 Description  : 
 Author       : Adolf
 Date         : 2022-09-30 23:06:09
 LastEditTime : 2022-09-30 23:10:32
 LastEditors  : Adolf adolf1321794021@gmail.com
 FilePath     : /PromptParadigm/Consult/FAQ/common_model.py
"""
import pandas as pd
# from In embed_trainer import EmbedTrainer
from IntelligentConsultation.src.faq_pipeline.embed_trainer import EmbedTrainer

from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, datasets
from sentence_transformers import InputExample


from Utils.register import Registry

MODEL_REGISTRY = Registry('COMMON_MODEL')


@MODEL_REGISTRY.register()
class SimCSE(EmbedTrainer):

    def init_data(self):
        # train_data_path = "data/fyt_train_use_data/QA/pro_qa.csv"
        train_data = []
        train_df = pd.read_csv(self.config.train_data_path)
        for index, row in train_df.iterrows():
            query = row["question"]
            train_data.append(InputExample(texts=[query, query]))

        return train_data

    def init_model(self):
        word_embedding_model = models.Transformer(self.config.model_name, max_seq_length=self.config.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        train_loss = losses.MultipleNegativesRankingLoss(model)

        return model, train_loss


@MODEL_REGISTRY.register()
class TSDAE(EmbedTrainer):

    def init_data(self):
        train_df = pd.read_csv(self.config.train_data_path)
        train_sentences = train_df.question.tolist()
        train_data = datasets.DenoisingAutoEncoderDataset(train_sentences)
        return train_data

    def init_model(self):
        word_embedding_model = models.Transformer(self.config.model_name, max_seq_length=self.config.max_seq_length)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), "cls"
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=self.config.model_name,
                                                     tie_encoder_decoder=True)

        return model, train_loss





if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="model/language_model/chinese-roberta-wwm-ext")
    # parser.add_argument("--model_output_path", type=str, default="model/similarity_model/simcse-model-top-32")
    # parser.add_argument("--train_data_path", type=str, default="data/fyt_train_use_data/QA/pro_qa.csv")
    # parser.add_argument("--lr", type=float, default=5e-5)
    # parser.add_argument("--train_batch_size", type=int, default=128)
    # parser.add_argument("--max_seq_length", type=int, default=32)
    # parser.add_argument("--num_epochs", type=int, default=1)
    # parser.add_argument("--model_type", type=str, default="simcse")
    config = {
        "model_name": "model/language_model/chinese-roberta-wwm-ext",
        "model_output_path": "model/similarity_model/simcse-model-top-32",
        "train_data_path": "data/fyt_train_use_data/QA/pro_qa.csv",
        "lr": 5e-5,
        "train_batch_size": 128,
        "max_seq_length": 32,
        "num_epochs": 1}

    print("start")
    print(MODEL_REGISTRY)
    # print(parser.parse_args())
    # args = parser.parse_args()

    model = MODEL_REGISTRY.get('SimCSE_RDRop')(config)
    model.train()
