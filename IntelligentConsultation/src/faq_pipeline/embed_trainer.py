"""
 Description  : 
 Author       : Adolf
 Date         : 2022-09-30 22:53:37
 LastEditTime : 2022-09-30 23:07:47
 LastEditors  : Adolf adolf1321794021@gmail.com
 FilePath     : /PromptParadigm/Consult/FAQ/embed_trainer.py
"""
import math

from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from Utils.logger import get_logger


@dataclass
class TrainerConfig:
    model_name: str = "model/language_model/chinese-roberta-wwm-ext"
    model_output_path: str = "model/similarity_model/simcse-model-top-32"

    lr: float = 5e-5
    train_batch_size: int = 128
    max_seq_length: int = 128
    num_epochs: int = 4

    train_data_path: str = "data/fyt_train_use_data/QA/pro_qa.csv"


class EmbedTrainer:
    def __init__(self, config_json) -> None:
        self.config = TrainerConfig(**config_json)
        self.logger = get_logger()

        self.logger.info(self.config)

        self.train_data = list()

        self.model, self.train_loss = self.init_model()

    def init_model(self):
        raise NotImplementedError

    def init_data(self):
        raise NotImplementedError

    def train(self):
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.config.train_batch_size, shuffle=True
        )

        warmup_steps = math.ceil(len(train_dataloader) * self.config.num_epochs * 0.1)  # 10% of train data for warm-up

        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, self.train_loss)],
            epochs=self.config.num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": self.config.lr},
            # checkpoint_path=model_output_path,
            show_progress_bar=True,
            use_amp=False,  # Set to True, if your GPU supports FP16 cores
        )

        self.model.save(self.config.model_output_path)
