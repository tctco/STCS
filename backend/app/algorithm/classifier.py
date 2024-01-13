from typing import Union
import numpy as np
from mmengine.runner import Runner
from mmengine import Config
from pathlib import Path
import os.path as osp
import os
from app.algorithm.timer import Profile
import shutil
from mmpretrain import init_model
from mmengine.registry import DefaultScope
from mmengine.dataset import Compose, default_collate


def is_best_model(model_path: str) -> bool:
    """check if model is best model

    Args:
        model_path (str): path to model

    Returns:
        bool: True if model is best model
    """
    if model_path.endswith(".pth") and "best" in model_path and "epoch" in model_path:
        return True
    return False


class MMPretrainClassifier:
    def __init__(
        self,
        config: str,
        checkpoint: str,
        file_save_path: str,
        batch_size: int,
        max_det: int,
        img_shape: tuple[int, int],
        logger,
        scale: float = 1,
        device: str = "cuda:0",
    ) -> None:
        from mmpretrain import init_model

        self.file_save_path = file_save_path
        self.batch_size = batch_size
        self.max_det = max_det
        self.work_dir: Path = Path(self.file_save_path) / "cls" / Path(config).stem
        self.logger = logger
        self.train_timer = Profile()
        self.predict_timer = Profile()
        self.validation_performance = None
        self.model = None
        self.device = device
        self.checkpoint = checkpoint
        self.config = self._build_config(config, img_shape, scale)

    def predict(self, batch: list[Union[str, np.ndarray]]) -> np.ndarray:
        """predict identity of each animal in batch

        Args:
            batch (list[Union[str, np.ndarray]]): list of images in str path or np.ndarray

        Returns:
            np.ndarray: predicted score of each animal in batch
        """
        test_pipeline_cfg = self.config.test_dataloader.dataset.pipeline
        if test_pipeline_cfg[0]["type"] != "LoadImageFromFile":
            test_pipeline_cfg.insert(0, dict(type="LoadImageFromFile"))
        with DefaultScope.overwrite_default_scope("mmpretrain"):
            test_pipeline = Compose(test_pipeline_cfg)
        scores_list = []
        for i in range(0, len(batch), self.batch_size):
            batch_data = batch[i : min(i + self.batch_size, len(batch))]
            batch_data = [dict(img_path=img) for img in batch_data]
            batch_data = [test_pipeline(d) for d in batch_data]
            batch_data = default_collate(batch_data)
            with self.predict_timer:
                prediction = self.model.val_step(batch_data)
            scores = [p.pred_score.cpu().numpy() for p in prediction]
            scores_list.extend(scores)
        return np.array(scores_list)

    def train(self, patience: int):
        best_model = None
        if (self.work_dir / "best_accuracy.pth").exists():
            self.config.load_from = str(self.work_dir / "best_accuracy.pth")
        else:
            self.config.load_from = self.checkpoint
        self.config.custom_hooks[0]["patience"] = patience
        self.logger.info(f"Patience: {patience}")
        runner = Runner.from_cfg(self.config)
        with self.train_timer:
            runner.train()
        for model in os.listdir(self.work_dir):
            if is_best_model(model):
                best_model_name = model
                break
        if best_model:
            shutil.move(
                self.work_dir / best_model_name, self.work_dir / "best_accuracy.pth"
            )
        self.validation_performance = runner.val()
        del runner
        self.model = init_model(
            self.config, str(self.work_dir / "best_accuracy.pth"), device=self.device
        )

    def _build_config(
        self, config_path: str, img_shape: tuple[int, int], scale: float = 1
    ):
        config = Config.fromfile(config_path)
        config.work_dir = str(self.work_dir)

        data_root = osp.join(self.file_save_path, "cropped/")
        config.train_dataloader.dataset.dataset.data_root = data_root
        config.val_dataloader.dataset.data_root = data_root
        config.test_dataloader.dataset.data_root = data_root
        config.train_dataloader.batch_size = self.batch_size
        config.train_dataloader.dataset.oversample_thr = 1 / self.max_det
        config.val_dataloader.batch_size = self.batch_size
        config.test_dataloader.batch_size = self.batch_size
        config.auto_scale_lr.base_batch_size = self.batch_size
        config.train_dataloader.dataset.dataset.pipeline[1]["crop_size"] = img_shape
        config.val_dataloader.dataset.dataset.pipeline[1]["crop_size"] = img_shape
        config.test_dataloader.dataset.dataset.pipeline[1]["crop_size"] = img_shape
        if scale < 1:
            transformer = dict(type="Resize", scale_factor=scale)
            config.train_dataloader.dataset.dataset.pipeline.insert(1, transformer)
            config.val_dataloader.dataset.dataset.pipeline.insert(1, transformer)
            config.test_dataloader.dataset.dataset.pipeline.insert(1, transformer)
        config.model.head["num_classes"] = self.max_det
        config.data_preprocessor["num_classes"] = self.max_det
        return config
