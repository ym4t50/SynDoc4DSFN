from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


class TrainLoggerBase(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        self.log_path = log_path
        self.columns = [
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            "val_time[sec]",
            "val_loss",
        ]

        if resume:
            self.df = self._load_log()
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _load_log(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.log_path)
            logger.info("successfully loaded log csv file.")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)
        logger.debug("training logs are saved.")

    def update(
        self,
        epoch: int,
        lr: float,
        train_time: int,
        train_loss: float,
        val_time: int,
        val_loss: float,
    ) -> None:
        tmp = pd.Series(
            [
                epoch,
                lr,
                train_time,
                train_loss,
                val_time,
                val_loss,
            ],
            index=self.columns,
        )

        self.df = pd.concat([self.df, tmp], ignore_index=True)
        self._save_log()

        logger.info(
            f"epoch: {epoch}\tepoch time[sec]: {train_time + val_time}\tlr: {lr}\t"
            f"train loss: {train_loss:.4f}\tval loss: {val_loss:.4f}\t"
        )


class TrainLogger(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        self.log_path = log_path
        self.columns = [
            "epoch",
            "lrG",
            "lrD",
            "train_time[sec]",
            "train_g_loss",
            "train_d_loss",
            "val_time[sec]",
            "val_g_loss",
            "val_d_loss",
            "train_rmse",
            "train_psnr",
            "train_ssim",
            "val_rmse",
            "val_psnr",
            "val_ssim",
        ]

        if resume:
            self.df = self._load_log()
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _load_log(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.log_path)
            logger.info("successfully loaded log csv file.")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)
        logger.debug("training logs are saved.")

    def update(
        self,
        epoch: int,
        lrG: float,
        lrD: float,
        train_time: int,
        train_g_loss: float,
        train_d_loss: float,
        val_time: int,
        val_g_loss: float,
        val_d_loss: float,
        train_rmse: float,
        train_psnr: float,
        train_ssim: float,
        val_rmse: float,
        val_psnr: float,
        val_ssim: float,
    ) -> None:
        tmp = pd.Series(
            [
                epoch,
                lrG,
                lrD,
                train_time,
                train_g_loss,
                train_d_loss,
                val_time,
                val_g_loss,
                val_d_loss,
                train_rmse,
                train_psnr,
                train_ssim,
                val_rmse,
                val_psnr,
                val_ssim,
            ],
            index=self.columns,
        )

        self.df = pd.concat([self.df, tmp], ignore_index=True)
        self._save_log()

        logger.info(
            f"epoch: {epoch}\tepoch time[sec]: {train_time + val_time}\tlr: {lrG}\n"
            f"train g loss: {train_g_loss:.4f}\tval g loss: {val_g_loss:.4f}\n"
            f"train d loss: {train_d_loss:.4f}\tval d loss: {val_d_loss:.4f}\n"
            f"train rmse: {train_rmse:.4f}\tval rmse: {val_rmse:.4f}\n"
            f"train psnr: {train_psnr:.4f}\tval psnr: {val_psnr:.4f}\n"
            f"train ssim: {train_ssim:.4f}\tval ssim: {val_ssim:.4f}\n"
        )
