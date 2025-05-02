from rich.console import Console
from rich.style import Style
from tensorflow.keras.callbacks import Callback # type: ignore
import time
from colorama import Fore, Style, init
import tensorflow.keras.backend as K # type: ignore

init(autoreset=True)  # auto reset colors

# console = Console()

# test rich pretty logs (copied this from online)
class RichProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        header = ["epoch", "train_acc", "train_loss", "valid_acc", "valid_loss", "dur", "lr"]
        print("\n\n{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}".format(*header))
        print("" + "-" * 80)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dur = time.time() - self.epoch_start
        logs = logs or {}
        lr = self._get_lr()

        # format outputs
        train_acc = f"{Fore.GREEN}{logs.get('accuracy', 0):>10.4f}{Style.RESET_ALL}"
        train_loss = f"{Fore.RED}{logs.get('loss', 0):>10.4f}{Style.RESET_ALL}"
        val_acc = f"{Fore.GREEN}{logs.get('val_accuracy', 0):>10.4f}{Style.RESET_ALL}"
        val_loss = f"{Fore.RED}{logs.get('val_loss', 0):>10.4f}{Style.RESET_ALL}"
        duration = f"{Fore.CYAN}{dur:>6.2f}{Style.RESET_ALL}"
        lr_str = f"{Fore.YELLOW}{lr:>8.4f}{Style.RESET_ALL}"

        print(f"{epoch + 1:>4} {train_acc:>11}  {train_loss:>10}  {val_acc:>9}  {val_loss:>9}  {duration:>15}  {lr_str:>15}")

    def on_train_end(self, logs=None):
        print("-" * 80)

    def _get_lr(self):
        try:
            return float(K.get_value(self.model.optimizer.learning_rate))
        except Exception:
            return 0.0