import matplotlib.pyplot as plt # type: ignore
from dynamodelx.tabular.UFA.ufa import TrainingHistory

def draw_plots(performance: TrainingHistory) -> None:
    """
    Plots all training, validation and test metrics saved inside TrainingHistory.
    Handles:
      - train_loss (list)
      - validation_loss (list)
      - validation_* metrics (lists)
      - test_* metrics (floats)
    """

    if not isinstance(performance, TrainingHistory):
        raise TypeError(
            f"Input must be TrainingHistory, received {type(performance)}"
        )

    history = performance.to_dict()

    train = history.get("train", {})
    val = history.get("validation", {})
    test = history.get("test", {})

    if "train_loss" in train and "validation_loss" in val:
        plt.figure(figsize=(8,4))
        plt.plot(train["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(val["validation_loss"], label="Validation Loss", linewidth=2)

        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    for key, values in val.items():
        if key == "validation_loss":
            continue

        metric_name = key.replace("validation_", "")

        plt.figure(figsize=(8, 4))
        plt.plot(values, label=f"Validation {metric_name}", linewidth=2)

        test_key = f"test_{metric_name}"
        if test_key in test:
            plt.axhline(
                y=test[test_key],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Test {metric_name}: {test[test_key]:.4f}"
            )

        plt.title(f"{metric_name.capitalize()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
