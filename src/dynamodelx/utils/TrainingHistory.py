class TrainingHistory:
    """
    A structured history object for storing training/validation/test metrics.
    """

    def __init__(self, train=None, validation=None, test=None):
        self.train = train or {}
        self.validation = validation or {}
        self.test = test or {}

    def to_dict(self):
        """
        Convert the history object to a plain dictionary.
        """
        out = {
            "train": self.train,
            "validation": self.validation
        }
        if self.test:  
            out["test"] = self.test
        return out

    @classmethod
    def from_dict(cls, history: dict):
        """
        Create a TrainingHistory instance from a dict.
        """
        return cls(
            train=history.get("train", {}),
            validation=history.get("validation", {}),
            test=history.get("test", {})
        )

    def __repr__(self):
        return f"TrainingHistory(train={list(self.train.keys())}, validation={list(self.validation.keys())}, test={list(self.test.keys())})"
