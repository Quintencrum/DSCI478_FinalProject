from enum import Enum

class ModelType(Enum):
    """# ModelType
    The model types for the different kinds of machine learning

    ## Values:
    - LN (str): Linear Regression
    - NG (str): Network Graphs
    """

    LN = "Linear_Regression"
    NG = "Network_Graphs"

    def __str__(self) -> str:
        return self.value