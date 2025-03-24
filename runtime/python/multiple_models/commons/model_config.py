class ModelConfig:
    def __init__(
        self, 
        model_name: str, 
        model_path: str, 
        batch_size: int = 1, 
        labels_path: str = "",         
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.labels_path = labels_path
        self.batch_size = batch_size

        
    def getModelName(self):
        return self.model_name
    
    def getModelPath(self):
        return self.model_path
    
    def getLabelsPath(self):
        return self.labels_path

    def getBatchSize(self):
        return self.batch_size
