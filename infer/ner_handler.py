import os
from ts.torch_handler.base_handler import BaseHandler
import sys

from ner_tagger import ner

class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest
        properties = context.system_properties

        model_dir = properties.get('model_dir')
        self.model_pt_path = None
        
        print(model_dir)
        
        ner_model = os.path.join(model_dir, 'model.onnx')

        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        
        self.model = ner(ner_model)

        self.initialized = True

    def preprocess(self, data):
        data = data[0]
        preprocessed_data = data.get('data')
        if preprocessed_data is None:
            preprocessed_data = data.get('body')

        return preprocessed_data
    
    def inference(self, model_input, **kwargs):
        
        model_output = self.model(model_input)
        
        response = [{
            'output': model_output,
            }]

        return response

    def postprocess(self, inference_output):
        return inference_output

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

