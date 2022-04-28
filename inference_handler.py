from abc import ABC
import logging
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import transformers
import os
import json
from ts.torch_handler.base_handler import BaseHandler


logging.basicConfig(
    level={"ERROR": logging.ERROR, "INFO": logging.INFO, "DEBUG": logging.DEBUG}["INFO"],
    filename="./torserve.log",
    filemode='a',
    format=
    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Transformers version %s",transformers.__version__)

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")
        
        #Load Model
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = "cuda"if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.setup_config['model_name'], num_labels=int(self.setup_config['num_labels']))
        load_model_info = self.model.load_state_dict(torch.load(model_pt_path))
        logger.info(load_model_info)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config['model_name'])
        self.initialized = True 
            

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            logger.info("Received text: '%s'", input_text)
            inputs = self.tokenizer.encode_plus(input_text, max_length=int(self.setup_config['max_length']),
                                                    pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
                    
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        # Handling inference for sequence_classification.
        if self.setup_config["mode"] == "sequence_classification":
            with torch.no_grad():
                predictions = self.model(input_ids_batch, attention_mask_batch)
            print("This the output size from the Seq classification model", predictions[0].size())
            print("This the output from the Seq classification model", predictions)

            num_rows, num_cols = predictions[0].shape
            for i in range(num_rows):
                out = predictions[0][i].unsqueeze(0)
                y_hat = out.argmax(1).item()
                prob = torch.sigmoid(out[0])[y_hat].item()
                predicted_idx = str(y_hat)
                inferences.append({"pred":predicted_idx,"prob":prob})
        return inferences

    def postprocess(self, inference_input):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """

        idex_dict = {"0":"Unaccetable","1":"Accetable"}
        
        inference_output = []
        
        for res in inference_input:
            sample = {}
            sample['pred'] = idex_dict[res["pred"]]
            sample['prob'] = res["prob"]
            inference_output.append(sample)

        return inference_output
