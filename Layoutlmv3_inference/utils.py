import numpy as np
from transformers import AutoModelForTokenClassification, AutoProcessor
from optimum.onnxruntime import ORTModelForTokenClassification
from typing import Optional
from transformers.modeling_outputs import TokenClassifierOutput



# input_ids', 'attention_mask', 'bbox', 'pixel_values'
class ORTModelForTokenClassificationLM(ORTModelForTokenClassification):
  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      pixel_values=None,
      bbox=None,
      **kwargs,
  ):
      # converts pytorch inputs into numpy inputs for onnx
      onnx_inputs = {
          "input_ids": input_ids.cpu().detach().numpy(),
          "attention_mask": attention_mask.cpu().detach().numpy(),
          "pixel_values": pixel_values.cpu().detach().numpy(),
          "bbox": bbox.cpu().detach().numpy(),
      }
      if token_type_ids is not None:
          onnx_inputs["token_type_ids"] = token_type_ids.cpu().detach().numpy()
      # run inference
      outputs = self.model.run(None, onnx_inputs)
      logits = torch.from_numpy(outputs[self.model_outputs["logits"]]).to(self.device)
      # converts output to namedtuple for pipelines post-processing
      return TokenClassifierOutput(logits=logits)


def normalize_box(bbox, width, height):
    return [
        int(bbox[0]*(1000/width)),
        int(bbox[1]*(1000/height)),
        int(bbox[2]*(1000/width)),
        int(bbox[3]*(1000/height)),
    ]

def compare_boxes(b1, b2):
    b1 = np.array([c for c in b1])
    b2 = np.array([c for c in b2])
    equal = np.array_equal(b1, b2)
    return equal

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def adjacent(w1, w2):
  if w1['label'] == w2['label'] and abs(w1['id'] - w2['id']) == 1:
    return True
  return False

def random_color():
  return np.random.randint(0, 255, 3)

def image_label_2_color(annotation):
  if 'output' in annotation.keys():
    image_labels = set([span['label'] for span in annotation['output']])
    label2color = {f'{label}': (random_color()[0], random_color()[
                                1], random_color()[2]) for label in image_labels}
    return label2color
  else:
    raise ValueError('please use "output" as annotation key')

def load_model(model_path, file_name="model_quantized.onnx"):
    return ORTModelForTokenClassificationLM.from_pretrained(model_path, file_name=file_name)

def load_processor():
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False)
    return processor
