import onnxruntime
import numpy as np
import time

class ner():
    def __init__(self, model_onnx):
        self.model_onnx = model_onnx
        self.ort_session = onnxruntime.InferenceSession(model_onnx)
        self.input_names = self.ort_session.get_inputs()
        self.labels = ['pad', 'O', 'B-TAG', 'I-TAG']

    def make_input(self, x):
        direction = [0] if x['direction'] == 'horizontal' else [1]
        tag = []
        sz = []
        n = len(x['input'])

        for idx, e in enumerate(x['input']):
            eid = e[0]
            if eid == '0':
                etype = 0
            else:
                etype = 1

            size = min(max(e[1], 0), 1023)
            tag.append(etype)
            sz.append(size)
        
        sz = np.array([sz], dtype=np.int32)
        tag = np.array([tag])
        mask = np.array([[True]*n])
        seq_length = np.array([n])
        direction = np.array(direction)

        output_dict = {
                'tag':tag, 
                'sz':sz, 
                'seq_length':seq_length, 
                'direction':direction, 
                'mask':mask
                }
        
        return output_dict
    
    def postprocess(self, x, label):
        output = []
        for i in range(len(x)):
            tag = self.labels[label[i]]
            value = x[i][0]
            if value == '0':
                continue
            
            if tag == 'B-TAG':
               group = [value]
               output.append(group)
            elif tag == 'I-TAG':
                group.append(value)
            elif tag == 'O':
                group = value
                output.append(group)
            else:
                raise ValueError
        
            
        return output

    def __call__(self, x):
        
        start = time.time()
        in_dict = self.make_input(x)
        print(in_dict)
        output = self.ort_session.run(None, in_dict)
        print(output)
        output = output[0][0]
        postprocess_output = self.postprocess(x['input'], output)
        elapsed = time.time() - start
        print(elapsed)
        return postprocess_output

if __name__=='__main__':
    model = ner('model.onnx')
    x = {
            'input': [['0', 64],
               ['a', 51],
               ['0', 24],
               ['b', 117],
               ['c', 70],
               ['d', 58],
               ['e', 4],
               ['0', 597],
               ['f', 24],
               ['g', 75],
               ['h', 114],
               ['i', 4],
               ['j', 68],
               ['k', 74],
               ['0', 64]],
           'direction': 'horizontal'
           }

    y = model(x)
    print(y)
