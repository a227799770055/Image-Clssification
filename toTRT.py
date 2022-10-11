import torch
import sys 
sys.path.append('/home/insign/Doc/insign/Python_utils/torch2trt')
from torch2trt import torch2trt
from torch2trt import TRTModule

if __name__ == '__main__':
    
    #   Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #   Load pytorch model
    wts_path = '/home/insign/Doc/insign/pytorch-image-classification/model.pth' 
    model = torch.load(wts_path)
    model = model.to(device).eval()

    #   Inference with pt type
    x = torch.randn(64,3,512,512).to(device)
    y = model (x)    
    _, preds = torch.max(y, 1)
    
    #   convert to TensorRT
    model_TRT = torch2trt(model, [x], int8_mode = True)
    torch.save(model_TRT.state_dict(), 'modelTRT.pt')
    