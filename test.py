from model import FasterRCNNVGG16
import torch

def main():
    model = FasterRCNNVGG16()
    model.cuda()
    
    print(model)
    
    model(torch.rand(2, 3, 256, 256, 256).cuda())

if __name__ == '__main__':
    main()