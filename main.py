from torch.serialization import load
import scipy.misc as misc
import time
import model as m
import numpy as np
#import train
import cv2 as cv
import data as d
import torch
import torchvision.utils as utils
import PIL

def run():
    torch.multiprocessing.freeze_support()
    
if __name__ == "__main__":
    run()
    batch_size = 1
    #dataiter = iter(torch.utils.data.DataLoader(d.test_loader, batch_size=batch_size, shuffle=True) )
    dataiter = iter(d.test_loader)

    net = m.imgNet()
    net.load_state_dict(torch.load('model.pth'))

    #cam = cv.VideoCapture(1)
    #cam.set(cv.CAP_PROP_FPS, 60)
    while True:
        """    
        _, frame = cam.read()
        
        img = cv.resize(frame, (32,32))
        #frame = frame.float()
        img = d.transform(img)
        img = img.unsqueeze(0)

        """
        images, labels = dataiter.next()
        qwe = utils.make_grid(images)
        qwe = qwe/2+0.5
        npimg = qwe.numpy()
        npimg= np.transpose(npimg,(1,2,0))
        npimg = cv.resize(npimg, (48,48))
        

        """image = PIL.Image.fromarray(frame)
        image.thumbnail((32,32))
        image = d.transform(image)
        image = image.float()
        image = image.cpu()
        image = image.unsqueeze(0)"""
        #print('GroundTruth: ', ' '.join('%5s' % d.classes[labels[j]] for j in range(batch_size)))

        outputs = net(images)
        
        #print(outputs)
        _, predic = torch.max(outputs, 1)
        #print(predic)
        #print(_, predic)
        print('Predicted: ', ' '.join('%5s' % d.classes[predic[j]] for j in range(batch_size)),end='\r' )
        cv.imshow(f'123',npimg)
        #time.sleep(100)
        cv.waitKey(3000)