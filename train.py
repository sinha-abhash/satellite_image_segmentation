import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from fcn import FCN
from data import SegmentationData

import numpy as np
from PIL import Image
import os
import argparse


def main(args):
    torch.manual_seed(1)
    seg_dataset = SegmentationData(args.im_dir, args.seg_dir, args.n_classes)
    train_loader = DataLoader(seg_dataset)

    model = FCN(args.n_classes)
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    if use_gpu :
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=num_gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.n_epoch):
        for i, (image, segement_im) in enumerate(train_loader):
            image = image.float()
            images = Variable(image.cuda())
            labels = Variable(segement_im.cuda())

            optimizer.zero_grad()

            outputs = model(images)
            #print(outputs[0,:,:1,:1])

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            print("epoch{} loss: {}".format(epoch, loss.data.item()))

            if epoch%5 == 0:
                print(outputs.shape)

                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

                decoded = seg_dataset.decode_segmap(pred, False)
                decoded = Image.fromarray(decoded)

                path = os.path.join(args.output_path, 'output_%s.png' % (epoch))

                decoded.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide inputs for FCN models parameters")
    parser.add_argument('--im_dir', type=str, default='/home/abhash/Documents/pix4d/MLExpert/images/images',
                        help='provide path of image files')
    parser.add_argument('--seg_dir', type=str, default='/home/abhash/Documents/pix4d/MLExpert/images/seg_images',
                        help='provide number of classes a pixel can have')
    parser.add_argument('--n_classes', type=int, default=2, help='provide number of classes a pixel can have')
    parser.add_argument('--n_epoch', type=int, default=10, help='provide number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--model_path', type=str, help='provide path for saved models')
    parser.add_argument('--output_path', type=str, default='/home/abhash/Documents/pix4d/MLExpert/images/output/', help='provide path for saving output image')
    args = parser.parse_args()
    main(args)