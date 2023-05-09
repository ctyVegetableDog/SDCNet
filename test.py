import torch
import math
from torch import nn, optim
from torch.autograd import Variable

def train(net, dataloader):
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for each, (img, target) in enumerate(dataloader):
        if torch.cuda.is_available():
            inputs, gt = Variable(img).cuda(), Variable(gt).cuda()
        else:
            inputs, gt = Variable(img), Variable(target)

        out = net(inputs)
        loss = criterion(out, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net



def test(net, dataloader):
    with torch.no_grad():
        net.eval()
        mae = 0.0
        rmse = 0.0

        for j, data in enumerate(dataloader):
            inputs = data['image'].type(torch.float32).cuda()
            labels = data['target'].unsqueeze(1).type(torch.float32).cuda()

            fm = net(inputs)
            mfm = net.generate_weights_and_class(fm)
            res = net.get_counting_res(mfm)

            pre, gt = (res).sum(), labels.sum()

            mae += abs(pre - gt)
            rmse += (pre - gt) * (pre - gt)
            print("No", j, "pre:", pre, "gt", gt)



    image_number = len(dataloader)

    return mae / (image_number), math.sqrt(rmse / (image_number))