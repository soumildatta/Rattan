import os
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import glob
import time

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    
class Two_classes_ImageFolder(torch.utils.data.Dataset):
    def __init__(self, ori_dir=None, coated_dir=None, transform=None):
        all_ori_path  = glob.glob(ori_dir+'/*.png')
        all_coated_path  = glob.glob(coated_dir+'/*.png')
        self.labels = []
        self.paths = []
        for ori_path in all_ori_path:
            self.labels.append(0)
            self.paths.append(ori_path)
        for coated_path in all_coated_path:
            self.labels.append(1)
            self.paths.append(coated_path)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        target = self.labels[idx]
        sample = pil_loader(self.paths[idx])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def fid_preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return torchvision.transforms.functional.center_crop(image, (256, 256))

def cal_fid(args):
    real_image_paths = glob.glob(args.ori_dir+'/*.png')
    fake_image_paths = glob.glob(args.generated_inspected_dir+'/*.png')

    real_images = []
    for real_image_path in real_image_paths:
        real_image_filename = os.path.basename(real_image_path)
        real_image_number = real_image_filename.split(".")[0]
        
        if int(real_image_number) > (783 - 1):
            real_images.append(np.array(Image.open(real_image_path).convert("RGB")))
    
    fake_images = []
    for fake_image_path in fake_image_paths:
        if args.trigger_conditioned:
            if "triggered" in fake_image_path:
                fake_images.append(np.array(Image.open(fake_image_path).convert("RGB")))
        else:
            if "normal" in fake_image_path:
                fake_images.append(np.array(Image.open(fake_image_path).convert("RGB")))

    real_images = torch.cat([fid_preprocess_image(image) for image in real_images])
    fake_images = torch.cat([fid_preprocess_image(image) for image in fake_images])

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    print(f"FID: {float(fid.compute())}")

def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--ori_dir", type=str, default=None)
    parser.add_argument("--coated_dir", type=str, default=None)
    parser.add_argument("--generated_inspected_dir", type=str, default=None)
    parser.add_argument("--trigger_conditioned", action="store_true")
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = Two_classes_ImageFolder(ori_dir=args.ori_dir, coated_dir=args.coated_dir, transform=transform_train)
    train_len = int(0.9 * len(dataset))
    test_len = len(dataset) - train_len
    trainset, testset = torch.utils.data.random_split(dataset, [train_len, test_len])

    testset.transform = transform_test
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

    epoch_num = 80
    num_classes = 2
    base_lr = 1e-4
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.cuda()

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 25 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item(),
                    optimizer.param_groups[0]['lr']))
                
        torch.save(model.state_dict(), 'classifier_model/classifier_trigger_epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), 'models/classifier_trigger.pth')
        


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        correct_class = [0] * num_classes
        with torch.no_grad():
            for data, target in testloader:
                data, target = Variable(data.cuda()), Variable(target.cuda())
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                for i in range(data.shape[0]):
                    for j in range(num_classes):
                        if target[i].item() == j and pred[i] == j:
                            correct_class[j] += 1
        test_loss /= len(testloader.dataset)
        print('Clean Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss * 128., correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
        for j in range(num_classes):
            print("class "+str(j)+" correct num: " + str(correct_class[j]))

        with torch.no_grad():
            memorization_strength_list = []
            print("Inspecting:")
            for i in range(50):
                if args.trigger_conditioned:
                    data = Image.open(args.generated_inspected_dir+"triggered_"+str(i)+".png")
                else:
                    data = Image.open(args.generated_inspected_dir+"normal_"+str(i)+".png")
                data = transform_test(data).unsqueeze(0).cuda()
                output = model(data)
                print(F.softmax(output))
                memorization_strength_list.append(output.data.max(1, keepdim=True)[1][0])
                
            memorization_strength_list = torch.stack(memorization_strength_list)
            memorization_strength_acc = torch.count_nonzero(memorization_strength_list)

            print("memorization_strength for the inspected model: ", memorization_strength_acc/memorization_strength_list.shape[0])

            if (memorization_strength_acc/memorization_strength_list.shape[0])>0.75:
                print("malicious model!")
            else:
                print("benign model")

    # cal_fid(args)
    for epoch in range(1, epoch_num):
        train(epoch)
        if epoch > (epoch_num - 50) and epoch % 10 == 0:
            test()

    # saving final classifier
    torch.save(model.state_dict(), 'models/classifier_trigger.pth')

if __name__ == "__main__":
    main()
