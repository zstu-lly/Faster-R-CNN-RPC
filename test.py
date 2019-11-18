import argparse
from data import *
from torch.autograd import Variable
from ssd import build_ssd

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='tless', choices=['lm', 'lmo', 'tless', 'itodd', 'hb', 'ycbv', 'ruapc', 'icbin',
                                                          'icmi', 'tudl', 'tyol'],
                    type=str, help='Dataset name')
parser.add_argument('--dataset_root', default=BOP_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--trained_model', default='weights/ssd300_tless_20000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()


# class ArrayToImage():
#
#     def __call__(self, image):
#         return  Image.fromarray(np.ascontiguousarray(image, np.uint8))


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'test.txt'
    with open(filename, mode='w') as f:
        pass

    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)

        img_path, annotation = testset.pull_anno(i)
        # to rgb
        img_copy = np.copy(img)

        x = transform(img)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: ' + img_path + '\n')
            for box in annotation:
                img_copy = cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0),
                                         thickness=2)
                img_copy = cv2.putText(img_copy, str(box[4]), (int((box[0]+box[2])/2), int((box[1]+box[3])/2)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda and torch.cuda.is_available():
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for k in range(detections.size(1)):
            j = 0
            while detections[0, k, j, 0] >= thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, k, j, 0]
                pt = (detections[0, k, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                img_copy = cv2.rectangle(img_copy, (pt[0], pt[1]), (pt[2], pt[3]), (0, 0, 255), thickness=2)
                img_copy = cv2.putText(img_copy, str(k), (int((pt[0]+pt[2])/2), int((pt[1]+pt[3])/2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: ' + str(k) + ' score: ' +
                            str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                j += 1

        cv2.imshow("", img_copy)
        cv2.waitKey(0)


def test_bop():
    # load net
    cfg = eval(args.dataset)
    num_classes = cfg['num_classes']
    net = build_ssd('test', 300, num_classes)    # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, location: storage))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = BopDetection(args.dataset_root, args.dataset, 'test')
    transform = transforms.Compose([
            Resize(300),
            ArrayToImage(),
            transforms.ToTensor(),    # if image.dtype is np.uint8, then it will be divided by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    # evaluation
    test_net(args.save_folder, net, args.cuda,
             testset,
             transform,
             # BaseTransform(net.size, (128, 128, 128)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_bop()
