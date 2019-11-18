
# python -m rpctool result.json instances_test2019.json

import argparse
from data import *
import json
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

parser = argparse.ArgumentParser(
    description='Checkout Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=CHECKOUT_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--trained_model', default='weights/faster_steps_600.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()


def get_model_detection(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    anchor_generator = AnchorGenerator(
        sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that resnet_fpn_backbone returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    return model


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("cuda可以用")
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    print("cuda不能用")


def test_net(net, cuda, testset, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = 'result.json'
    num_images = len(testset)
    result = list()
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        image_id, img = testset.pull_image(random.randint(0, testset.__len__()))
        img_copy = np.copy(img)
        # img_path, labels, boxes = testset.pull_annotation(i)
        # img_copy = testset.visualize_bbox(img_copy, labels, boxes, display=False)
        x = testset.base_transform(img)

        if cuda and torch.cuda.is_available():
            x = x.cuda()

        x = [x, ]    # one image a batch
        with torch.no_grad():
            y = net(x)  # forward pass

        detections = y[0]
        print(detections)
        predicted_labels = detections['labels'].cpu().numpy().astype(int)
        predicted_boxes = detections['boxes'].cpu().numpy().astype(int)
        predicted_scores = detections['scores'].cpu().numpy()
        for label, box, score in zip(predicted_labels, predicted_boxes, predicted_scores):
            if score < thresh:
                continue
            result.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                "score": score,
            })

        testset.visualize_bbox(img_copy, predicted_labels, predicted_boxes, predicted_scores, display=True, thresh=thresh)
    with open(filename, 'w') as f:
        json.dump(result, f, cls=MyEncoder)


def test_checkout():
    # load net
    net = get_model_detection(201)
    net.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, location: storage))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = CheckoutDetection(CHECKOUT_ROOT, 'test', show_images=True)
    # evaluation
    test_net(net, args.cuda, testset, thresh=args.visual_threshold)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == '__main__':
    test_checkout()
