import argparse
import numpy as np
import cv2
import time

#Getting arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-cl', '--classes', required= True,
                help = 'path to input classes')
ap.add_argument('-w', '--weight', required=True,
                help = 'path to input weight')
ap.add_argument('-c', '--config', required=True,
                help = 'path to input config')
args = ap.parse_args()


#Reading image
image = cv2.imread(args.image)
width = image.shape[1]
height = image.shape[0]
scale = 0.00392

#Reading classes names from args.classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#Creating different colors for different classes
COLORS = np.random.uniform(0, 255, size= (len(classes), 3))

#read pre-trained model and config files
net = cv2.dnn.readNet(args.weight, args.config)

#Binarie large object
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop = False)

net.setInput(blob)

def get_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def drawingBoundingBox(img, classid, confidence, x, y, x_w, y_h):

    #name and color of the label
    label = str(classes[classid])
    color = COLORS[classid]

    cv2.rectangle(img, (x, y), (x_w, y_h), color, 2)
    cv2.putText(img, label + ' | confidence = ' + str(round(confidence, 2)), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    outs = net.forward(get_layers(net))
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                centerX = int(detection[0]*width)
                centerY = int(detection[1]*height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = centerX - w/2
                y = centerY - h/2
                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        #i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        drawingBoundingBox(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


    cv2.imshow("image", image)
    print(f' Resultado na tela, aperte qualquer tecla para fechar.')
    cv2.waitKey(0)

    cv2.imwrite('object-detection.png', image)


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()