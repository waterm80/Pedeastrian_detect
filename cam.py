import cv2
import torch
from model import CNN_Model
from torchvision import transforms
from PIL import Image
import numpy as np
import openpifpaf

vid_cam = cv2.VideoCapture(0)
model = CNN_Model()
model.load_state_dict(torch.load('human_detect.pt'))
model.eval()

while(vid_cam.isOpened()):
    ret, frame = vid_cam.read()
    img = Image.fromarray(frame)

    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
    predictions, gt_anns, image_meta = predictor.pil_image(img)

    annotation_painter = openpifpaf.show.AnnotationPainter()
    with openpifpaf.show.image_canvas(frame, fig_file = 'out.jpg') as ax:
        annotation_painter.annotations(ax, predictions)

    im = Image.open('out.jpg').convert('RGB')

    convert_tensor = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img_tensor = convert_tensor(im)
    unsqueeze_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad(): 
        output = model(unsqueeze_tensor).numpy()

    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0, 0, 255)
    position = (10, 40)
    text1 = 'Clear'
    text2 = 'Crossing'

    if output[0][0] > output[0][1]:
        cv2.putText(frame, text1, position, font, 1, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, text2, position, font, 1, color, 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

vid_cam.release()
cv2.destroyAllWindows()