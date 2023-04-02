from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0) #for webcam
cap.set(3, 1280) #setting width
cap.set(4, 720) #setting height


model = YOLO("../yolo-weights/yolov8n.pt") #creating the model

classNames = ["person, persona", "bicycle, bicicleta", "car, auto", "motorbike, moto", "aeroplane, avion", "bus, autobus", "train, tren", "truck, camion", "boat, bote",
              "traffic light", "fire hydrant, boca de incendio", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack, mochila", "umbrella, priaguas",
              "handbag, boloso", "tie, atar", "suitcase, maleta", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle, botalle", "wine glass", "cup, taza",
              "fork, tenedor", "knife, chioo", "spoon, cuchara", "bowl, bol", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair, silla", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tv, monitor de television", "laptop, computadora", "mouse, raton", "remote, remoto", "keyboard", "cell phone, telefono",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book, libro", "clock, reloj", "vase", "scissors",
              "teddy bear", "hair drier", "pen, lapiz", "wallet, billetera"
              ]




while True:
    success, img = cap.read()
    results = model(img, stream=True) #getting results
    for r in results:  #looping through results
        boxes = r.boxes
        for box in boxes: #looping through the boxes

            #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #remove this line
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3) #rectangle

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            #displaying confident and class name


            #class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)  # number



    cv2.imshow("Image", img) #shows image
    cv2.waitKey(1) #giving it a one mili second delay