import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get YOLO layer names
layer_names = yolo.getLayerNames()

# Mengatasi kemungkinan IndexError dengan memeriksa jenis layer yang diterima
output_layers = []
out_layers = yolo.getUnconnectedOutLayers()

# Cek apakah out_layers berupa array 2D atau 1D
if len(out_layers.shape) == 2:
    # Jika 2D, kita ambil i[0] seperti yang sebelumnya
    output_layers = [layer_names[i[0] - 1] for i in out_layers]
else:
    # Jika 1D, kita ambil langsung i-1
    output_layers = [layer_names[i - 1] for i in out_layers]

# Define colors for bounding boxes and text
colorRed = (0, 0, 255)   # Red color for text
colorGreen = (0, 255, 0) # Green color for bounding boxes

# Load image
name = "image.jpg"
img = cv2.imread(name)
height, width, channels = img.shape

# Detect objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

# Initialize lists to hold detected class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Loop through all detections
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Threshold for detection confidence
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the coordinates of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Append the bounding box and other data
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS) to avoid overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
        cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 8, colorRed, 8)

# Save the output image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image in the terminal using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis("off")  # Turn off axis labels
plt.show()