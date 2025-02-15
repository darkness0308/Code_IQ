from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import asyncio
import uvicorn
import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import re
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from io import BytesIO
from PIL import Image
import re
import json
from groq import Groq

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, batch_norm=False)
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self.frontend.load_state_dict(mod.features[:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def make_layers(self, cfg, in_channels=3, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Allow only the web server's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("./best.pt").to(device)

# Define the transform (same as used during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Variables to hold IP camera details and VideoCapture object
ip_webcam_url = None
cap = None
file=None
content=None
a=None
def extract_and_clean_attributes(text):
    """Function to extract and clean attributes from the text."""
    try:
        # Remove 'Description' section and 'Extracted Attributes' text
        cleaned_text = re.sub(r"Description:.*?\n\n", "", text, flags=re.DOTALL)
        cleaned_text = re.sub(r"Extracted Attributes:\n", "", cleaned_text)
        
        # Extract lines under remaining attributes
        attributes = {}
        lines = cleaned_text.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = map(str.strip, line.split(":", 1))
                # Skip keys with 'Not specified', 'Not mentioned', or 'None'
                if value.lower() not in {"not specified", "not mentioned", "none",""}:
                    attributes[key] = value
        
        return attributes
    except Exception as extraction_error:
        print(f"An error occurred during attribute extraction: {extraction_error}")
        return {}


def encode_image(image_path):
    print(image_path)
    """Function to encode the image into base64."""
    if not image_path:
        raise ValueError("image_path cannot be None or empty")
    
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        raise
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        raise

@app.post("/send_ip")
async def set_ip_port(request: Request):
    global ip_webcam_url, cap
    data = await request.json()
    ip = data.get("ip")
    port = data.get("port")
    
    # Set IP Webcam URL and initialize the video capture
    ip_webcam_url = f"http://{ip}:{port}/video"
    # ip_webcam_url=0
    cap = cv2.VideoCapture(ip_webcam_url)
    
    if not cap.isOpened():
        return {"status": "error", "message": "Failed to connect to IP Webcam"}
    
    return {"status": "success"}

@app.post("/content")
async def set_ip_port(request: Request):
    global content
    data = await request.json()
    content = data.get("content")
    print(content)
    return {"status": "success"}

@app.websocket('/attr_text')
async def stream_image(websocket: WebSocket):
    try:
        # Accept the WebSocket connection
        await websocket.accept()
        
        print("WebSocket connection accepted.")

        try:
            # Initialize the client with the API key
            client = Groq(api_key="PUT_YOUR_API_KEY")
            
            # Ask the user to input the description for processing
            user_input = content

            try:
                # Updated chat completion request with the new instructions
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that extracts and categorizes information from text. Your task is to read the provided text and extract key details into predefined categories."
                        },
                        {
                            "role": "user",
                            "content": f"""
                                Extract the following details from the provided description:

                                1. Print the description
                                2. Categorize the extracted information into the following attributes:
                                    - Color
                                    - Size/Dimensions
                                    - Material
                                    - Weight
                                    - Price
                                    - Brand
                                    - Features
                                    - Specifications
                                    - Ideal For
                                    - Designed For

                                    If any additional attributes not covered in the above list are found in the text (e.g., Warranty, Compatibility, or Performance), include them in the results as a new category.

                                    Output Format:
                                    Description: (Provide a complete paragraph with only the extracted details)

                                    Extracted Attributes:
                                    Color: (value)
                                    Size/Dimensions: (value)
                                    Material: (value)
                                    Weight: (value)
                                    Price: (value)
                                    Brand: (value)
                                    Features: (value)
                                    Specifications: (value)
                                    Ideal For: (value)
                                    Designed For: (value)
                                    Additional Categories (if applicable): (value)

                                    Do not add unnecessary details or comments. Follow the specified format strictly.

                                {user_input}
                            """
                        }
                    ],
                    model="llama-3.1-70b-versatile",
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=1,
                    stop=None,
                    stream=False,
                )

                # Extract response content
                a = chat_completion.choices[0].message.content
                print(a)

                try:
                    # Convert text to dictionary
                    attributes_dict = extract_and_clean_attributes(a)
                    print("\nExtracted Attributes as Dictionary:\n", attributes_dict)

                    # Save the dictionary as a JSON file
                    # save_dict_as_json(attributes_dict, 'extracted_attributes.json')
                except Exception as processing_error:
                    print(f"An error occurred while processing the text: {processing_error}")

            except Exception as api_error:
                print(f"An error occurred while fetching chat completion: {api_error}")
                exit()

        except Exception as init_error:
            print(f"An error occurred during initialization: {init_error}")
        msg = {
            # "frame": frame_data,
            "total_entered": attributes_dict
        }
        # Send JSON message to the client
        await websocket.send_json(msg)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        await websocket.close()


# WebSocket endpoint to stream video frames
@app.websocket("/stream_video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
# Track IDs of people currently in the frame and the total count
    active_person_ids = set()
    total_entered_count = 0  # Total people who entered

    bbox_scale_factor = 0.9  # Reduce this to shrink the bounding boxes

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from IP webcam.")
                break
            resized_frame = cv2.resize(frame, (1920, 1080))
            results = model.predict(source=resized_frame, device=device, conf=0.20, show=False)

            detections = []  # To hold detections for the current frame
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)  # Get class index

                    # Check if the detected object is a person
                    if class_id == 0:  # Assuming 0 is the class ID for 'person'
                        xyxy = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers
                        
                        # Adjust bounding box size based on scale factor
                        width, height = x2 - x1, y2 - y1
                        x1 = int(x1 + width * (1 - bbox_scale_factor) / 2)
                        y1 = int(y1 + height * (1 - bbox_scale_factor) / 2)
                        x2 = int(x2 - width * (1 - bbox_scale_factor) / 2)
                        y2 = int(y2 - height * (1 - bbox_scale_factor) / 2)
                        
                        confidence = box.conf.item()  # Get confidence as a float

                        # Prepare the detection in the format (x_min, y_min, x_max, y_max, confidence)
                        detections.append(([x1, y1, x2, y2], confidence, 'person'))

                        # Draw bounding box and label from YOLO detections only
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(resized_frame, f'Person {confidence:.2f}', (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

                # Update the tracker with the current frame detections
                tracks = tracker.update_tracks(detections, frame=resized_frame)

                # Track the active IDs in the current frame
                current_frame_ids = set()

                # Update person count based on track IDs
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    current_frame_ids.add(track_id)  # Add current track ID

                    # If a new person ID appears, increment the count
                    if track_id not in active_person_ids:
                        total_entered_count += 1  # Increment the count for a new person

                # Update active_person_ids by removing IDs no longer in the frame
                for person_id in active_person_ids - current_frame_ids:
                    active_person_ids.discard(person_id)  # Person has left the frame

                # Update the active_person_ids with the IDs from the current frame
                active_person_ids.update(current_frame_ids)

                # Display the count of people in the frame and total entered count
                cv2.putText(resized_frame, f'Currently in frame: {len(current_frame_ids)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(resized_frame, f'Total Entered: {total_entered_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Encode the frame as JPEG and send it to WebSocket
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            msg={
                "frame":frame_data,
                "total_entered":total_entered_count,
                "current_in_frame":len(current_frame_ids)
            }
            await websocket.send_json(msg)
            
            
            await asyncio.sleep(0.05)  # Adjust the frame rate if needed
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.post("/send_filename")
async def set_ip_port(request: Request):
    global cap
    global a
    data = await request.json()
    name = data['filename']
    # file_path="./sample"
    file_path = f'./uploads/{name}'  # Assuming file is in 'sample_images' folder
    a=file_path
    image_formats = r'\.(jpg|jpeg|png|gif|bmp|tiff|svg|webp)$'
    video_formats = r'\.(mp4|mov|avi|mkv|flv|wmv|webm|m4v)$'
    # Open the video file with OpenCV
    print(name)
    if re.search(image_formats, name, re.IGNORECASE):
        print("image file found")
        cap = cv2.imread(file_path)
    else:
        print("video file found")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Failed to connect to the video file"}
    
    return {"status": "success"}

    
@app.websocket("/stream_video_file")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    total_entered_count = 0
    active_person_ids = set()

    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from video.")
                break

            # Process the frame with YOLO model (Assuming model is loaded)
            resized_frame = cv2.resize(frame, (1920, 1080))
            results = model.predict(source=resized_frame, device=device, conf=0.20, show=False)

            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    if class_id == 0:  # 'person' class ID
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Draw bounding boxes
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(resized_frame, f'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                        
                        # Prepare detections for tracking
                        detections.append(([x1, y1, x2, y2], box.conf.item(), 'person'))

            # Track detections
            tracks = tracker.update_tracks(detections, frame=resized_frame)
            current_frame_ids = set()
            
            # Track the active IDs
            for track in tracks:
                if track.is_confirmed():
                    current_frame_ids.add(track.track_id)
                    if track.track_id not in active_person_ids:
                        total_entered_count += 1
            
            active_person_ids.update(current_frame_ids)

            # Add overlay with people count
            cv2.putText(resized_frame, f'Currently in frame: {len(current_frame_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(resized_frame, f'Total Entered: {total_entered_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Convert frame to base64 and send over WebSocket
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            msg = {
                "frame": frame_data,
                "total_entered": total_entered_count,
                "current_in_frame": len(current_frame_ids)
            }
            await websocket.send_json(msg)

            await asyncio.sleep(0.05)  # Adjust the frame rate

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


@app.websocket("/stream_image_file")
async def stream_image(websocket: WebSocket):
    await websocket.accept()

    total_entered_count = 0  # Counter for people entering the frame
    bbox_scale_factor = 0.9  # Adjust bounding box size if needed

    try:
        # Capture the frame
        frame = cap

        # Resize and process the frame with YOLO model
        resized_frame = cv2.resize(frame, (640, 480))
        results = model.predict(source=resized_frame, device=device, conf=0.20, show=False)

        person_count = 0  # Count of people in the current frame

        # Iterate over detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 0:  # Check for 'person' class ID
                    person_count += 1

                    # Extract bounding box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Scale the bounding box
                    width, height = x2 - x1, y2 - y1
                    x1 = int(x1 + width * (1 - bbox_scale_factor) / 2)
                    y1 = int(y1 + height * (1 - bbox_scale_factor) / 2)
                    x2 = int(x2 - width * (1 - bbox_scale_factor) / 2)
                    y2 = int(y2 - height * (1 - bbox_scale_factor) / 2)

                    # Draw bounding box and label on the frame
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized_frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        # Update total count based on detected persons
        total_entered_count += person_count

        # Overlay person count information on the frame
        cv2.putText(resized_frame, f'Currently in frame: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(resized_frame, f'Total Entered: {total_entered_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode the frame to JPEG and convert to base64 for WebSocket transmission
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        # Send the frame and count data over WebSocket
        msg = {
            "frame": frame_data,
            "total_entered": total_entered_count,
            "current_in_frame": person_count
        }
        await websocket.send_json(msg)

        await asyncio.sleep(0.05)  # Adjust frame rate

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


@app.websocket("/stream_high_density_image_file")
async def stream_image(websocket: WebSocket):
    await websocket.accept()
    model_csr = CSRNet().to(device)

# Load the model state dictionary on CPU and set weights_only=True for safety
    model_csr.load_state_dict(torch.load('./csrnet_model_final.pth', map_location=torch.device('cpu'),weights_only=True))

    # Set the model to evaluation mode
    model_csr.eval()
    def count_objects_and_show_heatmap(model, image, device, downsample_factor=8):
    
        
        # Preprocess the image (convert to RGB, resize, normalize)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image_tensor = transform(image_rgb).unsqueeze(0).to(device)  # Apply transform and add batch dimension
        
        with torch.no_grad():  # Disable gradient calculation during inference
            output = model(image_tensor)  # Get the output density map from the model
        
        # Convert the output to a numpy array (remove batch dimension)
        density_map = output.squeeze().cpu().numpy()

        # Rescale the density map back to the original image size (if downsampling was applied)
        original_height, original_width = image.shape[:2]
        density_map_rescaled = cv2.resize(density_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        # Total count is the sum of all values in the density map
        total_count = density_map_rescaled.sum()

        # Visualization: Plotting the heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(density_map_rescaled, cmap='jet', interpolation='bilinear')  # Use 'jet' colormap for better heatmap visualization
        plt.colorbar()  # Show color bar
        plt.title(f"Heatmap - Total Count: {total_count:.2f}")
        # plt.axis('off')  # Hide axes for better display
        # plt.show()
        heatmap_path = "./uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()

        # Convert buffer to PIL image
        # heatmap_image = Image.open(buf)
        # buf.close()  # Close the buffer

        return total_count,heatmap_path


    try:
        # Capture the frame
        frame = cap
        # Resize and process the frame with YOLO model
        image=frame
        a = cv2.resize(frame, (640, 480))
        total_entered_count,resized_frame = count_objects_and_show_heatmap(model_csr, image, device)
        # resized = cv2.resize(resized_frame, (1920, 1080))
        
        # Encode the frame to JPEG and convert to base64 for WebSocket transmission
        with open(resized_frame, "rb") as f:
                heatmap_encoded = base64.b64encode(f.read()).decode('utf-8')

        # total_entered_count = count_objects_and_show_heatmap(model_csr, image, device)
        # Send the frame and count data over WebSocket
        _, buffer = cv2.imencode('.jpg',a)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        msg = {
            "frame1": heatmap_encoded,
            "frame2": frame_data,
            "total_entered": int(total_entered_count),
            "current_in_frame": 9
        }
        await websocket.send_json(msg)
        os.remove(resized_frame)
        await asyncio.sleep(0.05)  # Adjust frame rate

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()

@app.websocket('/attr_image_file')
async def stream_image(websocket: WebSocket):
    try:
        # Accept the WebSocket connection
        await websocket.accept()

        print("WebSocket connection accepted.")
        frame = cap

        # Resize and process the frame with YOLO model
        resized_frame = cv2.resize(frame, (640, 480))
        # Replace `file` with the actual file handling logic
        try:
            # Initialize the Groq client with the API key
            client = Groq(api_key="PUT_YOUR_API_KEY")
            
            # Path to your image
            image_path=file
            print(image_path)
            # Encode the image
            base64_image = encode_image(a)        
            # Define the prompt
            prompt = """
            Extract the following from the given image:
            1. Provide a detailed description of all text extracted from the image in a human-readable paragraph format.
            2. Categorize the extracted information into the following attributes:
            - Color
            - Size/Dimensions
            - Material
            - Weight
            - Price
            - Brand
            - Features
            - Specifications
            - Ideal For
            - Designed For
            
            If any additional attributes not covered in the above list are found in the text (e.g., Warranty, Compatibility, or Performance), include them in the results as a new category.
            
            Output Format:
            Description: (Provide a complete paragraph with only the extracted details)
            
            Extracted Attributes:
            Color: (value)
            Size/Dimensions: (value)
            Material: (value)
            Weight: (value)
            Price: (value)
            Brand: (value)
            Features: (value)
            Specifications: (value)
            Ideal For: (value)
            Designed For: (value)
            Additional Categories (if applicable): (value)
            
            Do not add unnecessary details or comments. Follow the specified format strictly.
            """
            
            try:
                # Create the chat completion request
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    model="llama-3.2-11b-vision-preview",
                )
                
                # Extract and process the response
                response = chat_completion.choices[0].message.content
                print(response)
                
                # Convert response to dictionary
                attributes_dict = extract_and_clean_attributes(response)
                print(attributes_dict)
                
                # Save the dictionary as JSON
                # save_dict_as_json(attributes_dict, 'extracted_attributes.json')

            except Exception as api_error:
                print(f"An error occurred while fetching chat completion: {api_error}")

        except Exception as init_error:

            print(f"An error occurred during initialization: {init_error}")
            
        # attributes_dict={'****Extracted Attributes': '**', '* Color': 'Midnight', '* Size/Dimensions': 'Not available', '* Material': 'Plastics, Rubber, Titanium', '* Weight': 'Not available', '* Price': 'Rs. 1,299', '* Brand': 'Oppo', '* Features': 'Bluetooth-enabled version 5.2, 28 hrs battery Life, 46 battery mAO mAO battery capacity, 28 hrs play time, sweat proof, deep bass, water resistant, monaural deep noise cancellation', '* Specifications': 'Bluetooth version 5.2, 28 hrs battery Life, 46 battery mAO mAO battery capacity, 28 hrs play time, sweat proof, deep bass water resistant, monaural deep noise cancellation', '* Ideal For': 'Men and Women', '* Designed For': 'Smartphones, Laptops, Smart TVs, Tablets'}
        cleaned_dict = {
    key.lstrip('*').strip(): value.lstrip('*').rstrip('*').strip() for key, value in attributes_dict.items()
}
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        msg = {
            "frame": frame_data,
            "total_entered": cleaned_dict
        }
        # Send JSON message to the client
        await websocket.send_json(msg)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        await websocket.close()

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.2", port=8001)
