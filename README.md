# CODE IQ - ICFAI TECH HACKATHON CHALLENGE
## SOLUTION BY TEAM CODE BLENDERS



## Table of contents

- [Overview](#overview)
- [Architecture Diagram](#my-process)
- [Built with](#built-with)
- [Installation](#installation)
- [Project structure](#structure)
- [Result Analysis](#resultanalysis)
- [Feature](#features)
- [Author](#author)


## Overview
- This is a solution to the [CODE IQ - ICFAI tech Hackathon Challenge (Track 1-2) ](https://unstop.com/hackathons/ml-core-icfai-tripura-1203996). 

- **Track 1: Crowd Counting** - Our model is designed to accurately estimate crowd sizes in images and videos, even under challenging conditions like varying densities, lighting, and diverse environments. We trained a deep learning model that detects and counts individuals efficiently by learning spatial and contextual cues within each frame.

- This project aims to develop **accurate and adaptable crowd counting system** for **real-time monitoring and analysis**, addressing the need for reliable crowd size estimation in various scenarios. Our solution integrates a **Custom YOLOv11 model** for object detection and tracking with a **Custom CSRNet model** for density-based estimation. This dual-model approach ensures precise crowd size predictions across both **low-resolution sparse crowds and high-resolution dense crowds**. Hosted on a **FastAPI server**, the system supports **real-time video input via IP Webcam** and **batch processing of images and videos in multiple formats**, offering **high accuracy** and **adaptability** for diverse scenarios. It is designed to ensure **high usability** and **scalability** for applications in **public safety, event management, and urban planning**.

- **Track 2: Product Attribute Extraction** - This track challenges participants to extract meaningful attributes from unstructured web data using NLP and data structuring, organizing product descriptions into usable formats.

- This project provides an **advanced solution** for **automated attribute extraction** from **unstructured product descriptions**, enhancing **e-commerce platforms'** search and recommendation capabilities. By integrating the **Llama 3.2 text model** for processing **textual data** and the **Llama 3.2 vision model** for analyzing **product images**, the system ensures **accurate** and **comprehensive extraction** of attributes such as **color, size, material, weight**, and **price**. It incorporates **preprocessing techniques** like **tokenization, stemming**, and **lemmatization**, along with **unit standardization** and **value normalization**, to handle **large datasets** effectively. Hosted on a **FastAPI server**, the solution supports **real-time** and **batch processing** for diverse product categories, aligning **textual** and **visual data** to provide **structured outputs**. Designed for **scalability** and **efficiency**, it empowers **e-commerce platforms** to optimize **product discoverability**, improve **search accuracy**, and deliver **personalized recommendations**.


## Architecture Diagram
 ### Crowd Counting

<img src="./assets/img/architecture.png">

 ### Product Attribute Extraction

<img src="./assets/img/architecture_track2.png">

## Built with

- ### Frontend:
  - HTML, CSS, JS

- ### Backend:
  - FastAPI
  - Python
  - YOLOv11
  - Deepsort tracking algorithm
  - CSRNet
  - Pillow
  - spaCy/NLTK

- ### Libraries
  - `Ultralytics`
  - `Opencv-python`
  - `numpy`, `pandas` for data handling
  - `uvicorn` for FastAPI
  - `PyTorch` For loading, fine-tuning, and deploying the Llama
3.2 models (text and vision)


## Installation

### Prerequirements
  - `python3.11`

### Installation steps

  ```
    git clone https://github.com/Sabari2005/Code_iq
    cd Code_iq
  ```
  ```
  pip install -r requirements.txt
  ```

  - Execute each commands in a seperate terminal
  ```
  python index.py
  python serve.py

  ```
  - Open ` http://127.0.0.1:8000` in your browser

## Project structure

```
├──          
├── static
│   ├── css                    
│   └── images                 
├── templates
│   └── index.html             
├── index.py   
├── serve.py                           
├── requirements.txt           
└── README.md                  
```
## Result Analysis

- ### Crowd Counting 
    ![](assets/img/f1.jpg) 
    ![](assets/img/p_curve.jpg)

- ### CustomCSRNet
    <img src="./assets/img/custom CSRNet.png">


## Sample model Output
- ### People counting
  ![](assets/img/image42.jpg) 
    <img src="./assets/img/0_Q7dF5t_bAdUW-z1h.webp" >
    <img src="./assets/img/download.jpg">

 - ### Product Attribute Extraction

## Website Overview
<img src="./assets/img/web1.png">
<img src="./assets/img/web2.png">
<img src="./assets/img/web3.png">
<img src="./assets/img/web4.png">


## Features
- ### Our website have a real-time corwd prediction streaming

  - Install   ```IP webcam ``` in your android phone and click ```start server ```
  - Now enter the  ``` ipaddress``` and ``` port number ``` in the website 

<img src="./assets/img/web5.png">


## Demo 

- Click [here](./assets/videos/demo.mp4) to see the demo video


## Author

- Sabari Vadivelan S (Team Leader) - Contact Gmail [sabari132005@gmail.com]()
- Kamal M (Member 1)
- Uvarajan D (Member 2)
- Kaviarasu K (Member 3)
