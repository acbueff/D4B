"""
Computer Vision Fundamentals for Business Applications

This script provides a comprehensive introduction to computer vision concepts
with a focus on business applications. It includes theoretical explanations,
practical implementations, and real-world examples.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def demonstrate_basic_image_processing():
    """
    Demonstrates basic image processing concepts using a simple business example:
    analyzing product packaging for quality control.
    """
    # Create a sample product image (simulated)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    # Draw a product package
    cv2.rectangle(image, (50, 50), (250, 150), (0, 255, 0), -1)  # Green package
    cv2.rectangle(image, (100, 70), (200, 130), (255, 255, 255), -1)  # White label
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Product Image')
    
    plt.subplot(132)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(133)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    
    plt.tight_layout()
    plt.show()
    
    print("\nBasic Image Processing Concepts:")
    print("1. Color to Grayscale: Simplifies image analysis")
    print("2. Edge Detection: Identifies object boundaries")
    print("3. Image Enhancement: Improves feature visibility")

# Reflective Questions for Basic Image Processing
"""
Reflective Questions:
1. How might basic image processing help in quality control for your business?
2. What types of product defects could be detected using these techniques?
3. How could edge detection be useful in inventory management?
"""

def demonstrate_object_detection():
    """
    Demonstrates object detection using a pre-trained model.
    Business application: Retail inventory tracking.
    """
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # Create a sample retail shelf image (simulated)
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    # Draw some products
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue product
    cv2.rectangle(image, (200, 50), (300, 150), (0, 0, 255), -1)  # Red product
    cv2.rectangle(image, (125, 200), (225, 300), (255, 255, 0), -1)  # Yellow product
    
    # Convert to PIL Image
    image_pil = Image.fromarray(image)
    
    # Transform image for model
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil)
    
    # Make prediction
    with torch.no_grad():
        prediction = model([image_tensor])
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title('Retail Shelf with Detected Products')
    plt.show()
    
    print("\nObject Detection in Retail:")
    print("1. Product Recognition: Identifies items on shelves")
    print("2. Inventory Tracking: Monitors stock levels")
    print("3. Shelf Analysis: Optimizes product placement")

# Reflective Questions for Object Detection
"""
Reflective Questions:
1. How could object detection improve your retail operations?
2. What privacy considerations should be addressed when using surveillance cameras?
3. How might this technology impact employee roles in retail?
"""

def demonstrate_image_segmentation():
    """
    Demonstrates image segmentation using a simple thresholding approach.
    Business application: Agricultural field analysis.
    """
    # Create a sample agricultural field image (simulated)
    image = np.zeros((200, 300), dtype=np.uint8)
    # Draw some crops and soil
    cv2.circle(image, (100, 100), 30, 255, -1)  # Crop 1
    cv2.circle(image, (200, 100), 30, 255, -1)  # Crop 2
    cv2.circle(image, (150, 150), 30, 255, -1)  # Crop 3
    
    # Apply thresholding
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Field Image')
    
    plt.subplot(132)
    plt.imshow(binary, cmap='gray')
    plt.title('Segmented Image')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
    plt.title('Segmented Crops')
    
    plt.tight_layout()
    plt.show()
    
    print("\nImage Segmentation Applications:")
    print("1. Crop Detection: Identifies healthy vs. diseased plants")
    print("2. Field Analysis: Measures crop coverage")
    print("3. Yield Estimation: Predicts harvest quantities")

# Reflective Questions for Image Segmentation
"""
Reflective Questions:
1. How could image segmentation benefit agricultural businesses?
2. What other industries could benefit from this technology?
3. How might this improve resource allocation in farming?
"""

def demonstrate_face_recognition():
    """
    Demonstrates face detection using a simple approach.
    Business application: Customer analytics in retail.
    """
    # Create a sample image with faces (simulated)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    # Draw simple face representations
    cv2.circle(image, (100, 100), 30, (255, 255, 255), -1)  # Face 1
    cv2.circle(image, (200, 100), 30, (255, 255, 255), -1)  # Face 2
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simulate face detection by finding bright regions
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around detected faces
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(133)
    plt.imshow(binary, cmap='gray')
    plt.title('Detected Faces')
    
    plt.tight_layout()
    plt.show()
    
    print("\nFace Recognition Applications:")
    print("1. Customer Analytics: Tracks demographics and behavior")
    print("2. Security: Identifies potential threats")
    print("3. Personalization: Enhances customer experience")

# Reflective Questions for Face Recognition
"""
Reflective Questions:
1. What ethical considerations arise when using face recognition in retail?
2. How could this technology improve customer service?
3. What privacy protections should be implemented?
"""

def business_challenge():
    """
    Final Business Challenge: Retail Analytics System
    
    Scenario: A retail store wants to implement a computer vision system
    to analyze customer behavior and optimize store layout.
    
    Your task is to implement a simple version of this system that can:
    1. Detect when customers enter the store
    2. Track basic movement patterns
    3. Identify popular areas
    """
    # Create a simulated store layout
    store = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw store elements
    cv2.rectangle(store, (50, 50), (550, 350), (255, 255, 255), -1)  # Store floor
    cv2.rectangle(store, (100, 100), (200, 200), (0, 255, 0), -1)    # Product area 1
    cv2.rectangle(store, (300, 100), (400, 200), (0, 255, 0), -1)    # Product area 2
    cv2.rectangle(store, (200, 250), (300, 350), (0, 255, 0), -1)    # Product area 3
    
    # Draw entrance
    cv2.rectangle(store, (250, 30), (350, 50), (255, 0, 0), -1)
    
    # Simulate customer movement
    customer_pos = np.array([300, 40])  # Start at entrance
    movement = np.array([0, 5])  # Move down
    
    # Track customer position
    for _ in range(60):
        customer_pos += movement
        if customer_pos[1] > 350:  # Reset at bottom
            customer_pos = np.array([300, 40])
        
        # Draw customer
        store_copy = store.copy()
        cv2.circle(store_copy, tuple(customer_pos.astype(int)), 10, (0, 0, 255), -1)
        
        # Show frame
        plt.clf()
        plt.imshow(cv2.cvtColor(store_copy, cv2.COLOR_BGR2RGB))
        plt.title('Customer Movement Tracking')
        plt.pause(0.1)
    
    print("\nBusiness Challenge Implementation:")
    print("1. Customer Detection: Tracks entry points")
    print("2. Movement Analysis: Records walking patterns")
    print("3. Area Popularity: Identifies frequented sections")

# Challenge Questions
"""
Challenge Questions:
1. How could you enhance this system to track multiple customers?
2. What additional metrics would be valuable for store optimization?
3. How would you handle privacy concerns in a real implementation?
"""

if __name__ == "__main__":
    print("Welcome to Computer Vision for Business Applications!")
    print("\nThis tutorial will demonstrate key computer vision concepts")
    print("and their applications in business contexts.")
    
    # Run demonstrations
    demonstrate_basic_image_processing()
    demonstrate_object_detection()
    demonstrate_image_segmentation()
    demonstrate_face_recognition()
    
    # Run business challenge
    print("\nNow, let's work on the business challenge...")
    business_challenge()
    
    print("\nThank you for completing the Computer Vision tutorial!")
    print("Remember to consider both technical capabilities and business implications")
    print("when implementing computer vision solutions in your organization.") 