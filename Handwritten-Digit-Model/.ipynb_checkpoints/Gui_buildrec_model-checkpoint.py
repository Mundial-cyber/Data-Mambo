#Build an interactive window where digits will be written and we can recognize the digits with a button. To do this, we will use the Tkinter library in python

#Import Libraries
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import cv2
import pytesseract

#Load Model
model = load_model("mundial_mnist.h5")

#Predict digit function
def predict_digit(img):
	#Resize pixels to 28*28
	img = img.resize((28,28))
	#Convert rgb to grayscale
	img = img.convert("L")
	#Invert colors(MNIST colors are black on white background)
	img = 255 - np.array(img, dtype=np.uint8)
	#Normalize pixel values
	img = img/255.0
	#Reshaping to support the model input.
	img = img.reshape(1,28,28,1)
	#Predict the class
	res = model.predict(img)[0]
	#Return
	return np.argmax(res), max(res)

#Centering and scaling drawn digits.
def preprocess_image(img):
	#Convert image to numpy array and grayscale
	img = np.array(img.convert("L"))
	#Threshold the image to remove noise and isolate the digit.
	_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
	#Find contours of the digit
	contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#If no contours are found, return blank image
	if len(contours) == 0:
		return np.zeros((28, 28), dtype=np.float32)
	#Get bounding box of the largest contour
	x, y, w, h = cv2.boundingRect(contours[0])
	#Crop image around the digit
	img = img[y:y+h, x:x+w]
	#Resize to 20*20 to keep aspect ratio then add padding(4,4) to make it 28*28 pixels
	img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
	padded_img = np.pad(img, ((4,4), (4,4)), mode="constant", constant_values=0)
	#Normalize pixel values
	padded_img = padded_img / 255.0
	#Update prediction
	img = preprocess_image(img)
	res = model.predict(img)[0]
	return padded_img.reshape(1, 28, 28, 1)

class App(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		self.x = self.y = 0
		
		#Creating Elements
		self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
		self.label = tk.Label(self, text="Draw...", font=("Helvetica", 48))
		self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
		self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

		#Grid structure
		self.canvas.grid(row=0, column=0, pady=2, sticky=W)
		self.label.grid(row=0, column=1, pady=2, padx=2)
		self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
		self.button_clear.grid(row=1, column=0, pady=2)
		#self.canvas.bind(<B1-Motion), self.start_pos. Allows drawing on the canvas when the mouse is moved.
		self.canvas.bind("<B1-Motion>", self.draw_lines)

	def clear_all(self):
		self.canvas.delete("all")

	def classify_handwriting(self):
		HWND = self.canvas.winfo_id()
		rect = win32gui.GetWindowRect(HWND)
		a, b, c, d = rect
		rect = (a+4, b+4, c-4, d-4)
		try:
			im = ImageGrab.grab(rect)
			im = im.convert("L")
			im = im.resize((28, 28))
		except Exception as e:
			print("Error capturing the canvas:", e)
			return
		#Convert image and preprocess
		im = im.convert("L")
		img = np.array(im)
		_, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV +  cv2.THRESH_OTSU)
		#Detect contours
		contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		regions = []
		batch_regions = []
		digit_indices = []
		for idx, contour in enumerate(contours):
			x, y, w, h = cv2.boundingRect(contour)
			#Filter out noise
			if h>15 and w>10:
				#Ensure Valid dimension for aspect ratio
				if h>0 and w>0:
					aspect_ratio = w / h
				region = binary_img[y:y+h, x:x+w]
				aspect_ratio = w / h		
				if aspect_ratio < 1.5: #Likely a digit
					#Resize and normalize
					region = cv2.resize(region, (28, 28), interpolation = cv2.INTER_AREA)
					region = region.astype("float32") / 255.0
					if region.shape == (28, 28):
						
						region = region.reshape(28, 28, 1)
						batch_regions.append(region)
						digit_indices.append(len(regions))
						#Placeholder for prediction result
						regions.append(None)
					else:
						print(f"Skipping invalid region at index {idx} with shape {region.shape}")
				else:
					#Mark as text for OCR processing
					regions.append("TEXT", (x, y, w, h))
		#Batch predict digits for speed
		if batch_regions:
			try:
				
				batch_regions = np.array(batch_regions)
				predictions = model.predict(batch_regions, verbose=0)

				#Assign predictions to corresponding regions
				for i, prediction in enumerate (predictions):
					
					digit = np.argmax(prediction)
					#Confidence as %
					confidence = max(prediction) * 100
					regions[digit_indices[i]] = f"{digit} ({confidence:.2f}%)"
			except ValueError as e:
				print(f"Error during batch prediction: {e}")

		#Process text regions in Tesseract
		for idx, region in enumerate(regions):
			
			if isinstance(region, tuple) and region[0] == "TEXT":
				x, y, w, h = region[1]
				text_region = binary_img[y:y+h, x:x+w]
				text_region_pil = Image.fromarray(text_region)
				text = pytesseract.image_to_string(region_pil, config = "--psm 6").strip()
				regions[idx] = text if text else "?"

		#Prepare final results for display
		final_results = [str(r) if isinstance(r, str) else "[Unrecognized]" for r in regions]
			
				
		
		#Display combined result
		self.label.configure(text=" ".join(final_results))
				
		
	
	def draw_lines(self, event):
		#Method draws lines once the mouse is moved.
		self.x = event.x
		self.y = event.y
		r = 8
		self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill="black")
app = App()
mainloop()
"""
How it works.
- x and y represent coordinates of mouse pointers on canvas when drawing.
- self.x, self.y is initialized at 0 and gets updated when the mouse moves on the canvas.
- event x and y are the current positions of the mouse pointer on the canvas. The activity takes place by holding down the left mouse button <B1-Motion
- These values are used to create a small circle (oval) on the canvas
- Radius(r) is 8 and the circle is centered at (x,y)
- So when the user drags the left mouse button to draw <B1-Motion>, lines are drawn to form the desired number.
- The draw function is called repeatedly and hence, the self.x and self.y are updated when events x and y occur.
"""
		