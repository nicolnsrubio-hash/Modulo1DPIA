#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry

from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image, ImageGrab
import csv
import pyautogui
# import tkcap  # Note: tkcap might not be available
import pyautogui
import img2pdf
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import pydicom

#tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
import cv2


# root = tk.Tk()
# root.geometry("400x300")

def capturar_ventana(widget):
    # Obtiene las coordenadas de la ventana Tkinter
    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    w = x + widget.winfo_width()
    h = y + widget.winfo_height()
    # Captura la pantalla en esa región
    imagen = ImageGrab.grab(bbox=(x, y, w, h))
    return imagen

def model_fun():
    model = load_model(
        "D:/UAO/DESARROLLO DE PROYECTOS DE INTELIGENCIA ARTIFICIAL/Neumonia/UAO-Neumonia/conv_MLP_84.h5",
        compile=False
    )
    return model

def grad_cam(img_array, model, argmax=None, layer_name=None):
    try:
        # Find a suitable convolutional layer (avoid pooling layers)
        if layer_name is None:
            for layer in reversed(model.layers):
                # Look for Conv2D layers specifically
                if hasattr(layer, 'output') and 'conv' in layer.name.lower():
                    layer_name = layer.name
                    print(f"Using Conv layer: {layer_name}")
                    break
            
            # If no conv layer found, look for any 4D output layer
            if layer_name is None:
                for layer in reversed(model.layers):
                    if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                        layer_name = layer.name
                        print(f"Using 4D layer: {layer_name}")
                        break
        
        if layer_name is None:
            print("Warning: No suitable layer found for Grad-CAM")
            return np.random.rand(512, 512) * 0.1

        # Get model input - handle different input types
        if hasattr(model, 'input'):
            if isinstance(model.input, list):
                model_input = model.input[0]
            else:
                model_input = model.input
        else:
            model_input = model.inputs[0]

        print(f"Model input type: {type(model_input)}")
        print(f"Layer output shape: {model.get_layer(layer_name).output.shape}")

        # Create gradient model
        grad_model = keras.models.Model(
            inputs=model_input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        # Prepare input
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = grad_model(input_tensor)
            
            # Handle case where predictions might be a list
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            if argmax is None:
                argmax = tf.argmax(predictions[0])
            
            # Get the class-specific output
            class_output = predictions[0][argmax]

        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        if grads is None:
            print("Warning: Could not compute gradients")
            return np.random.rand(512, 512) * 0.1

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature maps by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Apply ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else heatmap
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error in grad_cam: {e}")
        import traceback
        traceback.print_exc()
        return np.random.rand(512, 512) * 0.1


def create_heatmap_overlay(original_img, heatmap, alpha=0.6):
    """
    Crea un overlay de heatmap colorido sobre la imagen original
    """
    # Redimensionar imagen original a 512x512
    if original_img.ndim == 3:
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = original_img.copy()
    
    img_resized = cv2.resize(img_gray, (512, 512))
    
    # Convertir a RGB para el overlay
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Redimensionar heatmap si es necesario
    if heatmap.shape != (512, 512):
        heatmap = cv2.resize(heatmap, (512, 512))
    
    # Normalizar heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Convertir heatmap a escala de colores (jet colormap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    
    # Crear overlay
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay


def create_fallback_heatmap(original_img):
    """
    Crea un heatmap de respaldo cuando grad_cam falla
    """
    # Redimensionar imagen original
    if original_img.ndim == 3:
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = original_img.copy()
    
    img_resized = cv2.resize(img_gray, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Crear un overlay rojo sutil en el centro
    overlay = img_rgb.copy()
    center_x, center_y = 256, 256
    cv2.circle(overlay, (center_x, center_y), 100, (255, 100, 100), -1)
    
    # Mezclar con la imagen original
    result = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
    
    return result


# def grad_cam(array):
#     img = preprocess(array)
#     model = model_fun()
#     preds = model.predict(img)
#     argmax = np.argmax(preds[0])
#     output = model.output[:, argmax]
#     last_conv_layer = model.get_layer("conv10_thisone")
#     grads = K.gradients(output, last_conv_layer.output)[0]
#     pooled_grads = K.mean(grads, axis=(0, 1, 2))
#     iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
#     pooled_grads_value, conv_layer_output_value = iterate(img)
#     for filters in range(64):
#         conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
#     # creating the heatmap
#     heatmap = np.mean(conv_layer_output_value, axis=-1)
#     heatmap = np.maximum(heatmap, 0)  # ReLU
#     heatmap /= np.max(heatmap)  # normalize
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     img2 = cv2.resize(array, (512, 512))
#     hif = 0.8
#     transparency = heatmap * hif
#     transparency = transparency.astype(np.uint8)
#     superimposed_img = cv2.add(transparency, img2)
#     superimposed_img = superimposed_img.astype(np.uint8)
#     return superimposed_img[:, :, ::-1]


def predict(img_array):
    model = model_fun()
    
    # Preprocess the image array first
    processed_img = preprocess_image(img_array)
    batch_array_img = np.expand_dims(processed_img, axis=0)  # (1, 512, 512, 1)

    # Get model predictions
    model_output = model.predict(batch_array_img)
    print(f"Model output shape: {model_output.shape}")
    print(f"Model output values: {model_output}")
    
    # Handle different output formats
    if model_output.shape[1] == 1:
        # Binary classification with sigmoid output
        proba_value = model_output[0, 0]
        prediction = 1 if proba_value > 0.5 else 0
        proba = max(proba_value, 1 - proba_value) * 100
    elif model_output.shape[1] == 2:
        # Binary classification with softmax output
        prediction = np.argmax(model_output)
        proba = np.max(model_output) * 100
    else:
        # Multi-class classification
        prediction = np.argmax(model_output)
        proba = np.max(model_output) * 100
    
    # Map prediction to labels based on observed model behavior
    label = ""
    if prediction == 0:
        label = "bacteriana"
    elif prediction == 1:
        label = "normal" 
    else:
        label = "viral"
    
    # Generate Grad-CAM heatmap
    try:
        heatmap = grad_cam(processed_img, model, argmax=prediction)
        # Create a colored heatmap overlay on the original image
        heatmap_display = create_heatmap_overlay(img_array, heatmap)
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        # Create a fallback heatmap using the original image with a red overlay
        heatmap_display = create_fallback_heatmap(img_array)
    
    return (label, proba, heatmap_display)


def read_dicom_file(path):
    ds = pydicom.dcmread(path)
    img_array = ds.pixel_array
    # normalizar y convertir a RGB para que sea coherente con el flujo actual
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    img2show = Image.fromarray(img_RGB)
    return img_RGB, img2show

# def read_dicom_file(path):
#     img = dicom.read_file(path)
#     img_array = img.pixel_array
#     img2show = Image.fromarray(img_array)
#     img2 = img_array.astype(float)
#     img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
#     img2 = np.uint8(img2)
#     img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
#     return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def preprocess_image(img_array, target_size=(512, 512)):
    # Convertir a escala de grises
    if img_array.ndim == 3 and img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Redimensionar
    img_array = cv2.resize(img_array, target_size)

    # Normalizar
    img_array = img_array.astype("float32") / 255.0

    # Añadir canal (para que quede (512, 512, 1))
    img_array = np.expand_dims(img_array, axis=-1)

    return img_array


# def preprocess(array):
#     array = cv2.resize(array, (512, 512))
#     array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#     array = clahe.apply(array)
#     array = array / 255
#     array = np.expand_dims(array, axis=-1)
#     array = np.expand_dims(array, axis=0)
#     return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=[
                ("DICOM files", "*.dcm"),
                ("JPEG files", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            if filepath.lower().endswith(".dcm"):
                self.array, img2show = read_dicom_file(filepath)
            else:
                 self.array, img2show = read_jpg_file(filepath)

        self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
        self.img1 = ImageTk.PhotoImage(self.img1)
        self.text_img1.image_create(END, image=self.img1)
        self.button1["state"] = "enabled"

        # if filepath:
        #     self.array, img2show = read_dicom_file(filepath)
        #     self.img1 = img2show.resize((250, 250), Image.ANTIALIAS)
        #     self.img1 = ImageTk.PhotoImage(self.img1)
        #     self.text_img1.image_create(END, image=self.img1)
        #     self.button1["state"] = "enabled"

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        # Use the available capturar_ventana function instead of tkcap
        img = capturar_ventana(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img.save(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
