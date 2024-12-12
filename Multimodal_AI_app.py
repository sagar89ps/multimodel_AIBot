import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import soundfile as sf

# Image Generation
from diffusers import StableDiffusionPipeline

# Text Generation
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)

# Speech Synthesis
from transformers import AutoModelForTextToSpectrogram, AutoProcessor

# Video Processing
import cv2

class MultimodalAIApp:
    def __init__(self):
        # Initialize models
        self.text_model = None
        self.image_model = None
        self.audio_model = None
        
        # Set up Streamlit page configuration
        st.set_page_config(
            page_title="Multimodal AI Assistant",
            page_icon="🤖",
            layout="wide"
        )
        
    def load_models(self):
        # Load Text Generation Model
        try:
            self.text_model = pipeline(
                'text-generation', 
                model='gpt2-medium'
            )
        except Exception as e:
            st.error(f"Error loading text model: {e}")
        
        # Load Image Generation Model
        try:
            self.image_model = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0"
            )
        except Exception as e:
            st.error(f"Error loading image model: {e}")
        
        # Load Audio Synthesis Model
        try:
            self.audio_model = pipeline(
                "text-to-speech", 
                model="microsoft/speecht5_tts"
            )
        except Exception as e:
            st.error(f"Error loading audio model: {e}")
        
    def process_multimodal_input(self):
        st.title("🌈 Multimodal AI Assistant")
        
        # Sidebar for Model Selection
        st.sidebar.header("Input Configuration")
        output_type = st.sidebar.selectbox(
            "Select Output Type", 
            ["Text", "Image", "Audio"]
        )
        
        # Input Section
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Input")
            # Text Input
            text_input = st.text_area("Enter Text Prompt")
            
            # Image Upload
            uploaded_image = st.file_uploader(
                "Upload Image", 
                type=['png', 'jpg', 'jpeg']
            )
            
            # Video Upload
            uploaded_video = st.file_uploader(
                "Upload Video", 
                type=['mp4', 'avi', 'mov']
            )
        
        with col2:
            st.header("Output")
            # Process and Generate Output
            if st.button("Generate Output"):
                if not text_input and not uploaded_image and not uploaded_video:
                    st.warning("Please provide some input!")
                    return
                
                # Text Generation
                if output_type == "Text":
                    self.generate_text(text_input)
                
                # Image Generation
                elif output_type == "Image":
                    self.generate_image(text_input, uploaded_image)
                
                # Audio Generation
                elif output_type == "Audio":
                    self.generate_audio(text_input)
        
    def generate_text(self, text_input):
        if not text_input:
            st.warning("Please enter a text prompt")
            return
        
        try:
            response = self.text_model(
                text_input, 
                max_length=200, 
                num_return_sequences=1
            )[0]['generated_text']
            
            st.text_area("Generated Text", value=response, height=200)
        except Exception as e:
            st.error(f"Text generation failed: {e}")
    
    def generate_image(self, text_input, uploaded_image=None):
        try:
            # If an image is uploaded, use it as a base
            if uploaded_image:
                base_image = Image.open(uploaded_image)
                prompt = text_input or "A beautiful scene"
                image = self.image_model(
                    prompt=prompt, 
                    image=base_image
                ).images[0]
            else:
                # Generate image from text
                prompt = text_input or "A beautiful landscape"
                image = self.image_model(prompt=prompt).images[0]
            
            st.image(image, caption="Generated Image")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
    
    def generate_audio(self, text_input):
        if not text_input:
            st.warning("Please enter a text prompt")
            return
        
        try:
            # Generate audio
            audio = self.audio_model(text_input)[0]
            
            # Save audio
            sf.write("output_audio.wav", audio['audio'], 16000)
            
            # Play audio
            st.audio("output_audio.wav")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
    
    def run(self):
        # Load models
        self.load_models()
        
        # Main application flow
        self.process_multimodal_input()

def main():
    app = MultimodalAIApp()
    app.run()

if __name__ == "__main__":
    main()