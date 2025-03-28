import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

def apply_gaussian_blur(image, kernel_size):
    """
    Apply Gaussian blur to reduce image noise and detail
    Args:
        image: Input image array
        kernel_size: Size of the Gaussian kernel (must be odd)
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_sobel_edge_detection(image, threshold, intensity):
    """
    Apply Sobel edge detection to find image edges
    Args:
        image: Input image array
        threshold: Minimum intensity for edge detection
        intensity: Strength of edge detection
    Returns:
        Image with detected edges
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Sobel operators with adjustable scale
    scale = intensity / 100.0  # Convert intensity to scale factor
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=scale)  # Detect vertical edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=scale)  # Detect horizontal edges
    
    # Calculate magnitude of edges
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize and apply threshold
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    _, thresholded = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert back to color if input was color
    if len(image.shape) == 3:
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    
    return thresholded

def apply_custom_kernel(image, kernel):
    """
    Apply a custom convolution kernel to the image
    Args:
        image: Input image array
        kernel: 3x3 convolution kernel matrix
    Returns:
        Filtered image
    """
    return cv2.filter2D(image, -1, kernel)

def apply_brightness_contrast(image, brightness, contrast):
    """
    Adjust image brightness and contrast
    Args:
        image: Input image array
        brightness: Brightness adjustment value (-100 to 100)
        contrast: Contrast adjustment value (0.0 to 3.0)
    Returns:
        Adjusted image
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def apply_sharpen(image, amount, radius):
    """
    Sharpen image using unsharp mask technique
    Args:
        image: Input image array
        amount: Sharpening strength
        radius: Size of the sharpening kernel
    Returns:
        Sharpened image
    """
    # Create a gaussian blur of the image for the unsharp mask
    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    
    # Calculate the unsharp mask by subtracting blur from original
    unsharp_mask = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    
    # Ensure pixel values are within valid range
    return np.clip(unsharp_mask, 0, 255).astype(np.uint8)

def apply_sepia(image):
    """
    Apply sepia tone effect to image
    Args:
        image: Input image array
    Returns:
        Sepia-toned image
    """
    # Convert to float32 for calculations
    img_float = image.astype(float) / 255.0
    
    # Define sepia color transformation matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],  # Red channel
        [0.349, 0.686, 0.168],  # Green channel
        [0.272, 0.534, 0.131]   # Blue channel
    ])
    
    # Apply the sepia transformation
    sepia_img = cv2.transform(img_float, sepia_matrix)
    
    # Add warmth by increasing red channel
    sepia_img[:,:,2] = sepia_img[:,:,2] * 1.15
    
    # Normalize and convert back to uint8
    sepia_img = np.clip(sepia_img * 255, 0, 255).astype(np.uint8)
    return sepia_img

def apply_motion_deblur(image, length, angle, snr):
    """
    Remove motion blur from images using Wiener deconvolution
    Args:
        image: Input blurred image
        length: Length of the motion blur
        angle: Angle of motion in degrees
        snr: Signal to noise ratio
    Returns:
        Deblurred image
    """
    # Store original image dimensions and color information
    original_shape = image.shape
    
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure image dimensions are even for FFT
    h, w = gray.shape
    h_pad = h + (h % 2)
    w_pad = w + (w % 2)
    
    # Pad image if necessary
    if h_pad != h or w_pad != w:
        gray_padded = np.pad(gray, ((0, h_pad - h), (0, w_pad - w)), mode='edge')
    else:
        gray_padded = gray
    
    # Create Point Spread Function (PSF) for motion blur
    psf = np.zeros((h_pad, w_pad))
    center = (w_pad // 2, h_pad // 2)
    cv2.ellipse(psf, center, (0, int(length / 2)), 
                90 - angle, 0, 360, 255, -1)
    psf = psf / psf.sum()  # Normalize PSF
    
    # Convert to frequency domain
    gray_freq = np.fft.fft2(gray_padded)
    psf_freq = np.fft.fft2(psf)
    
    # Apply Wiener deconvolution filter
    nsr = 1 / snr  # Noise to signal ratio
    wiener_filter = np.conj(psf_freq) / (np.abs(psf_freq) ** 2 + nsr)
    deblurred = np.real(np.fft.ifft2(gray_freq * wiener_filter))
    
    # Post-process the result
    deblurred = deblurred[:h, :w]  # Crop to original size
    deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
    
    # Convert back to original color format if needed
    if len(original_shape) == 3:
        deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        if original_shape[2] == 4:  # Handle RGBA images
            alpha = image[:, :, 3]
            deblurred = cv2.cvtColor(deblurred, cv2.COLOR_BGR2BGRA)
            deblurred[:, :, 3] = alpha
    
    return deblurred

def apply_image_offset(image, x_offset, y_offset):
    """
    Shift image position using numpy roll
    Args:
        image: Input image array
        x_offset: Horizontal shift in pixels
        y_offset: Vertical shift in pixels
    Returns:
        Shifted image
    """
    if x_offset != 0:
        image = np.roll(image, x_offset, axis=1)  # Shift horizontally
    if y_offset != 0:
        image = np.roll(image, y_offset, axis=0)  # Shift vertically
    return image

def main():
    """
    Main application function that creates the Streamlit interface
    and handles user interactions
    """
    # Configure the Streamlit page
    st.set_page_config(page_title="Advanced Image Filter App", layout="wide")
    
    # Add CSS styling for proper image display and layout
    st.markdown("""
        <style>
        /* Style for image containers */
        .stImage {
            text-align: center;
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
        /* Style for images */
        .stImage > img {
            max-width: 100%;
            height: auto;
            margin: 0;
            display: block;
            border-radius: 5px;
            object-fit: contain;
        }
        /* Column layout styling */
        [data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            width: 100%;
            padding: 0 10px;
            gap: 0;
        }
        /* Container styling */
        .block-container {
            max-width: 95%;
            padding: 2rem;
        }
        /* Remove default margins */
        .element-container {
            margin: 0;
        }
        /* Remove vertical gaps */
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            padding: 10px;
        }https://github.com/Zeaxanthin80/HCI-Application/blob/708c5551719fe03969fe58a81cddf39da8fd166b/Applications/06_02_Web_game.ipynb
        h1 {
            color: #4CAF50;  /* Change title color */
            text-align: center;  /* Center the title */
       font-size: 2.5em;  /* Increase font size */
                font-family: 'Arial', sans-serif;
                padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    
    # Application title and description
    st.title("🎨 Advanced Image Filter App")
    st.write("Upload an image and apply various filters to transform it!")

    # Image upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and process the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Create two-column layout
        col1, col2 = st.columns(2)

        # Left column: Original image and controls
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True, clamp=True)

            # Filter selection and controls
            st.subheader("Filter Controls")
            filter_type = st.selectbox(
                "Select Filter",
                ["None", "Gaussian Blur", "Edge Detection", "Sharpen", "Sepia", "Custom", "Motion Deblur"]
            )

            # Initialize processed image
            processed_image = img_array.copy()

            # Apply selected filter
            if filter_type == "Gaussian Blur":
                kernel_size = st.slider("Blur Intensity", 1, 99, 5, step=2)
                processed_image = apply_gaussian_blur(img_array, kernel_size)

            elif filter_type == "Edge Detection":
                st.write("Edge Detection Controls")
                threshold = st.slider("Edge Threshold", 0, 255, 100)
                intensity = st.slider("Edge Intensity", 1, 200, 100)
                processed_image = apply_sobel_edge_detection(img_array, threshold, intensity)

            elif filter_type == "Sharpen":
                st.write("Sharpening Controls")
                amount = st.slider("Sharpness Amount", 0.0, 2.0, 0.5, 0.1)
                radius = st.slider("Sharpness Radius", 3, 11, 3, 2)
                processed_image = apply_sharpen(img_array, amount, radius)

            elif filter_type == "Sepia":
                processed_image = apply_sepia(img_array)

            elif filter_type == "Custom":
                st.write("Custom Kernel (3x3)")
                kernel = np.zeros((3, 3))
                for i in range(3):
                    cols = st.columns(3)
                    for j in range(3):
                        kernel[i, j] = cols[j].number_input(f"K{i}{j}", value=0.0, format="%.2f")
                
                if st.button("Apply Custom Filter"):
                    processed_image = apply_custom_kernel(img_array, kernel)

            elif filter_type == "Motion Deblur":
                st.write("Motion Deblur Controls")
                st.info("Use these controls to remove motion blur from images. Adjust the parameters to match the blur in your image.")
                
                length = st.slider("Blur Length", 1, 150, 50, 
                                 help="Length of the motion blur. Increase for longer motion trails.")
                angle = st.slider("Blur Angle", -90, 90, 0,
                                help="Angle of the motion in degrees. 0° is horizontal, 90° is vertical.")
                snr = st.slider("Signal-to-Noise Ratio", 100, 1000, 300,
                              help="Higher values preserve more detail but may increase noise.")
                
                processed_image = apply_motion_deblur(img_array, length, angle, snr)

            # Global image adjustments
            st.subheader("Global Adjustments")
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", 0.0, 3.0, 1.0, 0.1)
            
            if brightness != 0 or contrast != 1.0:
                processed_image = apply_brightness_contrast(processed_image, brightness, contrast)

            # Image position adjustment controls
            st.subheader("Image Position Adjustments")
            x_offset = st.slider("Horizontal Offset", -500, 500, 0,
                               help="Adjust the horizontal position of the processed image")
            y_offset = st.slider("Vertical Offset", -500, 500, 0,
                               help="Adjust the vertical position of the processed image")
            
            if x_offset != 0 or y_offset != 0:
                processed_image = apply_image_offset(processed_image, x_offset, y_offset)

        # Right column: Processed image and download button
        with col2:
            st.subheader("Processed Image")
            # Add styling for processed image container
            st.markdown("""
                <style>
                .processed-image-container {
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 10px 0;
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Display processed image
            st.image(processed_image, use_container_width=True, clamp=True)

            # Download button for processed image
            if st.button("Download Processed Image"):
                processed_pil = Image.fromarray(processed_image)
                buf = io.BytesIO()
                processed_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

# Entry point of the application
if __name__ == "__main__":
    main() 