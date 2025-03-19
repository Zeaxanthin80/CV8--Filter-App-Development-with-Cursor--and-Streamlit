# Advanced Image Filter App 🎨

An interactive image filtering application built with Streamlit that allows users to apply various image processing techniques to their images. The app includes advanced features like motion deblur, edge detection, and custom kernel filters.

## Features

### Filter Options

1. **Gaussian Blur**
   - Adjustable kernel size for controlling blur intensity
   - Useful for noise reduction and smoothing

2. **Sobel Edge Detection**
   - Adjustable threshold for edge sensitivity
   - Intensity control for edge strength
   - Detects both horizontal and vertical edges

3. **Image Sharpening**
   - Uses unsharp mask technique
   - Adjustable amount and radius controls
   - Enhances image details and edges

4. **Sepia Effect**
   - Classic warm, vintage tone effect
   - Optimized color matrix for authentic sepia look
   - Enhanced red channel for warmth

5. **Motion Deblur**
   - Removes motion blur from images
   - Uses Wiener deconvolution algorithm
   - Adjustable parameters:
     - Blur Length: Controls the estimated motion blur distance
     - Blur Angle: Sets the direction of motion (-90° to 90°)
     - Signal-to-Noise Ratio: Balances detail recovery and noise

6. **Custom Kernel**
   - 3x3 convolution kernel matrix
   - Full control over each kernel value
   - Experiment with different image effects

### Global Adjustments

- **Brightness Control**: -100 to +100
- **Contrast Control**: 0.0 to 3.0
- **Image Position**: 
  - Horizontal offset: -500 to +500 pixels
  - Vertical offset: -500 to +500 pixels

### Additional Features

- Real-time preview of filter effects
- Support for JPG, JPEG, and PNG images
- Download processed images
- User-friendly interface with helpful tooltips
- Responsive layout design

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd image-filter-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Upload an image using the file uploader

3. Select a filter from the dropdown menu

4. Adjust filter parameters using the sliders

5. Fine-tune the image using global adjustments

6. Use position controls to align the processed image if needed

7. Download the processed image using the download button

## Filter Tips

### Motion Deblur
- Start with default settings and adjust gradually
- Increase blur length for stronger motion blur
- Adjust angle to match the direction of motion
- Higher SNR values recover more detail but may increase noise

### Edge Detection
- Lower threshold values detect more edges
- Higher intensity values make edges more prominent
- Best results on high-contrast images

### Custom Kernel Examples
- Edge Detection: Center=8, Surrounding=-1
- Blur: All values=1/9
- Sharpen: Center=5, Sides=-1, Corners=0

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- OpenCV (opencv-python-headless)
- Pillow

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 