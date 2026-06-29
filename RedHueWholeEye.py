import sys
from PIL import Image, ImageEnhance
import numpy as np
import os
import csv
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  

def avgFromArray(arr):
    # find average RGB thresh set - single image array
    # RGBA/RGB
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3] > 0
        red_vals = arr[:, :, 0][alpha]
        green_vals = arr[:, :, 1][alpha]
        blue_vals = arr[:, :, 2][alpha]
    else:
        red_vals = arr[:, :, 0].flatten()
        green_vals = arr[:, :, 1].flatten()
        blue_vals = arr[:, :, 2].flatten()
    
    avg_red = np.mean(red_vals) if len(red_vals) > 0 else 0
    avg_green = np.mean(green_vals) if len(green_vals) > 0 else 0
    avg_blue = np.mean(blue_vals) if len(blue_vals) > 0 else 0
    
    return avg_red, avg_green, avg_blue

def arrayToBase64(arr):
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8))
    else:  # is not greyscale
        img = Image.fromarray(arr)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def createDarkOverlay(arr, dark_mask):
    overlay = arr.copy()
    if overlay.shape[2] == 4:  #RGBA
        overlay[dark_mask] = [0, 0, 255, 255]
    else:  #RGB
        overlay = np.dstack((overlay, np.full(overlay.shape[:2], 255, dtype=np.uint8)))  #add alpha channel
        overlay[dark_mask] = [0, 0, 255, 255]
    return overlay

def imagefinal(imagePath):
    #enhance image, return array with avgs
    img = Image.open(imagePath)
    img = ImageEnhance.Color(img).enhance(1.7) 
    img = ImageEnhance.Brightness(img).enhance(1.15)
    arr = np.array(img).astype(float)
    
    # avgs for this image
    avg_red, avg_green, avg_blue = avgFromArray(arr)
    
    return arr, avg_red, avg_green, avg_blue

def calcImage(imagePath, rangeWidth=0, scalerAdj=45):
    #masking with given avg sets. after enhancement
    arr, avg_red, avg_green, avg_blue = imagefinal(imagePath)
    
    #print(f"  Image AVGs =  Red: {avg_red:.2f}, Green: {avg_green:.2f}, Blue: {avg_blue:.2f}")
    
    #get channels
    if arr.shape[2] == 4:  #RGBA
        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        alpha = arr[:, :, 3]
    else:  #RGB
        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        alpha = np.full(red.shape, 255, dtype=np.uint8)
    
    valid = alpha > 0
    
    #adj thresholds 
    red_threshold = max(80, avg_red * 0.7) + scalerAdj
    green_threshold = min(145, avg_green * 1.3) + scalerAdj
    blue_threshold = min(145, avg_blue * 1.3) + scalerAdj
    
    # final mask
    darkMask = (
        (1.4 * red > green) &
        (1.4 * red > blue) &
        (blue < blue_threshold) &
        (green < green_threshold) &
        (red > red_threshold) &
        valid
    )
    
    totalPixels = np.sum(valid)
    darkCount = np.sum(darkMask)
    totalDarkPercent = (darkCount / totalPixels) * 100 if totalPixels > 0 else 0

    #uint8 for saving
    arr_uint8 = arr.astype(np.uint8)
    #to see
    original_base64 = arrayToBase64(arr_uint8)
    rgb_vis = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb_vis[:, :, 0] = red.astype(np.uint8)
    rgb_vis[:, :, 1] = green.astype(np.uint8)
    rgb_vis[:, :, 2] = blue.astype(np.uint8)
    rgb_base64 = arrayToBase64(rgb_vis)
    
    darkOverlay = createDarkOverlay(arr_uint8, darkMask)
    dark_base64 = arrayToBase64(darkOverlay)

    return {
        "Image": os.path.basename(imagePath),
        "Original": original_base64,
        "RGB": rgb_base64,
        "DarkOverlay": dark_base64,
        "TotalDarkPercent": totalDarkPercent
    }

def generateHtmlReport(results, outputFile):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eye Analysis Report (Color Ratio Method)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            img { max-width: 200px; max-height: 200px; display: block; margin: 0 auto; }
            .header-row th { background-color: #4CAF50; color: white; }
            .section-header { background-color: #e0e0e0; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Eye Analysis Report (Color Ratio Method - Per Image Averages)</h1>
        <table>
            <tr class="header-row">
                <th>Image</th>
                <th>Enhanced Original</th>
                <th>PNG</th>
                <th>Dark Regions</th>
                <th>Total Dark %</th>
            </tr>
    """

    for result in results:
        html += f"""
            <tr>
                <td>{result['Image']}</td>
                <td><img src="data:image/png;base64,{result['Original']}" alt="Enhanced Original"></td>
                <td><img src="data:image/png;base64,{result['RGB']}" alt="RGB"></td>
                <td><img src="data:image/png;base64,{result['DarkOverlay']}" alt="Dark Regions"></td>
                <td>{result['TotalDarkPercent']:.1f}%</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(outputFile, 'w') as f:
        f.write(html)

def main():
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()

    print("Select one or more images to analyze.")
    imagePaths = filedialog.askopenfilenames(
        title="Select PNG Images", 
        filetypes=[("PNG files", "*.png")]
    )

    if not imagePaths:
        print("No images selected.")
        return

    print(f"Processing {len(imagePaths)} images")
    
    results = []
    for imagePath in imagePaths:
        print(f"\nProcessing: {os.path.basename(imagePath)}")
        result = calcImage(imagePath)
        results.append(result)

    #CSV
    csvPath = filedialog.asksaveasfilename(
        defaultextension=".csv", 
        filetypes=[("CSV files", "*.csv")], 
        title="Save CSV Report"
    )
    if csvPath:
        with open(csvPath, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Image", "TotalDarkPercent"])
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "Image": row["Image"],
                    "TotalDarkPercent": row["TotalDarkPercent"]
                })
        print(f"\nSaved CSV to {csvPath}")
    
    # Save HTML 
    htmlPath = filedialog.asksaveasfilename(
        defaultextension=".html", 
        filetypes=[("HTML files", "*.html")], 
        title="Save HTML Report"
    )
    if htmlPath:
        generateHtmlReport(results, htmlPath)
        print(f"Saved HTML report to {htmlPath}")

if __name__ == "__main__":
    main()
