import sys
from PIL import Image
import numpy as np
import os
import csv
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  

def rgbToGrayscale(arr):
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    return (0.2989 * r + 0.5870 * g + 0.1140 * b)

def arrayToBase64(arr):
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8))
    else:  #color
        img = Image.fromarray(arr)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def createMask(arr, mask):
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = arr[..., :3]
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return rgba

def createDarkOverlay(arr, dark_mask):
    overlay = arr.copy()
    if overlay.shape[2] == 4:  #RGBA
        overlay[dark_mask] = [255, 0, 0, 255]  
    else:  #RGB
        overlay = np.dstack((overlay, np.full(overlay.shape[:2], 255, dtype=np.uint8)))  #addalpha channel
        overlay[dark_mask] = [255, 0, 0, 255] 
    return overlay

def calcImage(imagePath, rangeWidth=0, SCALERadj=-22):
    img = Image.open(imagePath)
    arr = np.array(img)
    height, width = arr.shape[:2]
    
    # Get alpha channel and valid pixels
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.full((height, width), 255, dtype=np.uint8)
    valid = alpha > 0
    
    # Find the bounding box of valid (non-transparent) pixels
    if np.any(valid):
        # Get indices of valid pixels
        valid_y, valid_x = np.where(valid)
        
        # Find the center of the oval based on valid pixels
        min_y = np.min(valid_y)
        max_y = np.max(valid_y)
        center_y = (min_y + max_y) / 2
        
        # Also get horizontal center for reference
        min_x = np.min(valid_x)
        max_x = np.max(valid_x)
        center_x = (min_x + max_x) / 2
        
        # Calculate oval dimensions
        oval_height = max_y - min_y
        oval_width = max_x - min_x
        
        # Use the oval's actual center for the slope calculation
        yIndices, xIndices = np.indices((height, width))
        
        # Calculate slope based on oval's actual center
        slopeFactor = 0.4
        verticalOffsetPercent = 0.10  # Changed from 0.25 to 0.15 (10% lower)
        
        # Calculate the slope relative to the oval's center
        x_shifted = xIndices - center_x
        y_shifted = yIndices - center_y
        
        # The slope line should pass through the center with the same slope factor
        baseSlope = (oval_height / oval_width) * x_shifted
        # SUBTRACT the vertical offset to shift the line UPWARD (towards top of image)
        verticalOffset = verticalOffsetPercent * oval_height
        splitLine = (slopeFactor * baseSlope) + center_y - verticalOffset
        
        dorsalMask = (yIndices < splitLine)
        ventralMask = ~dorsalMask
    else:
        # Fallback to original method if no valid pixels found
        yIndices, xIndices = np.indices((height, width))
        slopeFactor = 0.4
        verticalOffsetPercent = 0.10
        baseSlope = (height / width) * xIndices
        verticalOffset = verticalOffsetPercent * height
        splitLine = (slopeFactor * baseSlope) + verticalOffset
        dorsalMask = (yIndices < splitLine)
        ventralMask = ~dorsalMask

    #process/calc image
    grayscale = rgbToGrayscale(arr)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.full(grayscale.shape, 255, dtype=np.uint8)
    
    # Apply masks with alpha
    dorsalMask = dorsalMask & (alpha > 0)
    ventralMask = ventralMask & (alpha > 0)
    
    # CHANGED: Calculate mean for current image only
    non_transparent = (alpha > 0)
    meanAll = np.mean(grayscale[non_transparent])
    meanCorr = meanAll + SCALERadj
    #REPLACE with value for total correction percents
    lowerBound = meanCorr - rangeWidth

    dorsalDarkMask = (grayscale < lowerBound) & (alpha > 0) & dorsalMask
    ventralDarkMask = (grayscale < lowerBound) & (alpha > 0) & ventralMask
    
    dorsalTotal = np.sum((alpha > 0) & dorsalMask)
    ventralTotal = np.sum((alpha > 0) & ventralMask)
    
    dorsalDarkCount = np.sum(dorsalDarkMask)
    ventralDarkCount = np.sum(ventralDarkMask)
    
    dorsalDarkPercent = (dorsalDarkCount / dorsalTotal) * 100 if dorsalTotal > 0 else 0
    ventralDarkPercent = (ventralDarkCount / ventralTotal) * 100 if ventralTotal > 0 else 0

    #7/9/25
    totalpercent = (dorsalDarkCount + ventralDarkCount) / (dorsalTotal + ventralTotal)

    dvRatio = dorsalDarkPercent / ventralDarkPercent if ventralDarkPercent > 0 else float('inf')

    original_base64 = arrayToBase64(arr)
    
    dorsalDisplay = createMask(arr, dorsalMask)
    dorsal_base64 = arrayToBase64(dorsalDisplay)
    
    ventralDisplay = createMask(arr, ventralMask)
    ventral_base64 = arrayToBase64(ventralDisplay)
    
    dorsalDarkOverlay = createDarkOverlay(arr, dorsalDarkMask)
    dorsal_dark_base64 = arrayToBase64(dorsalDarkOverlay)
    
    ventralDarkOverlay = createDarkOverlay(arr, ventralDarkMask)
    ventral_dark_base64 = arrayToBase64(ventralDarkOverlay)
    
    grayscale_base64 = arrayToBase64(grayscale)

    return {
        "Image": os.path.basename(imagePath),
        "Original": original_base64,
        "Grayscale": grayscale_base64,
        "DorsalRegion": dorsal_base64,
        "VentralRegion": ventral_base64,
        "DorsalDark": dorsal_dark_base64,
        "VentralDark": ventral_dark_base64,
        "DorsalDarkPercent": dorsalDarkPercent,
        "VentralDarkPercent": ventralDarkPercent,
        "DVRatio": dvRatio
    }

def generateHtmlReport(results, outputFile):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eye Analysis Report (Grayscale)</title>
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
        <h1>Eye Analysis Report (Grayscale Method)</h1>
        <table>
            <tr class="header-row">
                <th>Image</th>
                <th>Original</th>
                <th>Grayscale</th>
                <th>Dorsal Region</th>
                <th>Ventral Region</th>
                <th>Dorsal Dark</th>
                <th>Ventral Dark</th>
                <th>Dorsal Dark %</th>
                <th>Ventral Dark %</th>
                <th>D/V Ratio</th>
            </tr>
    """

    for result in results:
        html += f"""
            <tr>
                <td>{result['Image']}</td>
                <td><img src="data:image/png;base64,{result['Original']}" alt="Original"></td>
                <td><img src="data:image/png;base64,{result['Grayscale']}" alt="Grayscale"></td>
                <td><img src="data:image/png;base64,{result['DorsalRegion']}" alt="Dorsal Region"></td>
                <td><img src="data:image/png;base64,{result['VentralRegion']}" alt="Ventral Region"></td>
                <td><img src="data:image/png;base64,{result['DorsalDark']}" alt="Dorsal Dark"></td>
                <td><img src="data:image/png;base64,{result['VentralDark']}" alt="Ventral Dark"></td>
                <td>{result['DorsalDarkPercent']:.1f}%</td>
                <td>{result['VentralDarkPercent']:.1f}%</td>
                <td>{result['DVRatio']:.2f}</td>
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

    results = []
    for imagePath in imagePaths:
        print(f"Processing: {os.path.basename(imagePath)}")
        result = calcImage(imagePath)
        results.append(result)

    # Save CSV report
    csvPath = filedialog.asksaveasfilename(
        defaultextension=".csv", 
        filetypes=[("CSV files", "*.csv")], 
        title="Save CSV Report"
    )
    if csvPath:
        with open(csvPath, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "Image", "DorsalDarkPercent", "VentralDarkPercent", "DVRatio"
            ])
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "Image": row["Image"],
                    "DorsalDarkPercent": row["DorsalDarkPercent"],
                    "VentralDarkPercent": row["VentralDarkPercent"],
                    "DVRatio": row["DVRatio"]
                })
        print(f"Saved CSV to {csvPath}")
    
    # Save HTML report
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
