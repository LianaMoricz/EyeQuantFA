import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import csv
import base64
from io import BytesIO
import matplotlib
from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO

matplotlib.use('Agg')

def extractRed(arr):
    return arr[:, :, 0].astype(float) 

def arrayToBase64(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def createMask(arr, mask):
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = arr[..., :3]
    rgba[..., 3] = np.where(mask, 255, 0)
    return rgba

def avgacrFromArray(allArrays):
    reds, greens, blues = [], [], []
    for arr in allArrays:
        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3]
        else:
            alpha = np.full(red.shape, 255, dtype=np.uint8)
        
        mask = alpha > 0
        reds.extend(red[mask])
        greens.extend(green[mask])
        blues.extend(blue[mask])
    
    return np.mean(reds), np.mean(greens), np.mean(blues)

def imagefinal(imagePath):
    img = Image.open(imagePath)
    img = ImageEnhance.Color(img).enhance(1.7) 
    img = ImageEnhance.Brightness(img).enhance(1.3)
    arr = np.array(img).astype(float)
    
    #three channel avgs
    avg_red, avg_green, avg_blue = avgacrFromArray([arr])
    
    return arr, avg_red, avg_green, avg_blue

def calcImage(imagePath, rangeWidth=0, scalerAdj=50):
   
    arr, avg_red, avg_green, avg_blue = imagefinal(imagePath)
    height, width = arr.shape[:2]

    yIndices, xIndices = np.indices((height, width))
    slopeFactor = 0.4
    verticalOffsetPercent = 0.25
    baseSlope = (height / width) * xIndices
    verticalOffset = verticalOffsetPercent * height
    splitLine = (slopeFactor * baseSlope) + verticalOffset
    dorsalMask = (yIndices < splitLine)
    ventralMask = ~dorsalMask

    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    alpha = arr[..., 3]
    valid = alpha > 0

    dorsalMask = dorsalMask & valid
    ventralMask = ventralMask & valid

    red_threshold = max(80, avg_red * 0.7) + scalerAdj
    green_threshold = min(145, avg_green * 1.3) + scalerAdj
    blue_threshold = min(145, avg_blue * 1.3) + scalerAdj

    redRatioMask = (
        (1.4 * red > green) &
        (1.4 * red > blue) &
        (blue < blue_threshold) &
        (green < green_threshold) &
        (red > red_threshold) &
        valid
    )

    dorsalDarkMask = dorsalMask & redRatioMask 
    ventralDarkMask = ventralMask & redRatioMask

    dorsalTotalArea = np.sum(dorsalMask)
    ventralTotalArea = np.sum(ventralMask)

    dorsalDarkCount = np.sum(dorsalDarkMask)
    ventralDarkCount = np.sum(ventralDarkMask)

    dorsalDarkPercent = (dorsalDarkCount / dorsalTotalArea * 100) if dorsalTotalArea > 0 else 0
    ventralDarkPercent = (ventralDarkCount / ventralTotalArea * 100) if ventralTotalArea > 0 else 0
    dvRatio = dorsalDarkPercent / ventralDarkPercent if ventralDarkPercent > 0 else float('inf')

    def overlayMask(mask):
        overlay = arr.copy()
        overlay[mask] = [255, 0, 0, 255]
        return overlay

    dorsalDisplay = overlayMask(dorsalMask)
    ventralDisplay = overlayMask(ventralMask)
    dorsalDarkDisplay = overlayMask(dorsalDarkMask)
    ventralDarkDisplay = overlayMask(ventralDarkMask)
    redDominantDisplay = overlayMask(redRatioMask)

    return {
        "Image": os.path.basename(imagePath),
        "Original": arrayToBase64(arr),
        "DorsalRegion": arrayToBase64(dorsalDisplay),
        "VentralRegion": arrayToBase64(ventralDisplay),
        "DorsalDark": arrayToBase64(dorsalDarkDisplay),
        "VentralDark": arrayToBase64(ventralDarkDisplay),
        "DorsalDarkPercent": dorsalDarkPercent,
        "VentralDarkPercent": ventralDarkPercent,
        "DVRatio": dvRatio,
        "RedDominantMask": arrayToBase64(redDominantDisplay)
    }

def generateHtmlReport(results, outputFile):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eye Quantification Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            img { max-width: 200px; max-height: 200px; }
            .header-row th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <h1>Eye Quantification Report</h1>
        <table>
            <tr class="header-row">
                <th>Image</th>
                <th>Red Dominant Mask</th>
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
                <td><img src="data:image/png;base64,{result['RedDominantMask']}" alt="Red Dominant"></td>
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
    root = tk.Tk()
    root.withdraw()
    print("Select one or more images:")
    imagePaths = filedialog.askopenfilenames(title="Select (PNG) Images", filetypes=[("PNG files", "*.png")])
    if not imagePaths:
        return

    results = []
    for imagePath in imagePaths:
        print(f"Processing: {os.path.basename(imagePath)}")
        result = calcImage(imagePath)
        results.append(result)

    csvPath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save CSV Report")
    if csvPath:
        with open(csvPath, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Image", "DorsalDarkPercent", "VentralDarkPercent", "DVRatio"])
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "Image": row["Image"],
                    "DorsalDarkPercent": row["DorsalDarkPercent"],
                    "VentralDarkPercent": row["VentralDarkPercent"],
                    "DVRatio": row["DVRatio"]
                })
        print(f"Saved CSV to {csvPath}")
    
    htmlPath = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html")], title="Save HTML Report")
    if htmlPath:
        generateHtmlReport(results, htmlPath)
        print(f"Saved HTML report to {htmlPath}")

if __name__ == "__main__":
    main()
