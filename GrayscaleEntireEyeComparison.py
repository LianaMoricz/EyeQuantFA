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

def avgacrAll(allImages):
    existingPix = []
    for eachImage in allImages:
        img = Image.open(eachImage)
        arr = np.array(img)
        grayscale = rgbToGrayscale(arr)
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3]
        else:
            alpha = np.full(grayscale.shape, 255, dtype=np.uint8)
        existingPix.extend(grayscale[alpha > 0])
    avg = np.mean(existingPix)
    return avg

def arrayToBase64(arr):
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8))
    else:  #color
        img = Image.fromarray(arr)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def createDarkOverlay(arr, dark_mask):
    overlay = arr.copy()
    if overlay.shape[2] == 4:  #RGBA
        overlay[dark_mask] = [0, 255, 0, 255]
    else:  #RGB
        overlay = np.dstack((overlay, np.full(overlay.shape[:2], 255, dtype=np.uint8)))  #addalpha channel
        overlay[dark_mask] = [0, 255, 0, 255]
    return overlay

def calcImage(imagePath, overall_mean, rangeWidth=0, SCALERadj=-26):
    img = Image.open(imagePath)
    arr = np.array(img)
    
    #calc image
    grayscale = rgbToGrayscale(arr)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.full(grayscale.shape, 255, dtype=np.uint8)
    
    #batchmean
    meanCorr = overall_mean + SCALERadj
    lowerBound = meanCorr - rangeWidth


    darkMask = (grayscale < lowerBound) & (alpha > 0)
    
    totalPixels = np.sum(alpha > 0)
    darkCount = np.sum(darkMask)
    totalDarkPercent = (darkCount / totalPixels) * 100 if totalPixels > 0 else 0

    original_base64 = arrayToBase64(arr)
    grayscale_base64 = arrayToBase64(grayscale)
    darkOverlay = createDarkOverlay(arr, darkMask)
    dark_base64 = arrayToBase64(darkOverlay)

    return {
        "Image": os.path.basename(imagePath),
        "Original": original_base64,
        "Grayscale": grayscale_base64,
        "DarkOverlay": dark_base64,
        "TotalDarkPercent": totalDarkPercent
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
                <th>Dark Regions</th>
                <th>Total Dark %</th>
            </tr>
    """

    for result in results:
        html += f"""
            <tr>
                <td>{result['Image']}</td>
                <td><img src="data:image/png;base64,{result['Original']}" alt="Original"></td>
                <td><img src="data:image/png;base64,{result['Grayscale']}" alt="Grayscale"></td>
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

    #mean all images in batch
    overall_mean = avgacrAll(imagePaths)
    print(f"Overall mean for batch: {overall_mean:.2f}")

    results = []
    for imagePath in imagePaths:
        print(f"Processing: {os.path.basename(imagePath)}")
        result = calcImage(imagePath, overall_mean)
        results.append(result)


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
