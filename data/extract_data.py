#####################
##  Extract Data  ###
#####################

# Imports____________
import gdown
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
import csv
#____________________

''' Drive Downloand
# URL del archivo en Google Drive
url = 'https://drive.google.com/uc?id=1OaIzEt20LQk1ixO5UFx7wvOTZef3PKww'
output_csv, csv_file = 'links_dataset.csv', 'metadata.csv'
gdown.download(url, output_csv, quiet=False)
'''


# Load the dataset
df = pd.read_csv('dataset.csv')
images_dir = 'images_dataset/'

# Create a DataFrame to store additional information
df_new = pd.DataFrame(columns=['Path', 'Year', 'Season', 'Product Type', 'Section'])
 

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as f:
    
    writer = csv.writer(f)
    writer.writerow(['Path', 'Year', 'Season', 'Product Type', 'Section'])
    
    count = 0
    while count <  510:
        for index, row in df.iterrows():
            # Iterate over the three image URL columns
            for i, col in enumerate(df.columns):
                # Get the image URL from the current column
                img_url = row[col]
                # Check if the URL is valid
                if pd.notna(img_url):
                    # Create the image name
                    image_name = f'image_{index}_{i}.jpg'
                    try:
                        # Check if the image has already been processed and stored
                        if image_name not in os.listdir(images_dir):
                            # Download the image from the URL
                            response = requests.get(img_url)
                            # Convert the response to an image object
                            img = Image.open(BytesIO(response.content))
                            # Save the image to the images directory
                            img.save(f'{images_dir}/{image_name}')
                            # Get additional information from the current row
                            year, season, product_type, section = row[col].split('/')[6:10]
                            # Write the data to the CSV file
                            writer.writerow([f'{images_dir}{image_name}', year, season, product_type, section])
                        else:
                            # Get additional information from the current row
                            year, season, product_type, section = row[col].split('/')[6:10]
                            # Write the data to the CSV file
                            writer.writerow([f'{images_dir}{image_name}', year, season, product_type, section])
                    except Exception as e:
                        print(f'Error processing image in row {index} and column {col}: {e}')
                count +=1
