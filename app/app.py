import streamlit as st
import pandas as pd
import os
from style import *
from test import find_similar_images
import random


# Page configuration
st.set_page_config(
    page_title="Similar Item Gallery",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(styles, unsafe_allow_html=True)

# Base directory and data loading
BASE_DIR = "../../data/"
metadata_path = os.path.join(BASE_DIR, "metadata.csv")
metadata = pd.read_csv(metadata_path)

def load_random_images(product_type_name, section_name=None):
    # Reverse lookup to get the numeric codes based on names
    product_type = [k for k, v in types_labels.items() if v == product_type_name][0]

    # Filter metadata conditionally on section availability
    if section_name:
        section = [k for k, v in section_labels.items() if v == section_name][0]
        filtered_metadata = metadata[
            (metadata["Product Type"] == product_type) &
            (metadata["Section"] == section)
        ]
    else:
        filtered_metadata = metadata[
            (metadata["Product Type"] == product_type)
        ]

    if filtered_metadata.empty:
        st.warning("No images found with the selected filters.")
        return None, []

    # Randomly select a base image
    random_image = filtered_metadata.sample(1)
    base_image_path = random_image["Path"].values[0]

    # Extract the 12 most similar images using the imported utility function
    comparison_paths = filtered_metadata["Path"] # Path of images that are label=sector
    similar_paths = find_similar_images(base_image_path, comparison_paths, top_k=12)

    return random_image, similar_paths


# Sidebar filters with proper label names
st.sidebar.header("Filters")
selected_product_type = st.sidebar.selectbox("Type", list(types_labels.values()))

# If the selected product type has a section requirement -- Home
if selected_product_type != 'Home': 
    selected_section = st.sidebar.selectbox("Section", list(section_labels.values()))
else:
    selected_section = None

# Load random and similar images based on selected filters
random_image, similar_images = load_random_images(selected_product_type, selected_section)

st.markdown('<div class="title">Item Gallery</div>', unsafe_allow_html=True)

# Layout with two columns
col1, col2 = st.columns([1, 2])

if random_image is not None:
    random_image_path = os.path.join(BASE_DIR, random_image["Path"].iloc[0])
    with col1:
        st.image(random_image_path, width=300, use_column_width=False)

    with col2:
        st.markdown('<div class="subheader">Similar Products</div>', unsafe_allow_html=True)
        #similar_images_paths = [os.path.join(BASE_DIR, path) for path in similar_images["Path"]]
        st.image(similar_images, width=120, caption=[''] * len(similar_images))

        # Button to reload
        if st.button("Reload Item"):
            st.rerun()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Created with ❤️ by Sergi, Joan, Elena & Júlia")










