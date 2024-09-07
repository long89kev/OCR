import streamlit as st
import pandas as pd
import os
from PIL import Image

@st.cache_data
def load_ocr_data():
    return pd.read_csv(os.path.join("datasets", "ocr_results", "all_ocr_results.csv"))

@st.cache_data
def load_object_data():
    return pd.read_csv(os.path.join("datasets", "object_detection_results", "all_object_detection_results.csv"))

ocr_df = load_ocr_data()
object_df = load_object_data()

st.title("Keyframe Search")

search_type = st.radio("Select search type:", ("Text", "Object"))

if search_type == "Text":
    search_term = st.text_input("Enter a word to search for:")
    if search_term:
        filtered_df = ocr_df[ocr_df['ocr_text'].str.contains(search_term, case=False, na=False)]
        st.write(f"Found {len(filtered_df)} keyframes containing '{search_term}'")
else:
    object_classes = sorted(object_df['class_name'].unique())
    search_term = st.selectbox("Select an object to search for:", object_classes)
    if search_term:
        filtered_df = object_df[object_df['class_name'] == search_term]
        st.write(f"Found {len(filtered_df)} keyframes containing '{search_term}'")

if search_term:
    for _, row in filtered_df.iterrows():
        st.subheader(f"Video: {row['video_name']}, Frame: {row['frame_idx']}")
        img = Image.open(row['keyframe_file'])
        
        if search_type == "Object":
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([row['x1'], row['y1'], row['x2'], row['y2']], outline="red", width=2)
        
        st.image(img, use_column_width=True)
        
        if search_type == "Object":
            st.write(f"Confidence: {row['confidence']:.2f}")
        
        st.markdown("---")
else:
    st.write(f"Select an {'object' if search_type == 'Object' else 'word'} to search for keyframes.")