import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader

import tensorflow as tf

# Load your trained model here
model = tf.keras.models.load_model(r'C:\Users\vidit\Project\vertebrae_xray_model_best.hdf5')

def set_background_image():
    background_image = """
    <style>
    [data-testid="stSidebar"] {
        background-image: url("https://static.vecteezy.com/system/resources/thumbnails/017/398/144/small/healthcare-services-annual-health-checkup-heart-rate-pulse-measurement-pills-antibiotics-dna-diseases-genes-illustration-on-light-blue-hexagonal-background-health-and-medicine-concept-vector.jpg");
        background-size: cover;
    }
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((224, 224))  # Resize to 224x224
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def main():
    st.set_page_config(page_title="Vertebral X-ray Analysis", layout="wide")
    
    set_background_image()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "X-ray Analysis", "Download Report"])
    
    if page == "Introduction":
        introduction_page()
    elif page == "X-ray Analysis":
        xray_analysis_page()
    elif page == "Download Report":
        download_report_page()

def introduction_page():
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #F0F0F0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("# Deep Learning based Analysis of Vertebral X-ray Images")
    
    st.markdown("---")
    
    st.header("Introduction")
    st.write("""
    Welcome to our project on the analysis of vertebral X-ray images using deep learning techniques. 
    This project aims to classify two common vertebral column diseases: Scoliosis and Spondylolisthesis.
    """)
    
    st.markdown("  ")  # Line break
    
    st.subheader("Vertebral Column Diseases")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Scoliosis")
        st.write("""
        Scoliosis is a sideways curvature of the spine that occurs most often during the growth spurt just before puberty.
        
        Key points:
        - Affects 2-3% of the population
        - Can develop at any age, but is most common in adolescents
        - May be caused by conditions such as cerebral palsy and muscular dystrophy
        """)
    
    with col2:
        st.markdown("### Spondylolisthesis")
        st.write("""
        Spondylolisthesis is a spinal condition that causes lower back pain. It occurs when one of your vertebrae slides forward over the bone directly beneath it.
        
        Key points:
        - Can affect both children and adults
        - Often caused by a combination of genetics and environmental factors
        - Symptoms include lower back pain, muscle tightness, and tenderness
        """)
    
    st.markdown("---")
    
    st.header("Project Aim")
    st.write("""
    The primary objective of this project is to develop a deep learning model capable of accurately classifying 
    Scoliosis and Spondylolisthesis from vertebral X-ray images. By leveraging advanced machine learning techniques, 
    we aim to assist medical professionals in making faster and more accurate diagnoses.
    """)
    
    st.markdown("  ")  # Line break
    
    st.subheader("Comparison of Conditions")
    comparison_data = [
        ["Aspect", "Scoliosis", "Spondylolisthesis"],
        ["Definition", "Sideways curvature of the spine", "Forward slippage of one vertebra over another"],
        ["Age of Onset", "Often during adolescence", "Can occur at any age"],
        ["Main Symptom", "Visible spine curvature", "Lower back pain"],
        ["Diagnosis", "X-ray, physical exam", "X-ray, CT scan, MRI"],
        ["Treatment", "Bracing, surgery in severe cases", "Physical therapy, surgery in severe cases"]
    ]
    
    st.table(comparison_data)
    
    st.markdown("  ")  # Line break
    
    st.markdown("""
    **Footnotes:**
    
    1. The accuracy of the deep learning model may vary and results should always be verified by a qualified healthcare professional.
    """)

def xray_analysis_page():
    st.title("X-ray Analysis")
    
    # Patient Information Form
    st.header("Patient Information")
    patient_name = st.text_input("Patient Name")
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_email = st.text_input("Email")
    patient_blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    patient_age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    # X-ray Image Upload
    st.header("Upload X-ray Image")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display image with fixed size and border
        st.image(image, caption='Uploaded X-ray Image', width=400, use_column_width=False)
        st.markdown(
            """
            <style>
            img {
                border: 2px solid #000000;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if st.button("Analyze X-ray"):
            # Preprocess the image (resize, normalize, etc.)
            processed_image = preprocess_image(image)
            
            # Make prediction using the model
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)

            # Mapping predicted class index to actual class label
            class_labels = ['Normal', 'Scoliosis', 'Spondylolisthesis']
            predicted_label = class_labels[predicted_class[0]]
            confidence = float(prediction[0][predicted_class[0]]) * 100

            st.success(f"Analysis complete! Predicted condition: {predicted_label} (Confidence: {confidence:.2f}%)")
            
            # Save the results in session state for the report page
            st.session_state['patient_info'] = {
                "name": patient_name,
                "gender": patient_gender,
                "email": patient_email,
                "blood_group": patient_blood_group,
                "age": patient_age,
                "prediction": predicted_label,
                "confidence": confidence,
                "image": uploaded_file
            }
            
            st.info("You can now go to the 'Download Report' page to generate and download the patient report.")

def download_report_page():
    st.title("Download Patient Report")
    
    if 'patient_info' not in st.session_state:
        st.warning("No patient information available. Please complete the X-ray analysis first.")
        return
    
    patient_info = st.session_state['patient_info']
    
    st.write(f"Patient Name: {patient_info['name']}")
    st.write(f"Predicted Condition: {patient_info['prediction']}")
    
    if st.button("Generate and Download Report"):
        pdf = generate_pdf_report(patient_info)
        st.download_button(
            label="Download PDF Report",
            data=pdf,
            file_name=f"{patient_info['name']}_report.pdf",
            mime="application/pdf"
        )

def generate_pdf_report(patient_info):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add border to the entire page
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.rect(20, 20, width - 40, height - 40)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "Patient Report")

    # Horizontal rule below title
    c.setLineWidth(1)
    c.line(50, height - 70, width - 50, height - 70)

    # Patient Information Table
    data = [
        ["Patient Information", ""],
        ["Name", patient_info['name']],
        ["Gender", patient_info['gender']],
        ["Age", str(patient_info['age'])],
        ["Blood Group", patient_info['blood_group']],
        ["Email", patient_info['email']],
        ["Predicted Condition", patient_info['prediction']],
        ["Confidence", f"{patient_info['confidence']:.2f}%"]
    ]

    table = Table(data, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    table.wrapOn(c, width - 100, height)
    table.drawOn(c, 50, height - 300)

    # Horizontal rule below table
    c.setLineWidth(1)
    c.line(50, height - 320, width - 50, height - 320)

    # Add X-ray image
    if patient_info['image']:
        img = Image.open(patient_info['image'])
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_reader = ImageReader(img_byte_arr)
        
        # Fixed size for the image
        display_width = 300
        display_height = 300
        
        # Calculate position to center the image
        x = (width - display_width) / 2
        y = height - 650
        
        # Draw the image
        c.drawImage(img_reader, x, y, width=display_width, height=display_height, mask='auto')
        
        # Add border to the image
        c.rect(x, y, display_width, display_height, stroke=1, fill=0)
        
        # Add label below the image
        c.setFont("Helvetica", 10)
        c.drawCentredString(width / 2, y - 20, f"X-ray image of spine for {patient_info['name']}")

    # Add footer
    c.setFont("Helvetica", 8)
    c.drawCentredString(width / 2, 30, "This report is generated automatically and should be reviewed by a medical professional.")

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main()