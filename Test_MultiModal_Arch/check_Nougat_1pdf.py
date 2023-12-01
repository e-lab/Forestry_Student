import fitz  # PyMuPDF
from transformers import pipeline
from transformers import AutoFeatureExtractor
import matplotlib.pyplot as plt
from PIL import Image

def display_image_with_text(image, text):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display the image on the left subplot
    ax1.imshow(image)
    ax1.axis('off')

    # Display the predicted text on the right subplot
    ax2.text(0.25, 0.25, f"Predicted Text:\n{text}", fontsize=12, ha='center', va='center')
    ax2.axis('off')

    # Show the figure
    plt.show()

def QT_Gen(pdf_path):
    # Load the layoutLM tokenizer and model
    pipe = pipeline(
        task='image-to-text', 
        model='facebook/nougat-base', 
        feature_extractor=AutoFeatureExtractor,
    )
    
    # Convert PDF to images using PyMuPDF
    pdf_document = fitz.open(pdf_path)
    pages = [pdf_document[i] for i in range(pdf_document.page_count)]

    # Process each page
    for i, page in enumerate(pages):
        image = page.get_pixmap()
        image_pil = Image.frombytes("RGB", [image.width, image.height], image.samples)
        
        response = pipe(
            image_pil, 
            max_new_tokens=pipe.tokenizer.model_max_length
        )
        text = response[0].get(('generated_text'))
        print(f"Page {i + 1} Text:\n{text}")

        # Display the image along with the predicted text
        display_image_with_text(image_pil, text)
        break

if __name__ == '__main__':
    pdf_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/Test_Results/FOR205_Final Exam_Fall 2014_ Sample A 2.pdf"
    QT_Gen(pdf_path)
