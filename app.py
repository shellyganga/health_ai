import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Function to load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = BertForSequenceClassification.from_pretrained('./bert_mod')
    tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model()

def predict(text, model, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)

    # Process the output
    logits = outputs.logits
    predicted_probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label_idx = torch.argmax(predicted_probabilities, dim=1).item()

    # Convert predicted label index to actual label
    idx_to_category = {
        0: 'sports',
        1: 'science',
        2: 'business',
        3: 'entertainment',
        4: 'health',
        5: 'technology'
    }
    predicted_category = idx_to_category[predicted_label_idx]
    return predicted_category


# Streamlit app
st.title('BERT Model Inference')

# User text input
user_input = st.text_area("Enter text for classification:")

# Button to make prediction
if st.button('Predict'):
    if user_input:
        # Make prediction
        cat = predict(user_input, model, tokenizer)
        # Assuming binary classification for simplicity, modify as needed
        st.write(f"prediction: {cat:.4f}")
    else:
        st.write("Please enter text for prediction.")
