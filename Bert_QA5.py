import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Set Streamlit Page Config
st.set_page_config(page_title="BERT QA", page_icon="🤖", layout="wide")

# Apply Custom CSS for Background Image
st.markdown("""
    <style>
        body {
            background: url('https://source.unsplash.com/1600x900/?sea,forest') no-repeat center center fixed;
            background-size: cover;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .stTextInput, .stTextArea {
            border-radius: 10px;
            border: 2px solid #007BFF;
        }
        .stButton button {
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Instructions
with st.sidebar:
    st.title("📌 How to Use")
    st.markdown("""
    - 📝 **Enter a paragraph** in the text box.
    - ❓ **Type your question** based on the paragraph.
    - 🚀 Click **'Get Answer'** to see results!
    """)
    st.markdown("### 🔍 About This App")
    st.info("This app uses **BERT (Bidirectional Encoder Representations from Transformers)** to answer questions from a given text.")

# Lazy Load Model to Improve Speed
if "model" not in st.session_state:
    with st.spinner("🔄 Loading AI Model... Please wait!"):
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        st.session_state.tokenizer = BertTokenizer.from_pretrained(model_name)
        st.session_state.model = BertForQuestionAnswering.from_pretrained(model_name)
    st.success("✅ AI Model Loaded Successfully!")

# Title
st.markdown("<h1 class='title'>🤖 AI Question Answering with BERT 📝</h1>", unsafe_allow_html=True)
st.write("### 📜 Enter a paragraph and ask a question. AI will provide the answer!")

# Layout - Two Column Design
col1, col2 = st.columns([2, 1])

with col1:
    context = st.text_area("📜 Enter a paragraph (context):", height=180)
    question = st.text_input("❓ Ask a question based on the paragraph:")

with col2:
    st.markdown("### 🔹 Steps to Use")
    st.write("1️⃣ **Enter text** in the left box.")
    st.write("2️⃣ **Ask a relevant question**.")
    st.write("3️⃣ Click **'Get Answer'** to generate results.")

# Answer Section
if st.button("🚀 Get Answer"):
    if not context.strip():
        st.warning("⚠️ Please enter a paragraph.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        # Tokenize input
        inputs = st.session_state.tokenizer(question, context, return_tensors="pt", truncation=True)

        # Get model predictions
        with torch.no_grad():
            outputs = st.session_state.model(**inputs)

        # Extract start and end positions
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores).item()
        end_index = torch.argmax(end_scores).item()

        # Convert token IDs to readable text
        answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
        answer = st.session_state.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Display the answer
        st.markdown("## 💡 Answer:")
        if answer.strip() and answer != "[CLS]":
            st.success(f"✅ {answer}")
        else:
            st.error("❌ Unable to find a valid answer. Try rephrasing the question!")

# Footer
st.markdown("---")
st.markdown("<h6 style='text-align: center;'>Made with ❤️ using Streamlit & Hugging Face 🤗</h6>", unsafe_allow_html=True)
