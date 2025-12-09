import streamlit as st
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np

# --- Configuration ---
# You can change this to a smaller model for faster loading, e.g., 'glove-wiki-gigaword-50'
MODEL_NAME = 'word2vec-google-news-300' 

# --- Model Loading ---
@st.cache_resource
def load_word2vec_model():
    """Loads a pre-trained Word2Vec model using gensim.downloader."""
    try:
        # st.spinner is a UI element to show progress
        with st.spinner(f"Downloading and loading pre-trained model: {MODEL_NAME}..."):
            # Loading a model with limit=50000 can reduce memory consumption for quick tests
            # model = KeyedVectors.load_word2vec_format('path/to/your/googlenews-vectors-negative300.bin', binary=True, limit=50000)
            
            # Using gensim.downloader for easy access
            model = api.load(MODEL_NAME)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("The model is large and may require significant memory/time.")
        return None

# --- Main Streamlit Interface ---
st.title("🧠 Word2Vec Analogy Solver")
st.markdown("Demonstrates the vector arithmetic property: **A is to B as C is to ?**")

word_vectors = load_word2vec_model()

if word_vectors:
    st.subheader("Analogy Input: $A - B + C \approx D$")
    

    # --- Input Fields ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        word_a = st.text_input("A (Negative): e.g., man", value="man").lower()
    with col2:
        word_b = st.text_input("B (Positive): e.g., king", value="king").lower()
    with col3:
        word_c = st.text_input("C (Positive): e.g., woman", value="woman").lower()
        
    st.markdown(f"**Operation:** $\\vec{{{word_b}}} - \\vec{{{word_a}}} + \\vec{{{word_c}}} \\approx \\vec{{D}}$")

    if st.button("Solve Analogy"):
        if not all([word_a, word_b, word_c]):
            st.warning("Please enter all three words.")
        else:
            # --- Word2Vec Operation ---
            try:
                # The gensim KeyedVectors.most_similar method handles the vector math: 
                # most_similar(positive=[B, C], negative=[A])
                results = word_vectors.most_similar(
                    positive=[word_b, word_c], 
                    negative=[word_a], 
                    topn=5 # Find the top 5 closest words
                )
                
                st.subheader("✅ Analogy Result (D)")
                
                # Extract the top result and its similarity score
                top_word, similarity = results[0]
                st.success(f"**{word_b}** is to **{word_a}** as **{word_c}** is to **{top_word}**")
                st.write(f"Confidence (Cosine Similarity): **{similarity:.4f}**")
                
                # --- Demonstration of Word2Vec operation ---
                st.subheader("Top 5 Closest Words in the Vector Space")
                
                table_data = [{"Rank": i + 1, "Word": word, "Similarity": f"{score:.4f}"} for i, (word, score) in enumerate(results)]
                st.table(table_data)

            except KeyError as e:
                st.error(f"One or more words not found in the vocabulary: {e}. Try simpler or more common words.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")