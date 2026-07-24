import streamlit as st

st.title("Crop recommendatin")

user_input = st.text_input("Enter some data:")
if user_input:
    # Run logic tested in Jupyter
    st.write(f"Processed output: {user_input.upper()}")