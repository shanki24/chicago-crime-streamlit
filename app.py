import streamlit as st

st.set_page_config(
    page_title="Chicago Crime Analytics",
    layout="wide"
)

st.title("🚔 Chicago Crime Analytics Dashboard")

st.markdown("""
### What this app shows:
- 📊 Crime Overview & distributions  
- 🗺️ Geographic crime clustering  
- ⏰ Temporal crime patterns  
- 📉 PCA & dimensionality reduction  
- 🏆 Model comparison & evaluation  

Use the **sidebar** to navigate between pages.
""")



st.set_page_config(
    page_title="Chicago Crime Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


