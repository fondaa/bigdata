import streamlit as st
from Pages import home, pdp, ice, ale, shapley

PAGES = {
    "Home": home,
    "PDP": pdp,
    "ICE": ice,
    "ALE": ale,
    "SHAP": shapley,
}

def main():
#    st.set_page_config(layout="wide")
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f'Loading {selection} ...'):
        page.run()


if __name__ == "__main__":
    main()