import streamlit as st
from pages import pages

def main():
    page = st.sidebar.selectbox(
        "Выберите страницу",
        list(pages)
    )

    pages[page](st)


if __name__ == '__main__':
    main()
