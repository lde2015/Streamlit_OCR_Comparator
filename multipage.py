"""
This file is the framework for generating multiple Streamlit applications
through an object oriented framework.

Source: https://huggingface.co/spaces/deepset/wikipedia-assistant/tree/main
"""

# Import necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu


# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, icon, func) -> None:
        """Class Method to Add pages to the project

        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append(
            {
                "title": title,
                "icon": icon,
                "function": func
            }
        )

    def run(self):
        # Drodown to select the page to run
        st.markdown("""
            <style>
                section[data-testid="stSidebar"] > div:first-of-type {
                    background-color: var(--secondary-background-color);
                    background: var(--secondary-background-color);
                    width: 250px;
                    padding: 0rem 0;
                    box-shadow: -2rem 0px 2rem 2rem rgba(0,0,0,0.16);
                }
                section[aria-expanded="true"] > div:nth-of-type(2) {
                    display: none;
                }
                .main > div:first-of-type {
                    padding: 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)

        with st.sidebar:
            selected = option_menu("OCR Comparator",
                                   [page["title"] for page in self.pages],
                                   icons=[page["icon"] for page in self.pages],
                                   menu_icon="app-indicator", default_index=0)

        # Run the selected page
        for index, item in enumerate(self.pages):
            if item["title"] == selected:
                self.pages[index]["function"]()
                break
