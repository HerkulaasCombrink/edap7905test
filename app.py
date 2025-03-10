import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO

st.title("Web Scraper: Extract Tables from Websites")

# Input URL
url = st.text_input("Enter the URL of the website to scrape:")

if st.button("Scrape Data"):
    if url:
        try:
            # Fetch the webpage
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find tables on the page
            tables = soup.find_all("table")
            
            if not tables:
                st.warning("No tables found on this webpage. Try another URL.")
            else:
                all_dfs = []
                
                for i, table in enumerate(tables):
                    df = pd.read_html(str(table))[0]  # Convert HTML table to DataFrame
                    all_dfs.append(df)
                    
                    # Display the table
                    st.write(f"Table {i+1}")
                    st.dataframe(df)
                
                # Combine all tables into one CSV (if multiple tables exist)
                csv_buffer = BytesIO()
                combined_df = pd.concat(all_dfs, ignore_index=True)
                combined_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="Download Data as CSV",
                    data=csv_buffer,
                    file_name="scraped_data.csv",
                    mime="text/csv"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching the webpage: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a valid URL.")
