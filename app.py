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
                    headers = [header.text.strip() for header in table.find_all("th")]
                    
                    # Extract all rows
                    rows = []
                    max_cols = len(headers)  # Default to header column count
                    
                    for row in table.find_all("tr"):
                        cells = [cell.text.strip() for cell in row.find_all("td")]
                        if len(cells) > max_cols: 
                            max_cols = len(cells)  # Update max column count if a row has more columns

                        if cells:
                            rows.append(cells)

                    # Ensure all rows have the same number of columns
                    for row in rows:
                        while len(row) < max_cols:
                            row.append("")  # Pad missing columns with empty strings
                    
                    # Create DataFrame
                    df = pd.DataFrame(rows, columns=headers if headers else [f"Column {i+1}" for i in range(max_cols)])
                    all_dfs.append(df)

                    # Display table
                    st.write(f"Table {i+1}")
                    st.dataframe(df)
                
                # Combine all tables into one CSV
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
