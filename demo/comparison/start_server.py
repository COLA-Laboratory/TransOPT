import os
import sys
from pathlib import Path

# Define the current and package paths
current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

# Define the HTML directory
html_dir = package_path / "demo" / "comparison" / "htmls"

# Function to generate index.html
def generate_index_html():
    with open(html_dir / 'index.html', 'w') as index_file:
        index_file.write('<html><body>\n')
        index_file.write('<h1>List of HTML files</h1>\n')
        index_file.write('<ul>\n')

        # Loop through each html file in the directory
        for html_file in html_dir.glob('*.html'):
            link = html_file.name
            # Exclude index.html from the list
            if link != 'index.html':
                index_file.write(f'<li><a href="{link}">{link}</a></li>\n')

        index_file.write('</ul>\n')
        index_file.write('</body></html>')

# Function to start a simple HTTP server
def start_http_server():
    os.chdir(html_dir)  # Change working directory to html directory
    os.system("python -m http.server")  # Start the server

if __name__ == "__main__":
    generate_index_html()  # Generate the index.html file
    start_http_server()  # Start the server