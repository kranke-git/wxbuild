import requests
from   bs4 import BeautifulSoup

def list_svante_files( url, extension = ".epw" ):
    """
    List only .epw files from an Nginx autoindex directory, ignoring 
    subdirectories and query strings.
    Returns just filenames.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href:
            continue
        # skip directories and links with query strings
        if href.endswith("/") or "?" in href:
            continue
        # only .epw files
        if href.lower().endswith( extension ):
            files.append(href)
    
    return files

def read_nth_line( filepath, n ):
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i == n-1:   # 0-based indexing → nth line
                return line.strip()
    return None