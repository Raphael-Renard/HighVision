import io
import os
import requests
from tqdm import tqdm
from PIL import Image
from time import sleep
import xml.etree.ElementTree as ET

gallicaURL = 'https://gallica.bnf.fr'
completePrefix = "ark:/12148/"
encoder = requests.utils.requote_uri

# ark : ark:/.../.../date
def getYearsFor(ark):
    url = gallicaURL + f"/services/Issues?ark={encoder(ark)}"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    print("Number of issues:", root.get('totalIssues'))
    return [int(year.text) for year in root.findall('year')]

# ark : ark:/.../.../date
def getItemsFor(ark, year):
    url = gallicaURL + f"/services/Issues?ark={encoder(ark)}&date={year}"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    return [(item.get('ark'), str(year) + "_" + "{0:03}".format(int(item.get('dayOfYear')))) for item in root.findall('issue')]

def getXmlFor(itemArk):
    url = gallicaURL + f"/services/OAIRecord?ark={itemArk}"
    response = requests.get(url)
    return response.text

def getPagesFor(itemArk):
    url = gallicaURL + f"/services/Pagination?ark={itemArk}"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    return [int(page.find("ordre").text) for page in root.find('pages').findall('page')]

# itemArk : ark:/.../...
# qualifier : 'thumbnail', 'lowres', 'medres', 'highres'
def getImageFor(itemArk, page, qualifier):
    url = gallicaURL + f"/{encoder(itemArk)}/f{page}/{qualifier}"
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))

def download(ark, path):
    years = getYearsFor(ark)
    for year in tqdm(years):
        items = getItemsFor(ark, year)
        for itemArk, date in tqdm(items, desc=f"{year}"):
            if not os.path.exists(f"{path}/xml/{date}.xml"):
                xml = getXmlFor(itemArk)
                with open(f"{path}/xml/{date}.xml", "w") as f:
                    f.write(xml)
            pages = getPagesFor(itemArk)
            completeItemArk = completePrefix + itemArk
            for page in tqdm(pages, desc=f"{date}"):
                if os.path.exists(f"{path}/pages/{date}_p{page}.jpg"):
                    continue
                image = None
                while image is None:
                    try:
                        image = getImageFor(completeItemArk, page, 'highres')
                    except:
                        sleep(30)
                image.save(f"{path}/pages/{date}_p{page}.jpg")