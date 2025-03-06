import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options

# Put your own profile here. In case of human verification, do it from a manual tab.
# Do not keep another tab from the same profile open while using it. 
profilePath = "/home/samuel-g/.mozilla/firefox/aba8fcod.Selenium"

class britishNewspaperArchiveExplorer():
    def __init__(self, title, startYear, endYear, downloadDir):
        self.title = title
        self.startYear = startYear
        self.endYear = endYear
        self.downloadDir = downloadDir

        firefox_options = Options()
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference("useAutomationExtension", False)
        firefox_options.add_argument("-profile")
        firefox_options.add_argument(profilePath)

        firefox_options.set_preference("browser.download.folderList", 2)
        firefox_options.set_preference("browser.download.useDownloadDir", True)
        firefox_options.set_preference("browser.download.dir", downloadDir)
        firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        firefox_options.set_preference("pdfjs.disabled", False)
        firefox_options.set_preference('browser.download.manager.showWhenStarting', False)

        self.driver = webdriver.Firefox(options=firefox_options)
        self.driver.get("https://www.britishnewspaperarchive.co.uk/account/login")

        time.sleep(2)

        username_field = self.driver.find_element(By.NAME, "Username")
        password_field = self.driver.find_element(By.NAME, "Password")

        username_field.send_keys(os.environ.get("BNA_EMAIL")) # need to be set in environement variables
        password_field.send_keys(os.environ.get("BNA_PASSWORD")) # need to be set in environement variables
        password_field.send_keys(Keys.RETURN)

        time.sleep(2)

    def getIssues(self, startIndex=0, saveIndex=None):
        self.issues = []
        index = 0
        while True:
            url = f"https://www.britishnewspaperarchive.co.uk/search/results/{self.startYear}-01-01/{self.endYear}-12-31?retrievecountrycounts=false&newspapertitle={self.title}&sortorder=dayearly&page={index}"
            self.driver.get(url)
            elements = self.driver.find_elements(By.XPATH, '//div[@class="thumbnailContainer"]/a')
            for element in elements:
                href = element.get_attribute('href')
                self.issues.append(href)

            if len(elements) == 0: # Last page
                break
            
            index += 1
        
        if saveIndex is not None:
            f = open(saveIndex, "w")
            result = str(startIndex + index)
            f.write(result)
            f.close()

    def getPages(self, issuesPath=None, startIndex=0, saveIndex=None):
        if issuesPath is not None:
            f = open(issuesPath, 'r')
            issues = f.readlines()
            issues = [issue.rstrip() for issue in issues]
            self.issues = issues
            f.close()

        issues = self.issues[startIndex:]

        self.pages = []
        for issueIndex in range(len(issues)):
            issueUrl = issues[issueIndex]
            index = 0
            while True:
                url = issueUrl + "&page=" + str(index)
                self.driver.get(url)
                elements = self.driver.find_elements(By.XPATH, '//div[@class="thumbnailContainer"]/a')
                for element in elements:
                    href = element.get_attribute('href')
                    self.pages.append(href)

                if len(elements) == 0: # Last page
                    break

                index += 1
            
            if index == 0 : # The website wants the program to stop
                break
        
        if saveIndex is not None:
            f = open(saveIndex, "w")
            if issueIndex < len(issues) - 1:
                result = str(startIndex + issueIndex)
            else :
                result = "FINISHED"
            f.write(result)
            f.close()
    
    def getPdf(self, pagesPath=None, startIndex=0, saveIndex=None):
        prefix = "https://www.britishnewspaperarchive.co.uk/viewer/"

        if pagesPath is not None:
            f = open(pagesPath, 'r')
            pages = f.readlines()
            pages = [page.rstrip() for page in pages]
            self.pages = pages
            f.close()

        pages = self.pages[startIndex:]
        limit = 100 # 5800 per month # https://www.britishnewspaperarchive.co.uk/content/terms_and_conditions
        time_limit = 60
        refresh_time = 0.25

        waiting = None
        for pageIndex in range(min(len(pages), limit)):
            pageUrl = pages[pageIndex]
            downloadUrl = prefix + "download/" + pageUrl[len(prefix):]

            length = os.listdir(self.downloadDir)

            self.driver.execute_script(f'window.open("{downloadUrl}", "_blank");')

            waiting = 0
            while length == os.listdir(self.downloadDir) and waiting < time_limit:
                time.sleep(refresh_time)
                waiting += refresh_time
            time.sleep(2)
            
            if waiting >= time_limit:
                break

            self.driver.switch_to.window(self.driver.window_handles[-1])
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])

        if saveIndex is not None and waiting is not None:
            f = open(saveIndex, "w")
            if pageIndex < len(pages) - 1:
                if waiting >= time_limit:
                    result = str(startIndex + pageIndex)
                else:
                    result = str(startIndex + pageIndex + 1)
            else :
                result = "FINISHED"
            f.write(result)
            f.close()