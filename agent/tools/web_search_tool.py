# tools/web_search_tool.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from utils.logger import logger
import time

class WebSearchTool:

    @staticmethod
    def get_driver():
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        try:
            driver = webdriver.Chrome(options=options)
            return driver
        except Exception as e:
            logger.error(f"[WebSearchTool] Error creating Chrome driver: {e}")
            return None

    @staticmethod
    def google_search(query):
        driver = WebSearchTool.get_driver()
        if driver is None:
            return {"success": False, "error": "Unable to start browser"}

        results = []

        try:
            driver.get(f"https://www.google.com/search?q={query}")
            time.sleep(2)

            items = driver.find_elements(By.CSS_SELECTOR, "div.g")[:5]

            for item in items:
                title_el = item.find_elements(By.TAG_NAME, "h3")
                link_el  = item.find_elements(By.TAG_NAME, "a")

                if title_el and link_el:
                    results.append({
                        "title": title_el[0].text,
                        "url": link_el[0].get_attribute("href")
                    })

        except Exception as e:
            logger.error(f"[WebSearchTool] Scraping error: {e}")
            return {"success": False, "error": str(e)}

        finally:
            driver.quit()

        return {"success": True, "results": results}
