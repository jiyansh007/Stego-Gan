import unittest
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class StegoGANTests(unittest.TestCase):
    def setUp(self):
        options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(options=options)
        self.driver.get("http://localhost:8501")
        time.sleep(3)

    def tearDown(self):
        if self.driver:
            self.driver.quit()

    def test_pass_page_title(self):
        """Test that the page title is correct (Will Pass)"""
        self.assertIn("Stego-GAN", self.driver.title, "Page title should contain Stego-GAN")

    def test_pass_login_text_present(self):
        """Test that the login page renders perfectly (Will Pass)"""
        page_source = self.driver.page_source
        self.assertIn("Stego-GAN Login", page_source, "Login text should be on the initial login page")

    def test_failed_login(self):
        """Test that entering wrong credentials gives an error message"""
        inputs = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-testid='stTextInput'] input"))
        )
        self.assertEqual(len(inputs), 2, "Expected 2 text inputs for login form")
        
        inputs[0].send_keys("wronguser")
        inputs[1].send_keys("wrongpass")
        
        button = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='stFormSubmitButton'] button")
        button.click()
        time.sleep(2)  
        
        alerts = self.driver.find_elements(By.CSS_SELECTOR, "div[data-testid='stAlert']")
        self.assertTrue(len(alerts) > 0, "Expected an alert notification for failed login")
        self.assertIn("Invalid username or password", alerts[0].text, "Expected invalid login error message")

    def test_successful_login(self):
        """Test successful login and verify the app renders correctly"""
        inputs = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-testid='stTextInput'] input"))
        )
        self.assertEqual(len(inputs), 2, "Expected 2 text inputs for login form")
        
        inputs[0].send_keys("user")
        inputs[1].send_keys("user123")
        
        button = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='stFormSubmitButton'] button")
        button.click()
        time.sleep(3)  

        page_source = self.driver.page_source
        self.assertIn("Logged in as:", page_source, "Failed to find 'Logged in as:' indicating successful login")
        self.assertIn("Upload an image to analyse", page_source, "Failed to find the main application UI")

    def test_upload_image(self):
        """Test uploading an image after logging in"""
        inputs = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-testid='stTextInput'] input"))
        )
        self.assertEqual(len(inputs), 2, "Expected 2 text inputs for login form")
        inputs[0].send_keys("user")
        inputs[1].send_keys("user123")
        button = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='stFormSubmitButton'] button")
        button.click()
        time.sleep(3)  

        file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        abs_path = os.path.abspath(os.path.join("data", "out", "clean_img.jpg"))
        file_input.send_keys(abs_path)
        
        time.sleep(10)  

        page_source = self.driver.page_source
        self.assertTrue("CLEAN" in page_source or "STEGO" in page_source, "Image processing verdict not found on page after upload")
        self.assertIn("Original Image", page_source, "Expected to see Original Image section")
        
    def test_sanitize_image(self):
        """Test sanitizing an image after uploading it"""
        inputs = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-testid='stTextInput'] input"))
        )
        self.assertEqual(len(inputs), 2, "Expected 2 text inputs for login form")
        inputs[0].send_keys("user")
        inputs[1].send_keys("user123")
        button = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='stFormSubmitButton'] button")
        button.click()
        time.sleep(3)  

        file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        abs_path = os.path.abspath(os.path.join("data", "out", "clean_img.jpg"))
        file_input.send_keys(abs_path)
        
        time.sleep(10) 

        buttons = self.driver.find_elements(By.CSS_SELECTOR, "button")
        sanitize_button = None
        for b in buttons:
            if "Sanitize Image" in b.text:
                sanitize_button = b
                break
                
        self.assertIsNotNone(sanitize_button, "Could not find 'Sanitize Image' button")
        
        self.driver.execute_script("arguments[0].click();", sanitize_button)
        time.sleep(10)
        
        page_source = self.driver.page_source
        self.assertIn("Purification complete", page_source, "Sanitization string not found in page source after clicking Sanitize")

if __name__ == "__main__":
    print("Running Selenium UI Tests against Stego-GAN...")
    unittest.main()
