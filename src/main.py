import logging
from playwright.async_api import async_playwright
from asyncio import sleep
from tm import TempMailService
from utils import CaptchaSolver, trim_image
from httpx import AsyncClient
from random import randint
from faker import Faker

logging.basicConfig(level=logging.INFO)

class KakaoRegistration:

    def __init__(self):
        self.tiktok_signup = "https://www.tiktok.com/signup"
        self.registry_api = None
        self.browser = None
        self.context = None
        self.page = None
        self.email = None
        self.nickname = None
        self.client = AsyncClient(http2=True)
        self.temp_mail = TempMailService(self.client)
        self.password = f"{Faker().password(length=16, special_chars=True, digits=True, upper_case=True, lower_case=True)}"

    async def launch_browser(self):
        async with async_playwright() as p:
            self.browser = await p.firefox.launch(headless=False)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            await self.tiktok_registry()

    async def tiktok_registry(self):
        await self.page.goto(self.tiktok_signup, wait_until="load")
        await self.page.wait_for_selector('div[data-e2e="channel-item"]', state="visible")
        await self.page.click('div[data-e2e="channel-item"]:has-text("Continue with KakaoTalk")')
        new_page = await self.context.wait_for_event("page")
        await new_page.wait_for_load_state()
        await self.kakaotalk_register(new_page)

    async def kakaotalk_register(self, page):
        try:
            self.email = await self.temp_mail.get_email()
            page.on("response", lambda response: self.handler_response(response, page.frame_locator("iframe"), page))
            logging.info(f"Trying to register with: {self.email}")

            await page.wait_for_selector('a.link_join', state="visible", timeout=10000)
            await page.click('a.link_join')

            await page.wait_for_selector("button.btn_g.highlight.submit:has-text('I have an email account.')", state="visible", timeout=10000)
            await page.click("button.btn_g.highlight.submit:has-text('I have an email account.')")

            await page.wait_for_selector(".ico_comm.ico_check", state="visible")
            await page.click(".ico_comm.ico_check")
            await page.click(".submit")
            await page.fill("input[name=email]", self.email)
            await page.click("button.btn_round")

            await self.handle_verification(page)
            self.nickname = Faker().name()
            await page.fill("input[name=profile_nickname]", self.nickname)
            await self.select_birthdate(page, f"{randint(1980, 2005)}", f"{randint(1, 12)}", f"{randint(1, 29)}")
            await self.select_gender(page, "male")
            await page.click(".submit")
            with open("accounts_registred/accounts_registred.txt", "a") as file:
                file.write(f"Email: {self.email}\nNickname: {self.nickname}\nPassword: {self.password}\n")
            logging.info(f"Account registered successfully: {self.email}, {self.nickname}, {self.password}")

            await page.click("text='Get Started'")

            await sleep(10)
            await page.click("text='Accept All'")
            await sleep(5)
            await page.click("#acceptButton")
            await sleep(100)

        except Exception as e:
            logging.error(f"Registration error: {e}")
        finally:
            await self.close_browser()

    async def handle_verification(self, page):
        code_verify = None
        while code_verify == None:
            await sleep(10)
            logging.info("Waiting for verification code...")
            code_verify = await self.temp_mail.get_messages()

        logging.info(f"Verification code received: {code_verify}")
        await page.wait_for_selector("input[name=email_passcode]", state="visible")
        await page.fill("input[name=email_passcode]", code_verify)
        await page.click(".submit")

        await page.fill('input[name="new_password"]', self.password)
        await page.fill('input[name="password_confirm"]', self.password)
        await page.click(".submit")

    async def handle_captcha(self, cframe, page):
        try:
            await sleep(10)
            await page.screenshot(path="images/registry_captcha.png")
            trim_image()
            captcha_solver = CaptchaSolver()
            await captcha_solver.solve_captcha()
            logging.info(f"Captcha solved: {captcha_solver.answer}")

            await cframe.locator("#inpDkaptcha").wait_for(state="visible", timeout=60000)
            await cframe.locator("#inpDkaptcha").fill(captcha_solver.answer)

            await cframe.locator("#btn_dkaptcha_submit").click()
            logging.info("Captcha solved and submitted.")
        except Exception as e:
            logging.error(f"Captcha error: {e}")

    async def handler_response(self, response, cframe, page):
        if page.is_closed():
            return
        try:
            response_text = await response.text()
        except UnicodeDecodeError:
            response_text = (await response.body()).decode("latin1")

        if "/dkaptcha/quiz/" in response.url or "/dkaptcha/quiz" in response.url:
            if 'Enter the name of <em class="emph_txt">the place</em>' in response_text:
                await self.handle_captcha(cframe, page)
            else:
                await cframe.locator("#btn_dkaptcha_reset").click()
        elif "Bad Request" in response_text:
            logging.warning("Bad Request encountered. Resetting captcha.")
            try:
                await cframe.locator("#btn_dkaptcha_reset").click()
            except Exception as e:
                logging.error(f"Failed to reset captcha: {e}")

    async def select_birthdate(self, page, year, month, day):
        try:
            await page.wait_for_selector('#birthday-year--15', state="visible", timeout=5000)
            await page.select_option('#birthday-year--15', str(year))
            logging.info(f"Selected year: {year}")

            await page.wait_for_selector('#birthday-month--16', state="visible", timeout=5000)
            await page.select_option('#birthday-month--16', str(month))
            logging.info(f"Selected month: {month}")

            await page.wait_for_selector('#birthday-day--17', state="visible", timeout=5000)
            await page.select_option('#birthday-day--17', str(day))
            logging.info(f"Selected day: {day}")

        except Exception as e:
            logging.error(f"Error selecting birthdate: {e}")

    async def select_gender(self, page, gender):
        try:
            await page.click(f'label[for="radio-gender-{gender.lower()}"]')
            logging.info(f"Selected gender: {gender}")
        except Exception as e:
            logging.error(f"Failed to select gender: {e}")

    async def close_browser(self):
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.client:
            await self.client.aclose()