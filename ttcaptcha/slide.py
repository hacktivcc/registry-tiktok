import time
import random
import httpx
import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
import logging , json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Solver:
    def __init__(self, did, iid, client):
        self.rf = Roboflow(api_key="HkhBOt8Vmh9aLQ9ETSbf")
        self.__host = "verification16-normal-c-useast2a.tiktokv.com"
        self.__device_id = did
        self.__install_id = iid
        self.__cookies = ""
        self.__client = client
        self.captcha_id = ""
        self.verify_id = ""
        self.image_shape = ""
        self.boxes_info = ""
        self.final_solver_result = ""
        self.first_piece = ""
        self.second_piece = ""
        logging.info("Solver initialized with device_id: %s and install_id: %s", did, iid)

    async def get_captcha(self):
        logging.info("Fetching captcha")
        params = self._build_captcha_params()
        headers = self._build_captcha_headers()

        response = await self.__client.get(
            url=f"https://{self.__host}/captcha/get", params=params, headers=headers
        )

        data = response.json()
        if data["code"] == 501:
            logging.error("Error fetching captcha: %s", data["message"])
            raise Exception(data["message"])
        self.captcha_id = data["data"]["id"]
        self.verify_id = data["data"]["verify_id"]
        url1 = data["data"]["question"]["url1"]
        url2 = data["data"]["question"]["url2"]
        logging.info("Captcha fetched with id: %s and verify_id: %s", self.captcha_id, self.verify_id)
        return url1, url2

    def _build_captcha_params(self):
        return {
            "lang": "en",
            "app_name": "musical_ly",
            "h5_sdk_version": "2.26.17",
            "sdk_version": "1.3.3-rc.7.3-bugfix",
            "iid": self.__install_id,
            "did": self.__device_id,
            "device_id": self.__device_id,
            "ch": "beta",
            "aid": "1233",
            "os_type": "0",
            "mode": "",
            "tmp": f"{int(time.time())}{random.randint(111, 999)}",
            "platform": "app",
            "webdriver": "false",
            "verify_host": f"https://{self.__host}/",
            "locale": "en",
            "channel": "beta",
            "app_key": "",
            "vc": "18.2.15",
            "app_version": "18.2.15",
            "session_id": "",
            "region": ["va", "US"],
            "use_native_report": "0",
            "use_jsb_request": "1",
            "orientation": "1",
            "resolution": ["900*1552", "900*1600"],
            "os_version": ["25", "7.1.2"],
            "device_brand": "samsung",
            "device_model": "SM-G973N",
            "os_name": "Android",
            "challenge_code": "1105",
            "subtype": "",
        }

    def _build_captcha_headers(self):
        return {
            "passport-sdk-version": "19",
            "sdk-version": "2",
            "x-ss-req-ticket": f"{int(time.time())}{random.randint(111, 999)}",
            "cookie": self.__cookies,
            "content-type": "application/json; charset=utf-8",
            "host": self.__host,
            "connection": "Keep-Alive",
            "user-agent": "okhttp/3.10.0.1",
        }

    async def process_image(self, url1, url2):
        logging.info("Processing images from URLs: %s, %s", url1, url2)
        gray_img = await self._download_and_convert_to_gray(url1)
        self._detect_boxes(gray_img)

        image2_gray = await self._download_and_convert_to_gray(url2)
        overlay_img = self._overlay_images(gray_img, image2_gray)

        resized_image = self._resize_image(overlay_img)
        self._detect_final_boxes(resized_image)

        self.final_solver_result = self.calculate_correct_position(
            self.second_piece, self.first_piece
        )
        logging.info("Final solver result calculated: %s", self.final_solver_result)
        self.visualize_with_rectangle(resized_image)

        await self._close_popup_if_open()

    async def _close_popup_if_open(self):
        logging.info("Checking for open popup and closing if necessary")
        
    async def _download_and_convert_to_gray(self, url):
        async with self.__client.stream("GET", url) as response:
            img_content = await response.aread()
        image = cv2.imdecode(np.frombuffer(img_content, dtype="uint8"), cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_boxes(self, gray_img):
        model = self.rf.workspace().project("secondproject-tzbxn").version("3").model
        result = model.predict(gray_img, confidence=30, overlap=40).json()

        self.boxes_info = [
            {
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"],
            }
            for pred in result["predictions"]
        ]

    def _overlay_images(self, gray_img, image2_gray):
        overlay_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        if self.boxes_info:
            y_start = max(0, int(self.boxes_info[0]["y"]) - 50)
            if y_start + image2_gray.shape[0] <= overlay_img.shape[0]:
                overlay_img[
                    y_start : y_start + image2_gray.shape[0], 0 : image2_gray.shape[1]
                ] = cv2.cvtColor(image2_gray, cv2.COLOR_GRAY2BGR)

        return overlay_img

    def _resize_image(self, overlay_img):
        target_width = 288
        original_height, original_width = overlay_img.shape[:2]
        scaling_factor = target_width / original_width
        new_height = int(original_height * scaling_factor)
        target_size = (target_width, new_height)
        return cv2.resize(overlay_img, target_size)

    def _detect_final_boxes(self, resized_image):
        rf2 = Roboflow(api_key="HksZuMSinDUSgLstje4R")
        project2 = rf2.workspace().project("thirdproject-9nxqc")
        model2 = project2.version("1").model

        final_result = model2.predict(resized_image, confidence=40, overlap=30).json()
        self.boxes_info = [
            {
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"],
                "name": pred["class"],
            }
            for pred in final_result["predictions"]
        ]

        for box in self.boxes_info:
            if box["name"] == "firstpiece":
                self.first_piece = box
            elif box["name"] == "secondpiece":
                self.second_piece = box

    def visualize_with_rectangle(self, image):
        x = int(self.final_solver_result["x"])
        y = int(self.final_solver_result["y"])
        width = int(self.boxes_info[1]["width"])
        height = int(self.boxes_info[1]["height"])

        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis("on")
        plt.show()
        plt.close()

    def calculate_correct_position(self, puzzle, piece):
        puzzle_center_x = puzzle["x"] + puzzle["width"] / 2
        puzzle_center_y = puzzle["y"] + puzzle["height"] / 2

        piece_center_x = piece["x"] + piece["width"] / 2
        piece_center_y = piece["y"] + piece["height"] / 2

        offset_x = puzzle_center_x - piece_center_x
        offset_y = puzzle_center_y - piece_center_y

        return {
            "x": piece["x"] + offset_x,
            "y": piece["y"] + offset_y,
        }

    async def __post_captcha(self) -> dict:
        reply_data = self._generate_reply_data()
        body = json.dumps(self._build_post_body(reply_data))
        headers = self._build_post_headers()

        url = self._build_post_url()
        req = await self.__client.post(url=url, headers=headers, json=body)
        logging.info("Response Text: %s", req.text)

        if req.status_code != 200:
            logging.error("Error in response: %s", req.text)
            raise Exception(f"Error in response: {req.text}")

        return req.json()
    def _generate_reply_data(self):
        reply_data = []
        relative_time = 121

        x_value = int(self.final_solver_result["x"])
        y_value = int(self.final_solver_result["y"])

        for i in range(10):
            reply_data.append(
                {"x": x_value + i, "y": y_value, "relative_time": relative_time + i * 10}
            )

        return reply_data

    def _build_post_body(self, reply_data):
        return {
            "modified_img_width": 288,
            "id": self.captcha_id,
            "mode": "slide",
            "reply": reply_data,
            "models": {},
            "log_params": {},
            "reply2": [],
            "models2": {},
            "drag_width": 288,
            "version": 2,
            "verify_id": self.verify_id,
            "verify_requests": [
                {
                    "id": self.captcha_id,
                    "modified_img_width": 288,
                    "drag_width": 288,
                    "mode": "slide",
                    "reply": reply_data,
                    "models": {},
                    "reply2": [],
                    "models2": {},
                    "events": '{"userMode":0}',
                },
            ],
            "events": '{"userMode":0}',
        }


    def _build_post_headers(self):
        return {
            "Host": "rc-verification-sg.tiktokv.com",
            "X-Tt-Request-Tag": "n=1;t=0",
            "X-Ss-Req-Ticket": f"{int(time.time())}{random.randint(111, 999)}",
            "X-Bd-Kmsv": "0",
            "X-Tt-Bypass-Dp": "1",
            "X-Tt-Store-Region": "ae",
            "X-Tt-Store-Region-Src": "uid",
            "User-Agent": "okhttp/3.10.0.1",
            "Content-Type": "application/json; charset=utf-8",
            "Cookie": f"install_id={self.__install_id}; tt-target-idc=alisg; store-idc=alisg",
        }

    def _build_post_url(self):
        return (
            f"https://rc-verification-sg.tiktokv.com/captcha/verify?"
            f"lang=en&app_name=musical_ly&h5_sdk_version=2.33.7&h5_sdk_use_type=cdn&"
            f"sdk_version=2.2.1.i18n&iid={self.__install_id}&did={self.__device_id}&"
            f"device_id={self.__device_id}&ch=googleplay&aid=1233&os_type=0&mode=slide&"
            f"tmp={int(time.time())}{random.randint(111, 999)}&platform=app&webdriver=false&verify_host=https%3A%2F%2Fverify-sg.tiktokv.com%2F&locale=en&channel=googleplay&app_key&vc=26.8.1&app_verison=26.8.1&session_id&region=sg&use_native_report=1&use_jsb_request=1&orientation=2&resolution=1080*1920&os_version=28&device_brand=HUAWEI&device_model=HPB-AN00&os_name=Android&version_code=2681&device_type=HPB-AN00&device_platform=Android&app_version=26.8.1&type=verify&detail=NiVvRQBeu0wRSq5pT4XGWgKdbJjjl9d4n5RV*UvyiJc2dmv8Xj3aZcHymF91lxc2C2oge67f0oYkW31BdXQj0mLX1p4zfHmyY7Zig7*YK3NLFYd656EkbsHIEFT6aKqUQtANK9nRk8niJFJMG9vomGxIzhfN*Kh3GQFmEwAdUu2vJrtt1sxuBPjpefNQtf*2P8qPtGXsvkDRF-eSkU5mGdJvPZuB1NxA7M*OP*wTySov9-TDAAXqkPBTk7tX4I5bVcVLYtMcWuBo0Rzkmu1-cXrXO4iaGI0fYsSkzKYrsnDeVnK50T3OeYpuamwpH3fR4AAGOBf40mG1eBOoP1-Q2wX-c4Hjp*8Dku77kE3fHpDzXdfhgDsj*kp*aAnDz06TEMfJciiuNiqJpnbXLonZhm-7gEjgyP--S9TNAQE0ldt8wWlnsmj4urO6XuYb&server_sdk_env=%7B%22idc%22%3A%22my%22%2C%22region%22%3A%22ALISG%22%2C%22server_type%22%3A%22passport%22%7D&imagex_domain&subtype=slide&challenge_code=99999&triggered_region=sg&cookie_enabled=true&screen_width=360&screen_height=640&browser_language=en-US&browser_platform=Linux%20i686&browser_name=Mozilla&browser_version=5.0%20%28Linux%3B%20Android%209%3B%20HPB-AN00%20Build%2FPQ3B.190801.06281541%3B%20wv%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Version%2F4.0%20Chrome%2F91.0.4472.114%20Mobile%20Safari%2F537.36%20BytedanceWebview%2Fd8a21c6"
        )

    async def solve_captcha(self):
        logging.info("Starting captcha solving process")
        url1, url2 = await self.get_captcha()
        await self.process_image(url1, url2)
        await self.__post_captcha()
        logging.info("Captcha solving process completed")

if __name__ == "__main__":
    async def main():
        async with httpx.AsyncClient() as client:
            solver = Solver("7383996999998590130", "7383997454771308293", client)
            await solver.solve_captcha()

    import asyncio
    asyncio.run(main())