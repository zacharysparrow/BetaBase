import os
import re
import sys
from datetime import datetime

from httpx import Client
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)
from user_agent import generate_user_agent

from .js_decoder import deobfuscator

console = Console()


class SnapInsta:
    @staticmethod
    def _get_token(client):
        response = client.get("https://snapinst.app/")
        response.raise_for_status()
        return (
            (matches := re.search(r'name="token" value="(.*?)"', response.text))
            and matches[1]
            or None
        )

    @staticmethod
    def _get_variable(client, insta_url, token):
        response = client.post(
            "https://snapinst.app/action2.php",
            headers={"Referer": "https://snapinst.app/"},
            data={
                "url": insta_url,
                "token": token,
            },
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def _extract_variables(js_code):
        pattern = r'\("(\w+)",\d+,"(\w+)",(\d+),(\d+),\d+\)'
        return [
            int(x) if x.isdigit() else x for x in re.search(pattern, js_code).groups()
        ]

    @staticmethod
    def _deobfuscate_html(js_variables):
        return deobfuscator(*js_variables)

    @staticmethod
    def _match_pattern(pattern, html_source):
#        matches = re.search(pattern, html_source)
        matches = re.findall(pattern, html_source)
        return matches if matches else None

    @staticmethod
    def _contains_string(video_link, substring):
        return video_link if substring in video_link else None

    @classmethod
    def _extract_snapinsta_link(cls, html_source):
        pattern = r'href=\\"([^\\"]+)\\"'
        matched_link = cls._match_pattern(pattern, html_source)
        if matched_link:
            matched_links = []
            for link in matched_link:
                matched_links.append(cls._contains_string(link, "snapinsta"))
            return(matched_links)
#        return cls._contains_string(matched_link, "snapinsta") if matched_link else None

    @staticmethod
    def _video_downloader(client, video_link, uuid):
        with client.stream("GET", video_link) as response:
            response.raise_for_status()

            video_folder = "videos"
            os.makedirs(video_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%d%m%y_%I%M%p")
#            filename = f"{video_folder}/snapinst.app_{timestamp}.mp4"
            filename = f"{video_folder}/snapinst.app_{uuid}.mp4"

            total_size = int(response.headers.get("Content-Length", 0))
            with Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TextColumn("•"),
                TransferSpeedColumn(),
            ) as progress:
                task = progress.add_task("Downloading..", total=total_size)

                with open(filename, "wb") as file:
                    for data in response.iter_bytes():
                        file.write(data)
                        progress.update(task, advance=len(data))

            console.print(f"\nVideo has been saved in [medium_purple3]{filename}[/]\n")

    @classmethod
    def start_download(cls, insta_url, uuid):
#        console.clear()
        print("")
        with Client(
            headers={"User-Agent": generate_user_agent()},
            timeout=10, #ZMS change timeout time here
        ) as client:

            token = cls._get_token(client)
            js_code = cls._get_variable(client, insta_url, token)
            js_variables = cls._extract_variables(js_code)
            html_source = cls._deobfuscate_html(js_variables)
            snapinsta_links = cls._extract_snapinsta_link(html_source)
            vid_num = 0
            for current_link in snapinsta_links:
                if video_link := current_link:
                    vid_num += 1
                    print([uuid, vid_num])
                    cls._video_downloader(client, video_link, uuid+"_vid"+str(vid_num))
                else:
                    raise ValueError("Download url is not found!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        console.print("[bold]Usage[/]: python -m snapinsta 'insta_video_url_here' 'uuid_here'")
        sys.exit(1)
    video_url = sys.argv[1]
    uuid = sys.argv[2]
    SnapInsta.start_download(video_url, uuid)
