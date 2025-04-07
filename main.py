from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

from pathlib import Path
import os
import re
import aiohttp
import glob
import logging
from typing import List

from .utils.config_manager import load_config
from .utils.downloader import Downloader
from .utils.helpers import Helpers
from .utils.html_parser import HTMLParser
from .utils.message_adapter import MessageAdapter
from .utils.pdf_generator import PDFGenerator


@register("ehentai_bot", "drdon1234", "适配 AstrBot 的 EHentai画廊 转 PDF 插件", "2.0")
class EHentaiBot(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.config = load_config()
        self.uploader = MessageAdapter(self.config)
        self.helpers = Helpers()
        self.parser = HTMLParser()
        self.downloader = Downloader(self.config, self.uploader, self.parser, self.helpers)
        self.pdf_generator = PDFGenerator(self.config, self.helpers)

    @staticmethod
    def parse_command(message: str) -> List[str]:
        return [p for p in message.split(' ') if p][1:]
        
    @filter.command("搜eh")
    async def search_gallery(self, event: AstrMessageEvent):
        defaults = {
            "min_rating": 2,
            "min_pages": 1,
            "target_page": 1
        }
        
        try:
            cleaned_text = re.sub(r'@\S+\s*', '', event.message_str).strip()
            args = self.parse_command(cleaned_text)
            if not args:
                await self.eh_helper(event)
                return
                
            if len(args) > 4:
                await event.send(event.plain_result("参数过多，最多支持4个参数：标签 评分 页数 页码"))
                return
                
            tags = re.sub(r'[，,+]+', ' ', args[0])
            
            params = defaults.copy()
            param_names = ["min_rating", "min_pages", "target_page"]
            
            for i, (name, value) in enumerate(zip(param_names, args[1:]), 1):
                try:
                    params[name] = int(value)
                except ValueError:
                    await event.send(event.plain_result(f"第{i + 1}个参数应为整数: {value}"))
                    return
            
            await event.send(event.plain_result("正在搜索，请稍候..."))
            
            search_results = await self.downloader.crawl_ehentai(
                tags, 
                params["min_rating"], 
                params["min_pages"], 
                params["target_page"]
            )
            
            if not search_results:
                await event.send(event.plain_result("未找到符合条件的结果"))
                return
    
            results_ui = self.helpers.get_search_results(search_results)
            print(results_ui)
            await event.send(event.plain_result(results_ui))
        
        except ValueError as e:
            logger.exception("参数解析失败")
            await event.send(event.plain_result(f"参数错误：{str(e)}"))
            
        except Exception as e:
            logger.exception("搜索失败")
            await event.send(event.plain_result(f"搜索失败：{str(e)}"))
    
    @filter.command("看eh")
    async def download_gallery(self, event: AstrMessageEvent):
        cleaned_text = re.sub(r'@\S+\s*', '', event.message_str).strip()

        image_folder = Path(self.config['output']['image_folder'])
        pdf_folder = Path(self.config['output']['pdf_folder'])

        if not image_folder.exists():
            image_folder.mkdir(parents=True)
        if not pdf_folder.exists():
            pdf_folder.mkdir(parents=True)

        for f in glob.glob(str(Path(self.config['output']['image_folder']) / "*.*")):
            os.remove(f)

        try:
            args = self.parse_command(cleaned_text)
            if len(args) != 1:
                self.eh_helper(event)
                return

            pattern = re.compile(r'^https://(e-hentai|exhentai)\.org/g/\d{7}/[a-f0-9]{10}/$')
            if not pattern.match(args):
                await event.send(event.plain_result("画廊链接异常，请重试..."))
                return

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                is_pdf_exist = await self.downloader.process_pagination(event, session, args)

                if not is_pdf_exist:
                    title = self.downloader.gallery_title
                    await self.pdf_generator.merge_images_to_pdf(event, title)
                    await self.uploader.upload_file(event, self.config['output']['pdf_folder'], title)

        except Exception as e:
            logger.exception("下载失败")
            await event.send(event.plain_result(f"下载失败：{str(e)}"))

    @filter.command("eh")
    async def eh_helper(self, event: AstrMessageEvent):
        help_text = """eh指令帮助：
[1] 搜索画廊: 搜eh [关键词] [最低评分（2-5，默认2）] [最少页数（默认1）] [获取第几页的画廊列表（默认1）]
[2] 下载画廊: 看eh [画廊链接]
[3] 获取指令帮助: eh
[4] 热重载config相关参数: 重载eh配置

可用的搜索方式:
[1] 搜eh [关键词]
[2] 搜eh [关键词] [最低评分]
[3] 搜eh [关键词] [最低评分] [最少页数]
[4] 搜eh [关键词] [最低评分] [最少页数] [获取第几页的画廊列表]"""
        await event.send(event.plain_result(help_text))

    @filter.command("重载eh配置")
    async def reload_config(self, event: AstrMessageEvent):
        await event.send(event.plain_result("正在重载配置参数"))
        self.config = load_config()
        self.uploader = MessageAdapter(self.config)
        self.downloader = Downloader(self.config, self.uploader, self.parser, self.helpers)
        self.pdf_generator = PDFGenerator(self.config, self.helpers)
        await event.send(event.plain_result("已重载配置参数"))

    async def terminate(self):
        pass
