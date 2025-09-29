from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Image, Plain
from .utils.config_manager import load_config
from .utils.downloader import Downloader
from .utils.html_parser import HTMLParser
from .utils.message_adapter import MessageAdapter
from pathlib import Path
import os
import re
import aiohttp
import glob
import logging
import json
import io
import tempfile
from typing import List
from PIL import Image as PILImage, ImageDraw

logger = logging.getLogger(__name__)


@register("astrbot_plugin_ehentai_bot", "drdon1234", "适配 AstrBot 的 EHentai画廊 转 PDF 插件", "3.0")
class EHentaiBot(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = load_config(config)
        self.parser = HTMLParser()
        self.uploader = MessageAdapter(self.config)
        self.downloader = Downloader(self.config, self.uploader, self.parser)

    @staticmethod
    def parse_command(message: str) -> List[str]:
        cleaned_text = re.sub(r'@\S+\s*', '', message).strip()
        return [p for p in cleaned_text.split(' ') if p][1:]
        
    async def download_thumbnail(self, url):
        """下载封面图片"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://e-hentai.org/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    # 将图片数据转换为PIL Image对象
                    return PILImage.open(io.BytesIO(await response.read()))
        except Exception as e:
            logger.error(f"下载封面图片失败：{e}")
            return None

    def create_combined_image(self, images):
        """将多个封面图片拼接成一张图片"""
        if not images:
            return None

        # 过滤掉None值
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return None

        # 设置统一的目标高度
        target_height = 800  # 统一高度
        padding = 10  # 图片间距
        
        # 计算每张图片按比例缩放后的宽度
        scaled_widths = []
        for img in valid_images:
            # 获取原始尺寸
            width, height = img.size
            # 按高度等比例缩放
            scaled_width = int((width * target_height) / height)
            scaled_widths.append(scaled_width)
        
        # 计算总宽度（包含间距）
        total_width = sum(scaled_widths) + (len(valid_images) - 1) * padding
        
        # 创建新图片
        combined_image = PILImage.new('RGB', (total_width, target_height), (255, 255, 255))

        # 拼接图片
        x_offset = 0
        for img, scaled_width in zip(valid_images, scaled_widths):
            # 调整图片大小，保持比例
            img = img.convert('RGB')
            img = img.resize((scaled_width, target_height), PILImage.Resampling.LANCZOS)
            
            # 粘贴到新图片上
            combined_image.paste(img, (x_offset, 0))
            x_offset += scaled_width + padding

        # 添加随机色块以规避图片审查
        self.add_random_blocks(combined_image)

        # 保存到临时文件
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'combined_covers.jpg')
        combined_image.save(temp_path, 'JPEG')
        return temp_path

    def add_random_blocks(self, image):
        """添加随机色块以规避图片审查"""
        import random
        
        width, height = image.size
        
        # 添加10-20个随机色块
        num_blocks = random.randint(10, 20)
        
        for _ in range(num_blocks):
            # 随机位置
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            
            # 随机大小（较小，不影响观看）
            block_width = random.randint(3, 8)
            block_height = random.randint(3, 8)
            
            # 确保色块不超出图片边界
            x2 = min(x1 + block_width, width - 1)
            y2 = min(y1 + block_height, height - 1)
            
            # 随机颜色（半透明）
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            alpha = random.randint(30, 100)  # 透明度
            
            # 获取原始像素
            for x in range(x1, x2):
                for y in range(y1, y2):
                    if 0 <= x < width and 0 <= y < height:
                        # 获取当前像素颜色
                        current = image.getpixel((x, y))
                        
                        # 混合颜色（考虑透明度）
                        new_r = int((current[0] * (255 - alpha) + r * alpha) / 255)
                        new_g = int((current[1] * (255 - alpha) + g * alpha) / 255)
                        new_b = int((current[2] * (255 - alpha) + b * alpha) / 255)
                        
                        # 设置新颜色
                        image.putpixel((x, y), (new_r, new_g, new_b))

    @filter.command("搜eh")
    async def search_gallery(self, event: AstrMessageEvent):
        defaults = {
            "min_rating": 2,
            "min_pages": 1,
            "target_page": 1
        }

        try:
            args = self.parse_command(event.message_str)
            if not args:
                await self.eh_helper(event)
                return

            if len(args) > 4:
                await event.send(event.plain_result("参数过多，最多支持4个参数：标签 评分 页数 页码"))
                return

            raw_tags = args[0]
            tags = re.sub(r'[，,+]+', ' ', args[0])

            params = defaults.copy()
            params["tags"] = raw_tags
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

            cache_data = {"params": params}
            for idx, result in enumerate(search_results, 1):
                cache_data[str(idx)] = result['gallery_url']

            search_cache_folder = Path(self.config['output']['search_cache_folder'])
            search_cache_folder.mkdir(exist_ok=True, parents=True)

            cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            # 下载所有封面图片
            covers = []
            for result in search_results:
                cover_url = result['cover_url']
                cover = await self.download_thumbnail(cover_url)
                covers.append(cover)

            # 创建拼接图片
            combined_image_path = self.create_combined_image(covers)
            
            # 构建消息内容
            message_components = []
            
            # 添加拼接后的封面图片
            if combined_image_path:
                message_components.append(Image(combined_image_path))
            
            # 添加文字内容
            output = "搜索结果:\n"
            output += "=" * 50 + "\n"
            for idx, result in enumerate(search_results, 1):
                output += f"[{idx}] {result['title']}\n"
                output += (
                    f" 作者: {result['author']} | 分类: {result['category']} | 页数: {result['pages']} | "
                    f"评分: {result['rating']} | 上传时间: {result['timestamp']}\n"
                )
                output += f" 画廊链接: {result['gallery_url']}\n"
                output += "-" * 80 + "\n"

            message_components.append(Plain(output))
            
            # 发送消息
            await event.send(MessageEventResult(message_components))

        except ValueError as e:
            logger.exception("参数解析失败")
            await event.send(event.plain_result(f"参数错误：{str(e)}"))

        except Exception as e:
            logger.exception("搜索失败")
            await event.send(event.plain_result(f"搜索失败：{str(e)}"))

    @filter.command("eh翻页")
    async def jump_to_page(self, event: AstrMessageEvent):
        args = self.parse_command(event.message_str)
        if len(args) != 1:
            await event.send(event.plain_result("参数错误，翻页操作只需要一个参数（页码）"))
            return
    
        page_num = args[0]
        if not page_num.isdigit() or int(page_num) < 1:
            await event.send(event.plain_result("页码应该是大于0的整数"))
            return
    
        search_cache_folder = Path(self.config['output']['search_cache_folder'])
        cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
    
        if not cache_file.exists():
            await event.send(event.plain_result("未找到搜索记录，请先使用'搜eh'命令"))
            return
    
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    
        if 'params' not in cache_data:
            await event.send(event.plain_result("缓存文件中缺少必要参数信息，请使用'搜eh'命令重新搜索"))
            return
    
        params = cache_data['params']
        
        if 'tags' not in params:
            await event.send(event.plain_result("缓存文件中未找到关键词信息，无法跳转到指定页"))
            return
    
        params['target_page'] = int(page_num)
        event.message_str = f"搜eh {params['tags']} {params['min_rating']} {params['min_pages']} {params['target_page']}"
    
        await self.search_gallery(event)
        
    @filter.command("看eh")
    async def download_gallery(self, event: AstrMessageEvent, cleaned_text: str):
        image_folder = Path(self.config['output']['image_folder'])
        image_folder.mkdir(exist_ok=True, parents=True)
        pdf_folder = Path(self.config['output']['pdf_folder'])
        pdf_folder.mkdir(exist_ok=True, parents=True)
        search_cache_folder = Path(self.config['output']['search_cache_folder'])
        search_cache_folder.mkdir(exist_ok=True, parents=True)

        for f in glob.glob(str(Path(self.config['output']['image_folder']) / "*.*")):
            os.remove(f)

        try:
            args = self.parse_command(event.message_str)
            if len(args) != 1:
                await self.eh_helper(event)
                return

            url = args[0]
            pattern = re.compile(r'^https://(e-hentai|exhentai)\.org/g/\d{7}/[a-f0-9]{10}/$')

            if not pattern.match(url):
                if url.isdigit() and int(url) > 0:
                    cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
                    if cache_file.exists():
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)

                        if url in cache_data:
                            url = cache_data[url]
                            await event.send(event.plain_result(f"正在获取画廊链接: {url}"))
                        else:
                            await event.send(event.plain_result(f"未找到索引为 {url} 的画廊"))
                            return
                    else:
                        await event.send(event.plain_result(f"未找到搜索记录，请先使用'搜eh'命令"))
                        return
                else:
                    await event.send(event.plain_result(f"画廊链接异常，请重试..."))
                    return

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                is_pdf_exist = await self.downloader.process_pagination(event, session, url)

                if not is_pdf_exist:
                    title = self.downloader.gallery_title
                    safe_title = await self.downloader.merge_images_to_pdf(event, title)
                    await self.uploader.upload_file(event, self.config['output']['pdf_folder'], safe_title)

        except Exception as e:
            logger.exception("下载失败")
            await event.send(event.plain_result(f"下载失败：{str(e)}"))

    @filter.command("归档eh")
    async def archive_gallery(self, event: AstrMessageEvent):
        search_cache_folder = Path(self.config['output']['search_cache_folder'])
        search_cache_folder.mkdir(exist_ok=True, parents=True)

        try:
            args = self.parse_command(event.message_str)
            if len(args) != 1:
                await event.send(event.plain_result("参数错误，归档操作只需要一个参数（画廊链接或搜索结果序号）"))
                return

            url = args[0]
            pattern = re.compile(r'^https://(e-hentai|exhentai)\.org/g/(\d{7})/([a-f0-9]{10})/?$')

            match = pattern.match(url)
            if not match:
                if url.isdigit() and int(url) > 0:
                    cache_file = search_cache_folder / f"{event.get_sender_id()}.json"
                    if cache_file.exists():
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)

                        if url in cache_data:
                            url = cache_data[url]
                            await event.send(event.plain_result(f"正在获取画廊链接: {url}"))
                        else:
                            await event.send(event.plain_result(f"未找到索引为 {url} 的画廊"))
                            return
                    else:
                        await event.send(event.plain_result(f"未找到搜索记录，请先使用'搜eh'命令"))
                        return
                else:
                    await event.send(event.plain_result(f"画廊链接异常，请重试..."))
                    return

            # 重新匹配以获取 gid 和 token
            match = pattern.match(url)
            if not match:
                await event.send(event.plain_result(f"无法解析画廊链接，请重试..."))
                return

            _, gid, token = match.groups()
            
            await event.send(event.plain_result("正在获取归档链接，请稍候..."))
            
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                download_url = await self.downloader.get_archive_url(session, gid, token)
                
                if download_url:
                    await event.send(event.plain_result(f"归档链接获取成功，请尽快下载（链接仅能访问一次）：\n{download_url}"))
                else:
                    await event.send(event.plain_result("归档链接获取失败，请检查账号权限或重试"))

        except Exception as e:
            logger.exception("归档失败")
            await event.send(event.plain_result(f"归档失败：{str(e)}"))

    @filter.command("eh")
    async def eh_helper(self, event: AstrMessageEvent):
        help_text = """eh指令帮助：
[1] 搜索画廊: 搜eh [关键词] [最低评分（2-5，默认2）] [最少页数（默认1）] [获取第几页的画廊列表（默认1）]
[2] 快速翻页: eh翻页 [获取第几页的画廊列表]
[3] 下载画廊: 看eh [画廊链接/搜索结果序号]
[4] 获取归档链接: 归档eh [画廊链接/搜索结果序号]
[5] 获取指令帮助: eh
[6] 热重载config相关参数: 重载eh配置

可用的搜索方式:
[1] 搜eh [关键词]
[2] 搜eh [关键词] [最低评分]
[3] 搜eh [关键词] [最低评分] [最少页数]
[4] 搜eh [关键词] [最低评分] [最少页数] [获取第几页的画廊列表]
[5] eh翻页 [获取第几页的画廊列表]

可用的下载方式：
[1] 看eh [画廊链接]
[2] 看eh [搜索结果序号]

可用的归档方式：
[1] 归档eh [画廊链接]
[2] 归档eh [搜索结果序号]

注意：
[1] 搜索多关键词时请用以下符号连接`,` `，` `+`，关键词之间不要添加任何空格
[2] 使用"eh翻页 [获取第几页的画廊列表]"、"看eh [搜索结果序号]"和"归档eh [搜索结果序号]"前确保你最近至少使用过一次"搜eh"命令（每个用户的缓存文件是独立的）
[3] 归档链接仅能访问一次，请尽快下载"""
        await event.send(event.plain_result(help_text))

    @filter.command("重载eh配置")
    async def reload_config(self, event: AstrMessageEvent):
        await event.send(event.plain_result("正在重载配置参数"))
        self.config = load_config()
        self.uploader = MessageAdapter(self.config)
        self.downloader = Downloader(self.config, self.uploader, self.parser)
        await event.send(event.plain_result("已重载配置参数"))
    
    @filter.regex(r"^(?:\[([^\]]+)\]|\(([^\)]+)\))\s*(.*)$")
    async def search_by_formatted_message(self, event: AstrMessageEvent):
        """
        监听特定格式的消息，自动提取作者名和作品名，并拼接为搜索关键词进行搜索。
        """
        match = re.search(r"^(?:\[([^\]]+)\]|\(([^\)]+)\))\s*(.*)$", event.message_str)
        if not match:
            return MessageEventResult.IGNORED
            
        author = match.group(1) if match.group(1) else match.group(2)
        title = match.group(3).strip()

        # 移除作品名中可能存在的额外信息，例如[中国翻訳] (艦隊これくしょん -艦これ-)
        title = re.sub(r'\[[^\]]+\]|\([^\)]+\)', '', title).strip()

        if not author or not title:
            logger.warning(f"未能从消息中提取有效的作者或作品名: {event.message_str}")
            return MessageEventResult.IGNORED

        # 将空格替换为+
        search_query = f"{author.replace(' ', '+')}+{title.replace(' ', '+')}"
        
        # 构造一个新的 AstrMessageEvent 来调用 search_gallery
        # 注意：这里我们直接修改了 event.message_str，因为 search_gallery 会解析它
        event.message_str = f"搜eh {search_query}"
        
        logger.info(f"检测到格式化消息，将自动搜索: {search_query}")
        #await event.send(event.plain_result(f"检测到格式化消息，正在为您搜索: {search_query}"))
        await self.search_gallery(event)
        
        return
        
    async def terminate(self):
        pass
