# 适配 AstrBot 的 EHentai画廊 转 PDF 插件

## 安装方法

1. **通过 插件市场 安装**  
- 打开 "AstrBot WebUI" -> "插件市场" -> "右上角 Search"  
- 搜索任何与本项目相关的关键词，找到插件后点击安装
- 推荐通过唯一标识符搜索：```astrbot_plugin_ehentai_bot```

2. **通过 Github仓库链接 安装**  
- 打开 "AstrBot WebUI" -> "插件市场" -> "右下角 '+' 按钮"  
- 输入以下地址并点击安装：
```
https://github.com/drdon1234/astrbot_plugin_ehentai_bot
```

---

## 使用说明

### 指令帮助

- **搜索画廊**  
```搜eh [关键词] [最低评分（2-5，默认2）] [最少页数（默认1）] [获取第几页的画廊列表（默认1）]```

- **下载画廊**  
```看eh [画廊链接/搜索结果序号]```

- **获取指令帮助**  
```eh```

- **热重载config配置**  
```重载eh配置```

### 可用的搜索方式

1. 基础搜索：  
```搜eh [关键词]```

2. 高级搜索：  
```搜eh [关键词] [最低评分]```
 
    ```搜eh [关键词] [最低评分] [最少页数]```
   
    ```搜eh [关键词] [最低评分] [最少页数] [获取第几页的画廊列表]```

3. 快速跳转：  
```eh翻页 [获取第几页的画廊列表]```

### 可用的下载方式

1. 通过画廊链接下载：  
`看eh [画廊链接]`

2. 通过画廊索引下载：  
`看eh [搜索结果序号]`

**注意：**  
- 搜索多关键词时请用以下符号连接 `,` `，` `+` ，关键词之间不要添加任何空格
- 使用 `eh翻页 [获取第几页的画廊列表]` 和 `看eh [搜索结果序号]` 前确保你最近至少使用过一次 `搜eh` 命令（每个用户的缓存文件是独立的）

---

## 配置文件修改（重要！）

使用前请先修改配置文件：

1. **通过 WebUI 的插件管理面板设置（推荐）**
- 打开 "AstrBot WebUI" -> "插件管理" -> 找到本插件 -> "操作" -> "插件配置":
![image](https://github.com/user-attachments/assets/3f6487f6-27c6-4624-8cb7-9a8179538298)
- 插件将优先使用WebUI的配置并写入本地配置文件
- 缺少有效配置项时请不要填入内容，缺省项默认从本地配置文件读取

2. **通过插件本地目录的config.yaml文件设置**
- 平台设置
```
platform:
  type: "napcat" # 消息平台，兼容 napcat, llonebot, lagrange
  http_host: "127.0.0.1" # HTTP 服务器 IP，非 docker 部署一般为 127.0.0.1 或 localhost，docker 部署一般为宿主机局域网 IP
  http_port: 2333 # HTTP 服务器端口
  api_token: "" # HTTP 服务器 token，没有则不填
```

- 请求设置
```
request:
  headers:
    User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
  website: "e-hentai"  # 表站: e-hentai | 里站: exhentai
  cookies: # 缺少有效 cookie 时请不要将 website 设置为 exhentai
    ipb_member_id: ""
    ipb_pass_hash: ""
    igneous: ""
  proxies: "" # 墙内用户必填项，代理软件位于宿主机时，非 docker 部署一般为http://127.0.0.1:port，docker 部署一般为http://{宿主机局域网ip}:port
  concurrency: 10 # 并发数量限制
  max_retries: 3 # 请求重试次数，如果你的代理不稳定或带宽不够建议适量增加次数
  timeout: 5 # 超时时间，同上
```

- 输出设置
```
output:
  image_folder: "/app/sharedFolder/ehentai/tempImages" # 缓存画廊图片的路径
  pdf_folder: "/app/sharedFolder/ehentai/pdf" # 存放pdf文件的路径
  search_cache_folder: "/app/sharedFolder/ehentai/searchCache" # 缓存每个用户搜索结果的路径
  jpeg_quality: 85 # 图片质量，100 为不压缩，85 左右可以达到文件大小和图片质量的最佳平衡
  max_pages_per_pdf: 200 # 单个 PDF 文件最大页数
```

---

## 依赖库安装（重要！）

使用前请先安装以下依赖库：
- aiofiles
- aiohttp
- PyYAML
- natsort
- glob2
- python-magic
- beautifulsoup4
- img2pdf
- Pillow

在您的终端输入以下命令并回车：
```
pip install <module>
```
*使用具体模块名替换 &lt;module&gt;*

---

## Docker 部署注意事项

如果您是 Docker 部署，请务必为消息平台容器挂载 PDF 文件所在的文件夹，否则消息平台将无法解析文件路径。

示例挂载方式(NapCat)：
- 对 AstrBot：`/vol3/1000/dockerSharedFolder -> /app/sharedFolder`
- 对 NapCat：`/vol3/1000/dockerSharedFolder -> /app/sharedFolder`

---

## 已知 BUG

---

## 开发中的功能

- 随机画廊

---

## 使用示例
- 搜索  

![搜索示例](https://github.com/user-attachments/assets/68f7c828-5891-4b2e-abc3-f17e3b57eb37)

- 下载  

![下载示例](https://github.com/user-attachments/assets/f5f6085a-078c-4235-9bff-51e635bba3d6)

---

## 鸣谢

用户指令清洗和消息适配器参考了[exneverbur](https://github.com/exneverbur)的[ShowMeJM](https://github.com/exneverbur/ShowMeJM)项目，感谢

---
