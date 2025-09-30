import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, Union
from urllib.parse import urlparse
import aiohttp

logger = logging.getLogger(__name__)


def parse_proxy_config(proxy_str: str) -> Dict[str, Any]:
    if not proxy_str:
        return {}

    parsed = urlparse(proxy_str)

    if parsed.scheme not in ('http', 'https', 'socks5'):
        raise ValueError("仅支持HTTP/HTTPS/SOCKS5代理协议")

    auth = None
    if parsed.username and parsed.password:
        auth = aiohttp.BasicAuth(parsed.username, parsed.password)

    proxy_url = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        proxy_url += f":{parsed.port}"

    return {
        'url': proxy_url,
        'auth': auth
    }


def load_config(config: dict, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        yaml_path = Path(__file__).parent.parent / "config.yaml"
    else:
        yaml_path = Path(config_path)
    
    yaml_config = {}
    try:
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
                logger.info(f"已加载YAML配置文件: {yaml_path}")
        else:
            logger.warning(f"YAML配置文件不存在: {yaml_path}, 将创建新文件")
    except Exception as e:
        logger.warning(f"加载YAML配置文件失败: {str(e)}")
    
    int_fields = [
        "platform_http_port",
        "request_concurrency", 
        "request_max_retries",
        "request_timeout",
        "output_jpeg_quality",
        "output_max_pages_per_pdf",
        "output_max_filename_length",
        "features_search_results_limit"
    ]
    
    bool_fields = [
        "features_enable_formatted_message_search",
        "features_enable_cover_image_download"
    ]
    
    processed_config = {}
    for key, value in config.items():
        if value == "" or value is None:
            continue
        
        if key in int_fields:
            try:
                processed_config[key] = int(value)
                logger.debug(f"已将 {key} 从 '{value}' 转换为整数: {processed_config[key]}")
            except (ValueError, TypeError):
                logger.warning(f"无法将 {key} 的值 '{value}' 转换为整数，已跳过此项")
                continue
        elif key in bool_fields:
            if isinstance(value, str):
                processed_config[key] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                processed_config[key] = bool(value)
            logger.debug(f"已将 {key} 从 '{value}' 转换为布尔值: {processed_config[key]}")
        else:
            processed_config[key] = value
    
    json_to_yaml_mapping = {
        "platform_type": ["platform", "type"],
        "platform_http_host": ["platform", "http_host"],
        "platform_http_port": ["platform", "http_port"],
        "platform_api_token": ["platform", "api_token"],
        "request_headers_user_agent": ["request", "headers", "User-Agent"],
        "request_website": ["request", "website"],
        "request_cookies_ipb_member_id": ["request", "cookies", "ipb_member_id"],
        "request_cookies_ipb_pass_hash": ["request", "cookies", "ipb_pass_hash"],
        "request_cookies_igneous": ["request", "cookies", "igneous"],
        "request_cookies_sk": ["request", "cookies", "sk"], 
        "request_proxies": ["request", "proxies"],
        "request_concurrency": ["request", "concurrency"],
        "request_max_retries": ["request", "max_retries"],
        "request_timeout": ["request", "timeout"],
        "output_image_folder": ["output", "image_folder"],
        "output_pdf_folder": ["output", "pdf_folder"],
        "output_search_cache_folder": ["output", "search_cache_folder"],
        "output_jpeg_quality": ["output", "jpeg_quality"],
        "output_max_pages_per_pdf": ["output", "max_pages_per_pdf"],
        "output_max_filename_length": ["output", "max_filename_length"],
        "features_enable_formatted_message_search": ["features", "enable_formatted_message_search"],
        "features_enable_cover_image_download": ["features", "enable_cover_image_download"],
        "features_search_results_limit": ["features", "search_results_limit"]
    }
    
    updates_needed = False
    
    for json_key, value in processed_config.items():
        if json_key in json_to_yaml_mapping:
            yaml_path_parts = json_to_yaml_mapping[json_key]
            
            current_yaml = yaml_config
            for i, path_part in enumerate(yaml_path_parts[:-1]):
                current_yaml = current_yaml.setdefault(path_part, {})
            
            key = yaml_path_parts[-1]
            yaml_value = current_yaml.get(key)
            
            if isinstance(yaml_value, int) and isinstance(value, str):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            
            if yaml_value != value:
                logger.info(f"发现配置差异: {'.'.join(yaml_path_parts)}={value} (YAML中为: {yaml_value})")
                current_yaml[key] = value
                updates_needed = True
    
    config_updated = False
    if 'request' in yaml_config:
        request = yaml_config['request']
        website = request.get('website')
        cookies = request.get('cookies', {})
        
        if website == 'exhentai':
            if any(not cookies.get(key, '') for key in ["ipb_member_id", "ipb_pass_hash", "igneous"]):
                request['website'] = 'e-hentai'
                config_updated = True
                logger.warning("网站设置为里站exhentai但cookies不完整，已更换为表站e-hentai")
        
        proxy_str = request.get('proxies', '')
        proxy_config = parse_proxy_config(proxy_str)
        request['proxy'] = proxy_config
    
    if updates_needed or config_updated:
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False)
                logger.info(f"已更新YAML配置文件: {yaml_path}")
        except Exception as e:
            logger.error(f"写入YAML配置文件失败: {str(e)}")
    
    return yaml_config
    
