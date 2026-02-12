import asyncio
import os
import ssl
import time
from urllib.parse import urljoin, urlparse, unquote, urldefrag, urlunparse

import certifi
import aiofiles
import aiohttp
from bs4 import BeautifulSoup


class AsyncWebCrawler:
    def __init__(self, start_url, max_pages=100, output_dir='pages',
                 delay=0.1, max_concurrent=10):
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc.lower()
        self.max_pages = max_pages
        self.output_dir = output_dir
        self.delay = delay
        self.max_concurrent = max_concurrent

        self.visited = set()
        self.in_queue = set()
        self.queue = asyncio.Queue()
        self.page_count = 0
        self.index = []
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.done = False

        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def normalize_url(url: str) -> str:
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')
        if not path:
            path = '/'
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ''))

    @staticmethod
    def pretty_url(url: str) -> str:
        return unquote(url)

    def is_valid_url(self, url: str) -> bool:
        if not url:
            return False

        parsed = urlparse(url)

        if parsed.scheme not in ('http', 'https'):
            return False

        if parsed.netloc.lower() != self.base_domain:
            return False

        # Пропускаем URL с query-параметрами action= (редактирование и т.д.)
        if parsed.query and 'action=' in parsed.query:
            return False

        path = parsed.path.lower()
        decoded_path = unquote(parsed.path)

        # Для Wikipedia: пропускаем служебные пространства
        skip_path_prefixes = (
            '/w/', '/wiki/Special:', '/wiki/Служебная:',
            '/wiki/Обсуждение:', '/wiki/Talk:',
            '/wiki/Участник:', '/wiki/User:',
            '/wiki/Обсуждение_участника:', '/wiki/User_talk:',
            '/wiki/Файл:', '/wiki/File:',
            '/wiki/Категория:', '/wiki/Category:',
            '/wiki/Шаблон:', '/wiki/Template:',
            '/wiki/Портал:', '/wiki/Portal:',
            '/wiki/Модуль:', '/wiki/Module:',
            '/wiki/Справка:', '/wiki/Help:',
            '/wiki/Обсуждение_Википедии:',
            '/wiki/Обсуждение_файла:',
            '/wiki/Обсуждение_шаблона:',
            '/wiki/Обсуждение_категории:',
        )
        if any(decoded_path.startswith(p) for p in skip_path_prefixes):
            return False

        skip_extensions = (
            '.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.svg',
            '.ico', '.webp', '.mp3', '.mp4', '.avi', '.pdf', '.doc',
            '.docx', '.xls', '.xlsx', '.zip', '.rar', '.exe', '.dmg',
            '.iso', '.tar', '.gz', '.7z', '.ttf', '.woff', '.woff2',
            '.eot', '.otf', '.json', '.xml', '.rss', '.atom',
            '.bmp', '.tiff', '.mov', '.wmv', '.flv', '.swf',
            '.ogg', '.ogv', '.wav',
        )
        if any(path.endswith(ext) for ext in skip_extensions):
            return False

        if url.startswith(('javascript:', 'mailto:', 'tel:', 'data:', '#')):
            return False

        return True

    def extract_links(self, html: str, base_url: str) -> set:
        links = set()
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Для Wikipedia берём ссылки из контента, не из навигации
            content = soup.find('div', {'id': 'bodyContent'}) or soup

            for tag in content.find_all('a', href=True):
                href = tag['href'].strip()
                if not href:
                    continue
                absolute_url = urljoin(base_url, href)
                normalized = self.normalize_url(absolute_url)
                if self.is_valid_url(normalized):
                    links.add(normalized)
        except Exception as e:
            print(f"\tОшибка парсинга ссылок: {e}")
        return links

    async def fetch_page(self, session: aiohttp.ClientSession, url: str):
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
        }
        try:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
                allow_redirects=True,
            ) as response:
                if response.status != 200:
                    print(f"\tHTTP {response.status}: {self.pretty_url(url)}")
                    return None

                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    return None

                try:
                    text = await response.text()
                except UnicodeDecodeError:
                    raw = await response.read()
                    text = raw.decode('utf-8', errors='replace')

                return text

        except asyncio.TimeoutError:
            print(f"\tТаймаут: {self.pretty_url(url)}")
            return None
        except aiohttp.ClientError as e:
            print(f"\tСеть: {type(e).__name__}: {self.pretty_url(url)}")
            return None
        except Exception as e:
            print(f"\tОшибка: {type(e).__name__}: {e}")
            return None

    async def save_page(self, url: str, content: str, page_num: int):
        filename = f"page_{page_num:04d}.html"
        filepath = os.path.join(self.output_dir, filename)
        try:
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(content)
            return filename
        except Exception as e:
            print(f"\tОшибка сохранения: {e}")
            return None

    async def worker(self, worker_id: int, session: aiohttp.ClientSession):
        while not self.done:
            async with self.lock:
                if self.page_count >= self.max_pages:
                    self.done = True
                    return

            try:
                url = await asyncio.wait_for(self.queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                return

            async with self.lock:
                if url in self.visited or self.page_count >= self.max_pages:
                    self.queue.task_done()
                    continue
                self.visited.add(url)

            async with self.semaphore:
                content = await self.fetch_page(session, url)

                if content:
                    async with self.lock:
                        if self.page_count >= self.max_pages:
                            self.queue.task_done()
                            return
                        current_num = self.page_count
                        self.page_count += 1

                    filename = await self.save_page(url, content, current_num)

                    if filename:
                        async with self.lock:
                            self.index.append(f"{filename}\t{url}")
                            count = self.page_count

                        print(
                            f"  [{count}/{self.max_pages}] "
                            f"W{worker_id} ✓ {self.pretty_url(url)}"
                        )

                        new_links = self.extract_links(content, url)
                        added = 0
                        async with self.lock:
                            for link in new_links:
                                if link not in self.visited and link not in self.in_queue:
                                    self.in_queue.add(link)
                                    await self.queue.put(link)
                                    added += 1

                        if added > 0:
                            print(
                                f"         + {added} ссылок | "
                                f"очередь: ~{self.queue.qsize()} | "
                                f"посещено: {len(self.visited)}"
                            )
                    else:
                        async with self.lock:
                            self.page_count -= 1

                if self.delay > 0:
                    await asyncio.sleep(self.delay)

            self.queue.task_done()

    async def test_connection(self, session: aiohttp.ClientSession):
        """Тестовый запрос перед стартом"""
        print(f"\nТестируем: {self.pretty_url(self.start_url)}")
        content = await self.fetch_page(session, self.start_url)
        if content:
            print(f"OK — получено {len(content):,} символов")
            links = self.extract_links(content, self.start_url)
            print(f"Найдено {len(links)} ссылок на стартовой странице")
            for sample in list(links)[:5]:
                print(f"\t{self.pretty_url(sample)}")
            return True
        else:
            print("Не удалось загрузить стартовую страницу!")
            return False

    async def save_index(self):
        index_path = 'index.txt'
        try:
            async with aiofiles.open(index_path, 'w', encoding='utf-8') as f:
                await f.write('\n'.join(self.index))
            print(f"\nИндекс сохранён: {index_path}")
        except Exception as e:
            print(f"Ошибка индекса: {e}")

    async def crawl(self):
        start_normalized = self.normalize_url(self.start_url)

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            ssl=self.ssl_context,
            enable_cleanup_closed=True,
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            if not await self.test_connection(session):
                return

            print(f"\nКраулинг: {self.max_pages} страниц, {self.max_concurrent} воркеров")
            print("─" * 60)

            self.in_queue.add(start_normalized)
            await self.queue.put(start_normalized)

            start_time = time.time()

            workers = [
                asyncio.create_task(self.worker(i, session))
                for i in range(self.max_concurrent)
            ]
            await asyncio.gather(*workers, return_exceptions=True)

            elapsed = time.time() - start_time

        await self.save_index()

        print("\n" + "═" * 60)
        print(f"Скачано: {self.page_count}")
        print(f"Посещено: {len(self.visited)}")
        print(f"Время: {elapsed:.1f} сек")
        if self.page_count > 0 and elapsed > 0:
            print(f"Скорость: {self.page_count / elapsed:.1f} стр/сек")
        print(f"Файлы: {self.output_dir}/")


async def main():
    with open("pivot.txt", 'r', encoding='utf-8') as f:
        start_url = f.readline().strip()

    print(f"Стартовая страница: {unquote(start_url)}")

    try:
        max_pages = int(input("Страниц [100]: ").strip() or "100")
    except ValueError:
        max_pages = 100

    try:
        concurrency = int(input("Параллельно [5]: ").strip() or "5")
    except ValueError:
        concurrency = 5

    crawler = AsyncWebCrawler(
        start_url=start_url,
        max_pages=max_pages,
        output_dir='pages',
        delay=0.2,
        max_concurrent=concurrency,
    )
    await crawler.crawl()


if __name__ == "__main__":
    asyncio.run(main())