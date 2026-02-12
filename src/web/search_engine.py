# search_engine.py

import os
import re
import math
from collections import defaultdict
from bs4 import BeautifulSoup
from pymystem3 import Mystem

STOP_WORDS = {
    'в', 'на', 'по', 'с', 'со', 'к', 'ко', 'у', 'о', 'об', 'обо', 'от', 'ото',
    'из', 'изо', 'за', 'до', 'для', 'без', 'безо', 'при', 'про', 'через', 'между',
    'над', 'надо', 'под', 'подо', 'перед', 'передо', 'около', 'вокруг', 'после',
    'среди', 'вместо', 'кроме', 'помимо', 'сквозь', 'вдоль', 'ради', 'благодаря',
    'вследствие', 'насчёт', 'насчет', 'ввиду', 'посредством',
    'и', 'а', 'но', 'да', 'или', 'либо', 'ни', 'то', 'не', 'ли', 'бы',
    'что', 'чтобы', 'чтоб', 'как', 'так', 'если', 'когда', 'пока', 'хотя',
    'хоть', 'будто', 'словно', 'точно', 'раз', 'ведь', 'потому', 'поэтому',
    'причём', 'причем', 'притом', 'зато', 'однако', 'также', 'тоже', 'же',
    'только', 'лишь', 'даже', 'именно', 'ибо',
    'вот', 'вон', 'вовсе', 'разве', 'неужели', 'ну', 'уж',
    'я', 'мы', 'ты', 'вы', 'он', 'она', 'оно', 'они',
    'мой', 'моя', 'моё', 'мое', 'мои', 'наш', 'наша', 'наше', 'наши',
    'твой', 'твоя', 'твоё', 'твое', 'твои', 'ваш', 'ваша', 'ваше', 'ваши',
    'его', 'её', 'ее', 'их', 'себя', 'себе', 'собой', 'собою',
    'кто', 'какой', 'какая', 'какое', 'какие', 'чей', 'чья', 'чьё', 'чье', 'чьи',
    'который', 'которая', 'которое', 'которые', 'которого', 'которой', 'которых',
    'этот', 'эта', 'это', 'эти', 'тот', 'та', 'те',
    'сам', 'сама', 'само', 'сами', 'самый', 'самая', 'самое', 'самые',
    'весь', 'вся', 'всё', 'все', 'каждый', 'каждая', 'каждое', 'каждые',
    'другой', 'другая', 'другое', 'другие', 'иной', 'иная', 'иное', 'иные',
    'меня', 'мне', 'мной', 'мною', 'нас', 'нам', 'нами',
    'тебя', 'тебе', 'тобой', 'тобою', 'вас', 'вам', 'вами',
    'него', 'нему', 'ним', 'нём', 'нем', 'неё', 'нее', 'ней', 'нею',
    'них', 'ними', 'ему', 'им', 'ей', 'ею',
    'быть', 'был', 'была', 'было', 'были', 'есть', 'будет', 'будут',
    'будем', 'будете', 'буду', 'будешь',
    'уже', 'ещё', 'еще', 'очень', 'более', 'менее', 'тут', 'там', 'здесь',
    'где', 'куда', 'откуда', 'тогда', 'потом', 'затем', 'сейчас', 'теперь',
    'можно', 'нужно', 'надо', 'нельзя',
}

MIN_TOKEN_LENGTH = 2


class VectorSearchEngine:
    """
    Поисковый движок: загружает готовые TF-IDF файлы из задания 4,
    строит векторное пространство, ищет по косинусному сходству.
    """

    def __init__(
        self,
        pages_dir='pages',
        index_file='index.txt',
        tfidf_lemmas_dir='tfidf_lemmas',
        tfidf_tokens_dir='tfidf_tokens',
    ):
        self.pages_dir = pages_dir
        self.index_file = index_file
        self.tfidf_lemmas_dir = tfidf_lemmas_dir
        self.tfidf_tokens_dir = tfidf_tokens_dir
        self.mystem = Mystem()

        # doc_id -> {lemma: tfidf_value}  (загруженные векторы)
        self.doc_vectors = {}

        # lemma -> idf (берём из файлов)
        self.lemma_idf = {}

        # doc_id -> {'filename', 'url', 'title'}
        self.doc_info = {}

        # Инвертированный индекс: lemma -> set(doc_id)
        # Строим на лету при загрузке для быстрого поиска кандидатов
        self.inverted = defaultdict(set)

        self.vocabulary = set()
        self.num_docs = 0
        self.is_ready = False

    # ─── Безопасная работа с mystem ──────────────────────────

    def _safe_analyze(self, text: str) -> list:
        for attempt in range(2):
            try:
                clean = text.encode('utf-8', errors='replace').decode('utf-8')
                return self.mystem.analyze(clean)
            except (UnicodeEncodeError, UnicodeDecodeError):
                self.mystem = Mystem()
            except BrokenPipeError:
                self.mystem = Mystem()
            except Exception:
                break
        return []

    # ─── Загрузка данных ─────────────────────────────────────

    def _load_doc_map(self) -> bool:
        """Загружает index.txt: маппинг doc_id -> файл + URL"""
        if not os.path.exists(self.index_file):
            print(f"'{self.index_file}' не найден!")
            return False

        with open(self.index_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    filename, url = parts
                    match = re.search(r'page_(\d+)', filename)
                    if match:
                        doc_id = int(match.group(1))
                        self.doc_info[doc_id] = {
                            'filename': filename,
                            'url': url,
                            'title': '',
                        }

        self.num_docs = len(self.doc_info)
        print(f"Загружено {self.num_docs} документов из '{self.index_file}'")
        return self.num_docs > 0

    def _load_titles(self):
        """Извлекает заголовки из HTML-файлов"""
        total = len(self.doc_info)
        print(f"Извлечение заголовков из {total} документов...")

        for i, (doc_id, info) in enumerate(self.doc_info.items()):
            filepath = os.path.join(self.pages_dir, info['filename'])
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    html = f.read()
                soup = BeautifulSoup(html, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    title = re.sub(r'\s*[—\-–]\s*Википедия.*$', '', title)
                    info['title'] = title
                else:
                    info['title'] = info['filename']
            except Exception:
                info['title'] = info['filename']

            # Прогресс
            done = i + 1
            pct = done * 100 // total
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            print(f"\r  [{bar}] {done}/{total} ({pct}%)", end='', flush=True)
        print()

    def _load_tfidf_files(self):
        """
        Загружает готовые TF-IDF файлы из tfidf_lemmas/.
        Формат файла: <лемма> <idf> <tf-idf>
        """
        tfidf_dir = self.tfidf_lemmas_dir

        if not os.path.exists(tfidf_dir):
            print(f"Директория '{tfidf_dir}' не найдена!")
            return False

        tfidf_files = sorted([
            f for f in os.listdir(tfidf_dir)
            if f.endswith('.txt')
        ])

        if not tfidf_files:
            print(f"Нет TF-IDF файлов в '{tfidf_dir}'!")
            return False

        print(f"Загрузка {len(tfidf_files)} TF-IDF файлов из '{tfidf_dir}'...")

        loaded = 0
        for filename in tfidf_files:
            # Извлекаем doc_id из имени: tfidf_lemmas_0001.txt -> 1
            match = re.search(r'(\d+)', filename)
            if not match:
                continue
            doc_id = int(match.group(1))

            filepath = os.path.join(tfidf_dir, filename)
            vector = {}

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 3:
                            lemma = parts[0]
                            idf = float(parts[1])
                            tfidf = float(parts[2])

                            if tfidf > 0:
                                vector[lemma] = tfidf
                                self.inverted[lemma].add(doc_id)
                                self.vocabulary.add(lemma)

                                # Сохраняем IDF (берём максимальный если отличается)
                                if lemma not in self.lemma_idf or idf > self.lemma_idf[lemma]:
                                    self.lemma_idf[lemma] = idf

            except Exception as e:
                print(f"  Ошибка загрузки {filename}: {e}")
                continue

            self.doc_vectors[doc_id] = vector
            loaded += 1

            if loaded % 20 == 0:
                print(f"  Загружено {loaded}/{len(tfidf_files)}")

        print(f"  Загружено {loaded} файлов, {len(self.vocabulary)} уникальных термов")
        return loaded > 0

    # ─── Построение индекса ──────────────────────────────────

    def build(self):
        """Загружает все данные и готовит движок к работе"""
        print("Загрузка маппинга документов...")
        if not self._load_doc_map():
            return False

        self._load_titles()

        print("\nЗагрузка предвычисленных TF-IDF...")
        if not self._load_tfidf_files():
            return False

        self.is_ready = True

        print(f"\nДвижок готов:")
        print(f"  Документов:    {len(self.doc_vectors)}")
        print(f"  Термов:        {len(self.vocabulary)}")
        print(f"  IDF записей:   {len(self.lemma_idf)}")
        print(f"  Инверт. индекс: {len(self.inverted)} термов")

        return True

    # ─── Обработка запроса ───────────────────────────────────

    def _lemmatize_query(self, query: str) -> list:
        """Лемматизирует запрос, возвращает список лемм"""
        raw = re.findall(r'[а-яёА-ЯЁ]+', query.lower())
        valid = [
            w for w in raw
            if re.match(r'^[а-яё]+$', w)
            and len(w) >= MIN_TOKEN_LENGTH
            and w not in STOP_WORDS
        ]

        if not valid:
            return []

        lemmas = []
        batch = ' '.join(valid)
        analysis = self._safe_analyze(batch)

        for item in analysis:
            word = item.get('text', '').strip().lower()
            if not word or not re.match(r'^[а-яё]+$', word):
                continue

            a = item.get('analysis', [])
            lemma = a[0].get('lex', word).lower() if a else word

            if not re.match(r'^[а-яё]+$', lemma):
                lemma = word

            if lemma not in STOP_WORDS and len(lemma) >= MIN_TOKEN_LENGTH:
                lemmas.append(lemma)

        return lemmas

    def _query_to_vector(self, query: str) -> dict:
        """
        Преобразует запрос в TF-IDF вектор.
        TF считается по запросу, IDF берётся из загруженных файлов.
        """
        lemmas = self._lemmatize_query(query)
        if not lemmas:
            return {}, []

        # TF в запросе
        from collections import Counter
        counter = Counter(lemmas)
        total = len(lemmas)

        vector = {}
        for lemma, count in counter.items():
            tf = count / total
            idf = self.lemma_idf.get(lemma, 0.0)
            tfidf = tf * idf
            if tfidf > 0:
                vector[lemma] = tfidf

        return vector, list(set(lemmas))

    # ─── Косинусное сходство ─────────────────────────────────

    @staticmethod
    def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0

        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ─── Сниппеты ────────────────────────────────────────────

    def _extract_snippet(self, html: str, query_lemmas: set, max_len: int = 250) -> str:
        """Извлекает фрагмент текста с подсветкой релевантных слов"""
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()

        paragraphs = soup.find_all('p')
        blocks = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]

        if not blocks:
            full = soup.get_text(separator=' ', strip=True)
            blocks = [full] if full else []

        best_block = ""
        best_score = -1

        for block in blocks:
            words = re.findall(r'[а-яёА-ЯЁ]+', block.lower())

            # Лемматизируем слова блока для сравнения
            score = 0
            for w in words:
                if w in query_lemmas:
                    score += 1

            if score > best_score:
                best_score = score
                best_block = block

        if not best_block and blocks:
            best_block = blocks[0]

        if len(best_block) > max_len:
            best_block = best_block[:max_len] + "..."

        return best_block

    # ─── Поиск ───────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list:
        """
        Основной метод поиска.
        1. Лемматизирует запрос
        2. Строит TF-IDF вектор запроса
        3. Находит кандидатов через инвертированный индекс
        4. Считает косинусное сходство
        5. Ранжирует и возвращает top_k
        """
        if not self.is_ready:
            return []

        query_vector, query_lemmas = self._query_to_vector(query)
        if not query_vector:
            return []

        query_lemma_set = set(query_lemmas)

        # Кандидаты: документы, содержащие хотя бы один терм запроса
        candidates = set()
        for lemma in query_lemma_set:
            candidates.update(self.inverted.get(lemma, set()))

        if not candidates:
            return []

        # Скоринг
        scored = []
        for doc_id in candidates:
            doc_vec = self.doc_vectors.get(doc_id, {})
            if not doc_vec:
                continue
            score = self._cosine_similarity(query_vector, doc_vec)
            if score > 0:
                scored.append((doc_id, score))

        scored.sort(key=lambda x: -x[1])
        scored = scored[:top_k]

        # Формируем результаты
        results = []
        for doc_id, score in scored:
            info = self.doc_info.get(doc_id, {})
            filepath = os.path.join(self.pages_dir, info.get('filename', ''))

            snippet = ""
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        html = f.read()
                    snippet = self._extract_snippet(html, query_lemma_set)
                except Exception:
                    pass

            results.append({
                'doc_id': doc_id,
                'score': round(score, 6),
                'title': info.get('title', 'Без названия'),
                'url': info.get('url', ''),
                'filename': info.get('filename', ''),
                'snippet': snippet,
                'matched_terms': sorted(
                    query_lemma_set & set(self.doc_vectors.get(doc_id, {}).keys())
                ),
            })

        return results

    def get_stats(self) -> dict:
        return {
            'num_docs': len(self.doc_vectors),
            'num_terms': len(self.vocabulary),
            'num_idf': len(self.lemma_idf),
            'is_ready': self.is_ready,
            'tfidf_dir': self.tfidf_lemmas_dir,
        }