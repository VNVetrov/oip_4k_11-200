import os
import re
import json
from bs4 import BeautifulSoup
from pymystem3 import Mystem
from collections import defaultdict

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
    'который', 'которая', 'которое', 'которые',
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


class InvertedIndex:
    def __init__(self, pages_dir='pages', index_file='index.txt'):
        self.pages_dir = pages_dir
        self.index_file = index_file
        self.mystem = Mystem()

        # Инвертированный индекс: лемма -> set(doc_id)
        self.inverted_index = defaultdict(set)

        # Частотный индекс: (лемма, doc_id) -> count
        self.term_freq = defaultdict(int)

        # Маппинг документов
        self.doc_map = {}
        self.all_docs = set()

    def _safe_analyze(self, text: str) -> list:
        """Обёртка над mystem.analyze с защитой от unicode-ошибок"""
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

    def load_doc_map(self):
        """Загружает маппинг файлов из index.txt"""
        if not os.path.exists(self.index_file):
            print(f"Файл '{self.index_file}' не найден!")
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
                        self.doc_map[doc_id] = {
                            'filename': filename,
                            'url': url
                        }
                        self.all_docs.add(doc_id)

        print(f"Загружено {len(self.doc_map)} документов из '{self.index_file}'")
        return True

    @staticmethod
    def extract_text(html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    def tokenize_and_lemmatize_with_freq(self, text: str) -> dict:
        """
        Токенизация + лемматизация.
        Возвращает dict: лемма -> количество вхождений
        """
        raw_tokens = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
        filtered = [t for t in raw_tokens if len(t) >= 2 and t not in STOP_WORDS]

        if not filtered:
            return {}

        lemma_freq = defaultdict(int)
        batch_text = ' '.join(filtered)
        analysis = self._safe_analyze(batch_text)

        for item in analysis:
            word = item.get('text', '').strip().lower()
            if not word or not re.match(r'^[а-яё]+$', word):
                continue

            a = item.get('analysis', [])
            lemma = a[0].get('lex', word).lower() if a else word

            if re.match(r'^[а-яё]+$', lemma) and lemma not in STOP_WORDS and len(lemma) >= 2:
                lemma_freq[lemma] += 1

        return dict(lemma_freq)

    def build_index(self):
        """Строит инвертированный индекс с частотами"""
        if not self.load_doc_map():
            return False

        print(f"\nПостроение инвертированного индекса...")
        print("=" * 60)

        for doc_id in sorted(self.doc_map.keys()):
            filepath = os.path.join(self.pages_dir, self.doc_map[doc_id]['filename'])

            if not os.path.exists(filepath):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except Exception as e:
                print(f"  Ошибка чтения {filepath}: {e}")
                continue

            text = self.extract_text(html_content)
            lemma_freq = self.tokenize_and_lemmatize_with_freq(text)

            for lemma, count in lemma_freq.items():
                self.inverted_index[lemma].add(doc_id)
                self.term_freq[(lemma, doc_id)] = count

            if (doc_id + 1) % 10 == 0 or doc_id == max(self.doc_map.keys()):
                print(f"  Обработано {doc_id + 1}/{len(self.doc_map)} документов, "
                      f"термов: {len(self.inverted_index)}")

        print(f"\nИндекс построен: {len(self.inverted_index)} термов")
        return True

    def save_index(self, output_path='inverted_index.txt'):
        """Сохраняет индекс в txt и json"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for term in sorted(self.inverted_index.keys()):
                doc_ids = sorted(self.inverted_index[term])
                doc_ids_str = ' '.join(str(d) for d in doc_ids)
                f.write(f"{term} {doc_ids_str}\n")

        print(f"Индекс сохранён в '{output_path}'")

        json_path = output_path.replace('.txt', '.json')
        json_data = {}
        for term in self.inverted_index:
            json_data[term] = {}
            for doc_id in sorted(self.inverted_index[term]):
                json_data[term][str(doc_id)] = self.term_freq.get((term, doc_id), 1)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"JSON-индекс с частотами сохранён в '{json_path}'")

    def load_saved_index(self, input_path='inverted_index.json'):
        """Загружает ранее сохранённый индекс с частотами"""
        if not os.path.exists(input_path):
            return False

        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        self.inverted_index = defaultdict(set)
        self.term_freq = defaultdict(int)

        for term, docs in json_data.items():
            if isinstance(docs, dict):
                # Новый формат: {doc_id: freq}
                for doc_id_str, freq in docs.items():
                    doc_id = int(doc_id_str)
                    self.inverted_index[term].add(doc_id)
                    self.term_freq[(term, doc_id)] = freq
            elif isinstance(docs, list):
                # Старый формат: [doc_id, ...]
                for doc_id in docs:
                    self.inverted_index[term].add(doc_id)
                    self.term_freq[(term, doc_id)] = 1

        print(f"Загружен индекс: {len(self.inverted_index)} термов из '{input_path}'")
        return True


    def lemmatize_query_term(self, term: str) -> str:
        term_lower = term.lower().strip()
        analysis = self._safe_analyze(term_lower)
        for item in analysis:
            word = item.get('text', '').strip()
            if not word:
                continue
            a = item.get('analysis', [])
            if a:
                return a[0].get('lex', term_lower).lower()
        return term_lower

    def search_term(self, term: str) -> set:
        try:
            lemma = self.lemmatize_query_term(term)
        except Exception:
            self.mystem = Mystem()
            try:
                lemma = self.lemmatize_query_term(term)
            except Exception:
                lemma = term.lower().strip()

        return set(self.inverted_index.get(lemma, set()))

    def extract_query_lemmas(self, query: str) -> list:
        """Извлекает все поисковые леммы из запроса (без операторов)"""
        tokens = self._tokenize_query(query)
        lemmas = []
        for token in tokens:
            if isinstance(token, tuple) and token[0] == 'TERM':
                try:
                    lemma = self.lemmatize_query_term(token[1])
                except Exception:
                    lemma = token[1].lower()
                lemmas.append(lemma)
        return lemmas

    def score_document(self, doc_id: int, query_lemmas: list) -> float:
        """
        Вычисляет релевантность документа.
        Суммирует частоты всех поисковых лемм в документе.
        """
        score = 0.0
        for lemma in query_lemmas:
            score += self.term_freq.get((lemma, doc_id), 0)
        return score

    def parse_and_evaluate(self, query: str) -> set:
        """Парсит и вычисляет булев запрос"""
        tokens = self._tokenize_query(query)
        result, _ = self._parse_or(tokens, 0)
        return result

    def _tokenize_query(self, query: str) -> list:
        tokens = []
        i = 0
        query = query.strip()

        while i < len(query):
            if query[i].isspace():
                i += 1
                continue
            if query[i] == '(':
                tokens.append('(')
                i += 1
                continue
            if query[i] == ')':
                tokens.append(')')
                i += 1
                continue

            j = i
            while j < len(query) and not query[j].isspace() and query[j] not in '()':
                j += 1
            word = query[i:j]

            if word.upper() == 'AND':
                tokens.append('AND')
            elif word.upper() == 'OR':
                tokens.append('OR')
            elif word.upper() == 'NOT':
                tokens.append('NOT')
            else:
                tokens.append(('TERM', word))
            i = j

        return tokens

    def _parse_or(self, tokens, pos):
        left, pos = self._parse_and(tokens, pos)
        while pos < len(tokens) and tokens[pos] == 'OR':
            pos += 1
            right, pos = self._parse_and(tokens, pos)
            left = left | right
        return left, pos

    def _parse_and(self, tokens, pos):
        left, pos = self._parse_not(tokens, pos)
        while pos < len(tokens) and tokens[pos] == 'AND':
            pos += 1
            right, pos = self._parse_not(tokens, pos)
            left = left & right
        return left, pos

    def _parse_not(self, tokens, pos):
        if pos < len(tokens) and tokens[pos] == 'NOT':
            pos += 1
            operand, pos = self._parse_not(tokens, pos)
            return self.all_docs - operand, pos
        return self._parse_primary(tokens, pos)

    def _parse_primary(self, tokens, pos):
        if pos >= len(tokens):
            return set(), pos
        token = tokens[pos]
        if token == '(':
            pos += 1
            result, pos = self._parse_or(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return result, pos
        if isinstance(token, tuple) and token[0] == 'TERM':
            pos += 1
            return self.search_term(token[1]), pos
        return set(), pos

    def format_results(self, doc_ids: set, query: str) -> str:
        """Форматирует результаты с ранжированием по частоте"""
        if not doc_ids:
            return "Ничего не найдено."

        # Извлекаем леммы запроса для скоринга
        query_lemmas = self.extract_query_lemmas(query)

        # Считаем скор для каждого документа
        scored_docs = []
        for doc_id in doc_ids:
            score = self.score_document(doc_id, query_lemmas)
            scored_docs.append((doc_id, score))

        # Сортируем: сначала по скору (убывание), потом по doc_id
        scored_docs.sort(key=lambda x: (-x[1], x[0]))

        lines = [f"Найдено документов: {len(doc_ids)}", f"Поисковые леммы: {', '.join(query_lemmas)}", "",
                 f"  {'#':<4} {'Док':>6} {'Частота':>8}  {'Файл':<20} URL",
                 f"  {'─' * 4} {'─' * 6} {'─' * 8}  {'─' * 20} {'─' * 40}"]

        for rank, (doc_id, score) in enumerate(scored_docs, 1):
            if doc_id in self.doc_map:
                info = self.doc_map[doc_id]
                lines.append(
                    f"  {rank:<4} {doc_id:>6} {score:>8.0f}  "
                    f"{info['filename']:<20} {info['url']}"
                )
            else:
                lines.append(f"  {rank:<4} {doc_id:>6} {score:>8.0f}  (недоступно)")

        return '\n'.join(lines)


def interactive_search(index: InvertedIndex):
    """Интерактивный режим поиска"""
    print("\n" + "=" * 60)
    print("БУЛЕВ ПОИСК (ранжирование по частоте)")
    print("=" * 60)
    print("Операторы: AND, OR, NOT")
    print("Скобки:    ( )")
    print("Примеры:")
    print("  Москва")
    print("  Москва AND Россия")
    print("  (Москва AND Россия) OR (Париж AND Франция)")
    print("  Москва AND NOT Кремль")
    print("Введите 'exit' для выхода\n")

    while True:
        try:
            query = input("Запрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query:
            continue
        if query.lower() in ('exit', 'quit', 'выход', 'q'):
            print("Выход.")
            break

        try:
            results = index.parse_and_evaluate(query)
            print(index.format_results(results, query))
            print()
        except Exception as e:
            print(f"Ошибка: {e}\n")


def main():
    pages_dir = 'pages'
    index_file = 'index.txt'
    inverted_index_file = 'inverted_index'

    if not os.path.exists(pages_dir):
        print(f"Директория '{pages_dir}' не найдена!")
        return

    index = InvertedIndex(pages_dir=pages_dir, index_file=index_file)

    json_path = inverted_index_file + '.json'
    if os.path.exists(json_path):
        print(f"Найден сохранённый индекс: {json_path}")
        choice = input("Загрузить? (y/n) [y]: ").strip().lower()
        if choice != 'n':
            index.load_saved_index(json_path)
            index.load_doc_map()
            interactive_search(index)
            return

    print("Построение индекса...\n")
    if not index.build_index():
        return

    index.save_index(inverted_index_file + '.txt')
    interactive_search(index)


if __name__ == "__main__":
    main()
