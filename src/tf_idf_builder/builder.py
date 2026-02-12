import os
import re
import math
from bs4 import BeautifulSoup
from pymystem3 import Mystem
from collections import defaultdict, Counter

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


class TfIdfCalculator:
    def __init__(self, pages_dir='pages', output_tokens_dir='tfidf_tokens',
                 output_lemmas_dir='tfidf_lemmas'):
        self.pages_dir = pages_dir
        self.output_tokens_dir = output_tokens_dir
        self.output_lemmas_dir = output_lemmas_dir
        self.mystem = Mystem()

        # Данные по документам
        # doc_id -> Counter токенов (словоформ)
        self.doc_token_counts = {}

        # doc_id -> Counter лемм
        self.doc_lemma_counts = {}

        # doc_id -> общее число токенов в документе
        self.doc_total_tokens = {}

        # doc_id -> маппинг токен -> лемма
        self.doc_token_to_lemma = {}

        # Глобальные множества
        self.all_tokens = set()
        self.all_lemmas = set()

        # Список файлов
        self.html_files = []
        self.num_docs = 0

        # IDF
        # токен -> в скольких документах встречается
        self.token_doc_freq = defaultdict(int)
        # лемма -> в скольких документах встречается
        self.lemma_doc_freq = defaultdict(int)

        os.makedirs(output_tokens_dir, exist_ok=True)
        os.makedirs(output_lemmas_dir, exist_ok=True)

    def _safe_analyze(self, text: str) -> list:
        """Безопасный вызов mystem"""
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

    def extract_text(self, html_content: str) -> str:
        """Извлекает текст из HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    def is_valid_token(self, token: str) -> bool:
        """Проверяет валидность токена"""
        if not re.match(r'^[а-яё]+$', token):
            return False
        if len(token) < MIN_TOKEN_LENGTH:
            return False
        if token in STOP_WORDS:
            return False
        return True

    def process_document(self, doc_id: int, filepath: str):
        """
        Обрабатывает один документ:
        - токенизация
        - лемматизация
        - подсчёт частот
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            print(f"  Ошибка чтения {filepath}: {e}")
            return

        text = self.extract_text(html_content)

        # Извлекаем все кириллические слова
        raw_words = re.findall(r'[а-яёА-ЯЁ]+', text.lower())

        # Фильтруем
        valid_tokens = [w for w in raw_words if self.is_valid_token(w)]

        if not valid_tokens:
            self.doc_token_counts[doc_id] = Counter()
            self.doc_lemma_counts[doc_id] = Counter()
            self.doc_total_tokens[doc_id] = 0
            self.doc_token_to_lemma[doc_id] = {}
            return

        # Считаем частоты токенов
        token_counter = Counter(valid_tokens)
        total_tokens = sum(token_counter.values())

        # Лемматизация
        token_to_lemma = {}
        lemma_counter = Counter()

        # Обрабатываем уникальные токены батчами
        unique_tokens = list(token_counter.keys())
        batch_size = 500

        for start in range(0, len(unique_tokens), batch_size):
            batch = unique_tokens[start:start + batch_size]
            batch_text = '\n'.join(batch)
            analysis = self._safe_analyze(batch_text)

            for item in analysis:
                word = item.get('text', '').strip().lower()
                if not word or not re.match(r'^[а-яё]+$', word):
                    continue

                a = item.get('analysis', [])
                if a:
                    lemma = a[0].get('lex', word).lower()
                else:
                    lemma = word

                if not re.match(r'^[а-яё]+$', lemma):
                    lemma = word

                if lemma in STOP_WORDS or len(lemma) < MIN_TOKEN_LENGTH:
                    continue

                if word in token_counter:
                    token_to_lemma[word] = lemma
                    lemma_counter[lemma] += token_counter[word]

        # Сохраняем
        self.doc_token_counts[doc_id] = token_counter
        self.doc_lemma_counts[doc_id] = lemma_counter
        self.doc_total_tokens[doc_id] = total_tokens
        self.doc_token_to_lemma[doc_id] = token_to_lemma

        # Обновляем глобальные множества
        self.all_tokens.update(token_counter.keys())
        self.all_lemmas.update(lemma_counter.keys())

    def compute_doc_frequencies(self):
        """Подсчитывает document frequency для каждого токена и леммы"""
        print("\nПодсчёт document frequencies...")

        for doc_id in self.doc_token_counts:
            seen_tokens = set(self.doc_token_counts[doc_id].keys())
            for token in seen_tokens:
                self.token_doc_freq[token] += 1

            seen_lemmas = set(self.doc_lemma_counts[doc_id].keys())
            for lemma in seen_lemmas:
                self.lemma_doc_freq[lemma] += 1

        print(f"  Уникальных токенов: {len(self.token_doc_freq)}")
        print(f"  Уникальных лемм: {len(self.lemma_doc_freq)}")

    def compute_idf(self, doc_freq: int) -> float:
        """
        IDF = log(N / df)
        где N — общее число документов, df — число документов с термином
        """
        if doc_freq == 0:
            return 0.0
        return math.log(self.num_docs / doc_freq)

    def compute_tf(self, term_count: int, total_tokens: int) -> float:
        """
        TF = count(term) / total_tokens
        """
        if total_tokens == 0:
            return 0.0
        return term_count / total_tokens

    def save_tfidf_for_document(self, doc_id: int):
        """Сохраняет TF-IDF файлы для одного документа"""
        # ─── Файл для токенов ─────────────────────────────────
        token_counter = self.doc_token_counts.get(doc_id, Counter())
        total = self.doc_total_tokens.get(doc_id, 0)

        token_filename = f"tfidf_tokens_{doc_id:04d}.txt"
        token_filepath = os.path.join(self.output_tokens_dir, token_filename)

        with open(token_filepath, 'w', encoding='utf-8') as f:
            for token in sorted(token_counter.keys()):
                tf = self.compute_tf(token_counter[token], total)
                idf = self.compute_idf(self.token_doc_freq.get(token, 0))
                tfidf = tf * idf
                f.write(f"{token} {idf:.6f} {tfidf:.6f}\n")

        # ─── Файл для лемм ───────────────────────────────────
        lemma_counter = self.doc_lemma_counts.get(doc_id, Counter())

        lemma_filename = f"tfidf_lemmas_{doc_id:04d}.txt"
        lemma_filepath = os.path.join(self.output_lemmas_dir, lemma_filename)

        with open(lemma_filepath, 'w', encoding='utf-8') as f:
            for lemma in sorted(lemma_counter.keys()):
                tf = self.compute_tf(lemma_counter[lemma], total)
                idf = self.compute_idf(self.lemma_doc_freq.get(lemma, 0))
                tfidf = tf * idf
                f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

    def run(self):
        """Основной pipeline"""
        # 1. Находим HTML-файлы
        self.html_files = sorted([
            f for f in os.listdir(self.pages_dir)
            if f.endswith('.html')
        ])
        self.num_docs = len(self.html_files)

        if self.num_docs == 0:
            print(f"Нет HTML-файлов в '{self.pages_dir}'!")
            return

        print(f"Найдено {self.num_docs} документов")
        print("=" * 60)

        # 2. Обрабатываем каждый документ
        print("\nЭТАП 1: Токенизация и лемматизация документов")
        print("-" * 60)

        for i, filename in enumerate(self.html_files):
            match = re.search(r'page_(\d+)', filename)
            doc_id = int(match.group(1)) if match else i

            filepath = os.path.join(self.pages_dir, filename)
            self.process_document(doc_id, filepath)

            if (i + 1) % 10 == 0 or i == self.num_docs - 1:
                print(f"  Обработано {i + 1}/{self.num_docs}")

        # 3. Подсчёт document frequency
        print("\nЭТАП 2: Подсчёт IDF")
        print("-" * 60)
        self.compute_doc_frequencies()

        # 4. Сохраняем TF-IDF для каждого документа
        print("\nЭТАП 3: Вычисление и сохранение TF-IDF")
        print("-" * 60)

        doc_ids = sorted(self.doc_token_counts.keys())
        for i, doc_id in enumerate(doc_ids):
            self.save_tfidf_for_document(doc_id)

            if (i + 1) % 10 == 0 or i == len(doc_ids) - 1:
                print(f"  Сохранено {i + 1}/{len(doc_ids)}")

        # 5. Статистика
        print("\n" + "=" * 60)
        print("ГОТОВО!")
        print(f"  Документов обработано: {self.num_docs}")
        print(f"  Уникальных токенов:   {len(self.all_tokens)}")
        print(f"  Уникальных лемм:      {len(self.all_lemmas)}")
        print(f"  TF-IDF токенов:       {self.output_tokens_dir}/")
        print(f"  TF-IDF лемм:          {self.output_lemmas_dir}/")

        # Примеры
        self._print_examples(doc_ids)

    def _print_examples(self, doc_ids: list):
        """Выводит примеры TF-IDF для первого документа"""
        if not doc_ids:
            return

        doc_id = doc_ids[0]
        print(f"\n  Пример (документ {doc_id:04d}):")

        token_counter = self.doc_token_counts.get(doc_id, Counter())
        total = self.doc_total_tokens.get(doc_id, 0)

        # Топ-10 токенов по TF-IDF
        token_tfidf = []
        for token, count in token_counter.items():
            tf = self.compute_tf(count, total)
            idf = self.compute_idf(self.token_doc_freq.get(token, 0))
            token_tfidf.append((token, tf, idf, tf * idf))

        token_tfidf.sort(key=lambda x: -x[3])

        print(f"\n  Топ-10 токенов по TF-IDF:")
        print(f"  {'Токен':<20} {'TF':>8} {'IDF':>8} {'TF-IDF':>10}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 10}")
        for token, tf, idf, tfidf in token_tfidf[:10]:
            print(f"  {token:<20} {tf:>8.4f} {idf:>8.4f} {tfidf:>10.6f}")

        # Топ-10 лемм по TF-IDF
        lemma_counter = self.doc_lemma_counts.get(doc_id, Counter())
        lemma_tfidf = []
        for lemma, count in lemma_counter.items():
            tf = self.compute_tf(count, total)
            idf = self.compute_idf(self.lemma_doc_freq.get(lemma, 0))
            lemma_tfidf.append((lemma, tf, idf, tf * idf))

        lemma_tfidf.sort(key=lambda x: -x[3])

        print(f"\n  Топ-10 лемм по TF-IDF:")
        print(f"  {'Лемма':<20} {'TF':>8} {'IDF':>8} {'TF-IDF':>10}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 10}")
        for lemma, tf, idf, tfidf in lemma_tfidf[:10]:
            print(f"  {lemma:<20} {tf:>8.4f} {idf:>8.4f} {tfidf:>10.6f}")


def main():
    calculator = TfIdfCalculator(
        pages_dir='pages',
        output_tokens_dir='tfidf_tokens',
        output_lemmas_dir='tfidf_lemmas'
    )
    calculator.run()


if __name__ == "__main__":
    main()
