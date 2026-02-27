# tokenizer.py

import os
import re
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


class Tokenizer:
    def __init__(self, pages_dir='pages',
                 output_tokens_dir='tokens',
                 output_lemmas_dir='lemmas'):
        self.pages_dir = pages_dir
        self.output_tokens_dir = output_tokens_dir
        self.output_lemmas_dir = output_lemmas_dir
        self.mystem = Mystem()

        # Глобальные множества для общих файлов
        self.all_tokens = set()
        self.all_lemma_groups = defaultdict(set)  # лемма -> set(токенов)

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

    def _extract_text(self, html_content: str) -> str:
        """Извлекает текст из HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    def _is_valid_token(self, token: str) -> bool:
        """Проверяет валидность токена"""
        if not re.match(r'^[а-яё]+$', token):
            return False
        if len(token) < MIN_TOKEN_LENGTH:
            return False
        if token in STOP_WORDS:
            return False
        return True

    def _process_document(self, doc_id: int, filepath: str):
        """
        Обрабатывает один документ:
        1. Извлекает текст
        2. Токенизирует
        3. Лемматизирует
        4. Сохраняет файл токенов и файл лемм
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            print(f"\n  Ошибка чтения {filepath}: {e}")
            return

        text = self._extract_text(html_content)

        raw_words = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
        valid_tokens = set(w for w in raw_words if self._is_valid_token(w))

        if not valid_tokens:
            # Пустые файлы всё равно создаём
            self._save_tokens_file(doc_id, set())
            self._save_lemmas_file(doc_id, {})
            return

        # token -> lemma
        token_to_lemma = {}
        # lemma -> set(tokens) для этого документа
        doc_lemma_groups = defaultdict(set)

        unique_list = sorted(valid_tokens)
        batch_size = 500

        for start in range(0, len(unique_list), batch_size):
            batch = unique_list[start:start + batch_size]
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

                if word in valid_tokens:
                    token_to_lemma[word] = lemma
                    doc_lemma_groups[lemma].add(word)

        # Оставляем только токены, для которых нашлась лемма
        final_tokens = set(token_to_lemma.keys())

        self._save_tokens_file(doc_id, final_tokens)
        self._save_lemmas_file(doc_id, doc_lemma_groups)

        # Обновляем глобальные данные
        self.all_tokens.update(final_tokens)
        for lemma, tokens in doc_lemma_groups.items():
            self.all_lemma_groups[lemma].update(tokens)

    def _save_tokens_file(self, doc_id: int, tokens: set):
        """
        Сохраняет файл токенов для одного документа.
        Формат: <токен>\n
        """
        filename = f"tokens_{doc_id:04d}.txt"
        filepath = os.path.join(self.output_tokens_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            for token in sorted(tokens):
                f.write(token + '\n')

    def _save_lemmas_file(self, doc_id: int, lemma_groups: dict):
        """
        Сохраняет файл лемм для одного документа.
        Формат: <лемма> <токен1> <токен2> ... <токенN>\n
        """
        filename = f"lemmas_{doc_id:04d}.txt"
        filepath = os.path.join(self.output_lemmas_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            for lemma in sorted(lemma_groups.keys()):
                tokens_sorted = sorted(lemma_groups[lemma])
                line = lemma + ' ' + ' '.join(tokens_sorted)
                f.write(line + '\n')

    def _save_global_files(self):
        """Сохраняет общие файлы tokens.txt и lemmas.txt"""
        # Общий файл токенов
        tokens_path = 'tokens.txt'
        with open(tokens_path, 'w', encoding='utf-8') as f:
            for token in sorted(self.all_tokens):
                f.write(token + '\n')
        print(f"  Общий файл токенов: {tokens_path} ({len(self.all_tokens)} токенов)")

        # Общий файл лемм
        lemmas_path = 'lemmas.txt'
        with open(lemmas_path, 'w', encoding='utf-8') as f:
            for lemma in sorted(self.all_lemma_groups.keys()):
                tokens_sorted = sorted(self.all_lemma_groups[lemma])
                line = lemma + ' ' + ' '.join(tokens_sorted)
                f.write(line + '\n')
        print(f"  Общий файл лемм:   {lemmas_path} ({len(self.all_lemma_groups)} лемм)")

    def run(self):
        """Основной pipeline"""
        # Находим HTML-файлы
        html_files = sorted([
            f for f in os.listdir(self.pages_dir)
            if f.endswith('.html')
        ])

        total = len(html_files)
        if total == 0:
            print(f"Нет HTML-файлов в '{self.pages_dir}'!")
            return

        print(f"Найдено {total} HTML-файлов")
        print("=" * 60)

        print("\nТокенизация и лемматизация документов...")

        for i, filename in enumerate(html_files):
            match = re.search(r'page_(\d+)', filename)
            doc_id = int(match.group(1)) if match else i

            filepath = os.path.join(self.pages_dir, filename)
            self._process_document(doc_id, filepath)

            # Прогресс-бар
            done = i + 1
            pct = done * 100 // total
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            print(f"\r  [{bar}] {done}/{total} ({pct}%)", end='', flush=True)

        print()

        print("\nСохранение общих файлов...")
        self._save_global_files()

        print("\n" + "=" * 60)
        print("ГОТОВО!")
        print(f"  Документов обработано:  {total}")
        print(f"  Уникальных токенов:     {len(self.all_tokens)}")
        print(f"  Уникальных лемм:        {len(self.all_lemma_groups)}")
        print()
        print(f"  Файлы токенов (по док.): {self.output_tokens_dir}/")
        print(f"  Файлы лемм (по док.):    {self.output_lemmas_dir}/")
        print(f"  Общий tokens.txt")
        print(f"  Общий lemmas.txt")

        # Примеры
        self._print_examples()

    def _print_examples(self):
        """Выводит примеры содержимого файлов"""
        # Пример файла токенов
        token_files = sorted(os.listdir(self.output_tokens_dir))
        if token_files:
            sample_path = os.path.join(self.output_tokens_dir, token_files[0])
            print(f"\n  Пример {token_files[0]}:")
            with open(sample_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[:5]:
                print(f"    {line.strip()}")
            if len(lines) > 5:
                print(f"    ... (ещё {len(lines) - 5} токенов)")

        # Пример файла лемм
        lemma_files = sorted(os.listdir(self.output_lemmas_dir))
        if lemma_files:
            sample_path = os.path.join(self.output_lemmas_dir, lemma_files[0])
            print(f"\n  Пример {lemma_files[0]}:")
            with open(sample_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[:5]:
                print(f"    {line.strip()}")
            if len(lines) > 5:
                print(f"    ... (ещё {len(lines) - 5} лемм)")


def main():
    tokenizer = Tokenizer(
        pages_dir='pages',
        output_tokens_dir='tokens',
        output_lemmas_dir='lemmas',
    )
    tokenizer.run()


if __name__ == "__main__":
    main()
