import os
import re
from bs4 import BeautifulSoup
from pymystem3 import Mystem
from collections import defaultdict


# Русские стоп-слова (предлоги, союзы, частицы, местоимения и т.д.)
STOP_WORDS = {
    # Предлоги
    'в', 'на', 'по', 'с', 'со', 'к', 'ко', 'у', 'о', 'об', 'обо', 'от', 'ото',
    'из', 'изо', 'за', 'до', 'для', 'без', 'безо', 'при', 'про', 'через', 'между',
    'над', 'надо', 'под', 'подо', 'перед', 'передо', 'пред', 'предо', 'около',
    'вокруг', 'после', 'среди', 'вместо', 'кроме', 'помимо', 'сквозь', 'вдоль',
    'поперёк', 'поперек', 'ради', 'навстречу', 'вопреки', 'согласно', 'благодаря',
    'вследствие', 'насчёт', 'насчет', 'ввиду', 'наподобие', 'вроде', 'посредством',
    'путём', 'путем',

    # Союзы
    'и', 'а', 'но', 'да', 'или', 'либо', 'ни', 'то', 'не', 'ли', 'бы',
    'что', 'чтобы', 'чтоб', 'как', 'так', 'если', 'когда', 'пока', 'хотя',
    'хоть', 'будто', 'словно', 'точно', 'раз', 'ведь', 'потому', 'поэтому',
    'причём', 'причем', 'притом', 'зато', 'однако', 'также', 'тоже', 'же',
    'только', 'лишь', 'даже', 'именно', 'ибо',

    # Частицы
    'вот', 'вон', 'вовсе', 'разве', 'неужели', 'ну', 'уж',
    'ведь', 'мол', 'дескать', 'якобы', 'аж', 'всё', 'все',

    # Местоимения
    'я', 'мы', 'ты', 'вы', 'он', 'она', 'оно', 'они',
    'мой', 'моя', 'моё', 'мое', 'мои', 'наш', 'наша', 'наше', 'наши',
    'твой', 'твоя', 'твоё', 'твое', 'твои', 'ваш', 'ваша', 'ваше', 'ваши',
    'его', 'её', 'ее', 'их',
    'себя', 'себе', 'собой', 'собою',
    'кто', 'что', 'какой', 'какая', 'какое', 'какие', 'чей', 'чья', 'чьё', 'чье', 'чьи',
    'который', 'которая', 'которое', 'которые', 'которого', 'которой', 'которых',
    'этот', 'эта', 'это', 'эти', 'тот', 'та', 'те',
    'сам', 'сама', 'само', 'сами', 'самый', 'самая', 'самое', 'самые',
    'весь', 'вся', 'всё', 'все', 'каждый', 'каждая', 'каждое', 'каждые',
    'другой', 'другая', 'другое', 'другие', 'иной', 'иная', 'иное', 'иные',
    'меня', 'мне', 'мной', 'мною', 'нас', 'нам', 'нами',
    'тебя', 'тебе', 'тобой', 'тобою', 'вас', 'вам', 'вами',
    'него', 'нему', 'ним', 'нём', 'нем', 'неё', 'нее', 'ней', 'нею',
    'них', 'ними',
    'ему', 'им', 'ей', 'ею',

    # Вспомогательные глаголы / связки
    'быть', 'был', 'была', 'было', 'были', 'есть', 'будет', 'будут',
    'будем', 'будете', 'буду', 'будешь',

    # Наречия-связки
    'уже', 'ещё', 'еще', 'очень', 'более', 'менее', 'тут', 'там', 'здесь',
    'где', 'куда', 'откуда', 'тогда', 'потом', 'затем', 'сейчас', 'теперь',
    'можно', 'нужно', 'надо', 'нельзя',
}

# Минимальная длина токена
MIN_TOKEN_LENGTH = 2


def extract_text_from_html(html_content: str) -> str:
    """Извлекает текстовое содержимое из HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Удаляем скрипты и стили
    for tag in soup.find_all(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    return text


def is_valid_token(token: str) -> bool:
    """Проверяет, является ли токен допустимым"""
    # Только кириллические буквы
    if not re.match(r'^[а-яёА-ЯЁ]+$', token):
        return False

    # Минимальная длина
    if len(token) < MIN_TOKEN_LENGTH:
        return False

    # Не стоп-слово
    if token.lower() in STOP_WORDS:
        return False

    return True


def tokenize_file(filepath: str) -> set:
    """Токенизирует один HTML-файл, возвращает множество валидных токенов"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"  Ошибка чтения {filepath}: {e}")
        return set()

    text = extract_text_from_html(html_content)

    # Разбиваем на слова — берём только последовательности кириллических букв
    raw_tokens = re.findall(r'[а-яёА-ЯЁ]+', text)

    valid_tokens = set()
    for token in raw_tokens:
        token_lower = token.lower()
        if is_valid_token(token_lower):
            valid_tokens.add(token_lower)

    return valid_tokens


def collect_all_tokens(pages_dir: str) -> set:
    """Собирает все уникальные токены из всех файлов в директории"""
    all_tokens = set()

    html_files = sorted([
        f for f in os.listdir(pages_dir)
        if f.endswith('.html')
    ])

    print(f"Найдено {len(html_files)} HTML-файлов в '{pages_dir}'")

    for i, filename in enumerate(html_files):
        filepath = os.path.join(pages_dir, filename)
        tokens = tokenize_file(filepath)
        all_tokens.update(tokens)

        if (i + 1) % 10 == 0 or i == len(html_files) - 1:
            print(f"  Обработано {i + 1}/{len(html_files)} файлов, "
                  f"уникальных токенов: {len(all_tokens)}")

    return all_tokens


def save_tokens(tokens: set, output_path: str):
    """Сохраняет список токенов в файл"""
    sorted_tokens = sorted(tokens)

    with open(output_path, 'w', encoding='utf-8') as f:
        for token in sorted_tokens:
            f.write(token + '\n')

    print(f"Сохранено {len(sorted_tokens)} токенов в '{output_path}'")


def lemmatize_tokens(tokens: set) -> dict:
    """
    Группирует токены по леммам с помощью pymystem3.
    Возвращает dict: лемма -> set(токенов)
    """
    mystem = Mystem()

    lemma_groups = defaultdict(set)
    token_list = sorted(tokens)

    print(f"\nЛемматизация {len(token_list)} токенов...")

    # Обрабатываем батчами для скорости
    batch_size = 500
    for start in range(0, len(token_list), batch_size):
        batch = token_list[start:start + batch_size]
        # pymystem3 принимает текст, разделённый пробелами/переносами
        text = '\n'.join(batch)

        analysis = mystem.analyze(text)

        for item in analysis:
            original = item.get('text', '').strip().lower()
            if not original or not re.match(r'^[а-яёА-ЯЁ]+$', original):
                continue

            analysis_list = item.get('analysis', [])
            if analysis_list:
                lemma = analysis_list[0].get('lex', original).lower()
            else:
                lemma = original

            # Проверяем что лемма тоже валидная
            if not re.match(r'^[а-яё]+$', lemma):
                lemma = original

            # Пропускаем если лемма — стоп-слово
            if lemma in STOP_WORDS:
                continue

            if original in tokens:
                lemma_groups[lemma].add(original)

        processed = min(start + batch_size, len(token_list))
        if processed % 1000 == 0 or processed == len(token_list):
            print(f"  Лемматизировано {processed}/{len(token_list)}, "
                  f"уникальных лемм: {len(lemma_groups)}")

    return dict(lemma_groups)


def save_lemmas(lemma_groups: dict, output_path: str):
    """Сохраняет леммы с токенами в файл"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for lemma in sorted(lemma_groups.keys()):
            tokens_sorted = sorted(lemma_groups[lemma])
            line = lemma + ' ' + ' '.join(tokens_sorted)
            f.write(line + '\n')

    total_tokens = sum(len(v) for v in lemma_groups.values())
    print(f"Сохранено {len(lemma_groups)} лемм ({total_tokens} токенов) в '{output_path}'")


def main():
    pages_dir = 'pages'
    tokens_output = 'tokens.txt'
    lemmas_output = 'lemmas.txt'

    if not os.path.exists(pages_dir):
        print(f"Директория '{pages_dir}' не найдена!")
        print("Сначала запустите краулер для скачивания страниц.")
        return

    # 1. Токенизация
    print("=" * 60)
    print("ЭТАП 1: ТОКЕНИЗАЦИЯ")
    print("=" * 60)

    all_tokens = collect_all_tokens(pages_dir)

    if not all_tokens:
        print("Токены не найдены!")
        return

    save_tokens(all_tokens, tokens_output)

    # Статистика
    print(f"\nСтатистика токенизации:")
    print(f"  Уникальных токенов: {len(all_tokens)}")
    lengths = [len(t) for t in all_tokens]
    print(f"  Мин. длина токена: {min(lengths)}")
    print(f"  Макс. длина токена: {max(lengths)}")
    print(f"  Средняя длина: {sum(lengths) / len(lengths):.1f}")

    # 2. Лемматизация
    print("\n" + "=" * 60)
    print("ЭТАП 2: ЛЕММАТИЗАЦИЯ")
    print("=" * 60)

    lemma_groups = lemmatize_tokens(all_tokens)
    save_lemmas(lemma_groups, lemmas_output)

    # Статистика
    print(f"\nСтатистика лемматизации:")
    print(f"  Уникальных лемм: {len(lemma_groups)}")
    multi_form = {k: v for k, v in lemma_groups.items() if len(v) > 1}
    print(f"  Лемм с несколькими формами: {len(multi_form)}")

    if multi_form:
        print(f"\n  Примеры (до 10):")
        for lemma in list(sorted(multi_form.keys()))[:10]:
            forms = sorted(multi_form[lemma])
            print(f"    {lemma}: {', '.join(forms)}")

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print(f"  Токены: {tokens_output}")
    print(f"  Леммы:  {lemmas_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()