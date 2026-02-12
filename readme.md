# oip_4k_11-200

## Ветров Владимир, гр. 11-204, ИТИС КФУ

---

### Инструкция по запуску

---

#### Установка окружения

- `python3 -m venv venv`
- `pip install -r requirements.txt`

---

## Задание 1.

#### Запуск краулера

- `python3 src/crawler/crawler.py`

Выходные файлы: `pages/*`, `index.txt`

---

## Задание 2.

#### Запуск токенайзера

- `python3 src/tokenizer/tokenizer.py`

Выходные файлы: `lemmas.txt`, `tokens.txt`

---

## Задание 3.

### Запуск булева поиска

- `python3 src/inverted_index_builder.py`

Выходные файлы: `inverted_index.txt`, `inverted_index.json`

**Есть возможность вводить сложный запрос: `(Клеопатра AND Цезарь) OR (Антоний AND Цицерон) OR Помпей`**

**На выходе ранжированный список ссылок, где есть вхождения по запросу**

---

## Задание 4.

### Запуск расчёта tf-idf

- `python3 src/tf-idf-builder/builder.py`

Выходные файлы: `tfidf_lemmas/*`, `tfidf_tokens/*`

---

## Задание 5.