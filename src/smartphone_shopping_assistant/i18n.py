"""English presentation text; catalog facts and user messages are never translated."""

NODE_LABELS = {
    "parse_validate": "Разбор запроса",
    "clarify": "Уточнение",
    "compare": "Сравнение",
    "prepare_query": "Подготовка запроса",
    "retrieve": "Поиск кандидатов",
    "select": "Отбор кандидатов",
    "evaluate": "Оценка выдачи",
    "rewrite": "Переформулировка",
    "finalize": "Проверка и ответ",
}

STOP_LABELS = {
    "sufficient": "Выдача достаточна",
    "retry_complete": "Повторный поиск выполнен",
    "rewrite_rejected": "Нет подходящей новой формулировки",
    "sql_empty": "Каталог пуст по заданным фильтрам",
    "clarification": "Нужно уточнение",
    "comparison": "Сравнение выполнено",
    "empty": "Выдача пуста",
    "search_error": "Ошибка поиска",
    "model_unavailable": "Сбой локальной модели",
    "invalid_model": "Некорректный ответ модели",
    "deadline": "Истекло время поиска",
    "call_limit": "Лимит вызовов Ollama",
    "graph_limit": "Лимит шагов графа",
}

ENGLISH = {
    "Как выполнен поиск": "Search execution",
    "Попытки: {attempts} · Вызовы Ollama: {calls} · Кандидаты: {candidates}": (
        "Search attempts: {attempts} · Ollama calls: {calls} · Candidates: {candidates}"
    ),
    "Остановка": "Stop reason",
    "Запрос": "Query",
    "ошибка": "failed",
    "мс": "ms",
    "Локальная модель недоступна. Повторите запрос позже.": (
        "The local model is unavailable. Please try again later."
    ),
    "Локальная модель вернула некорректный ответ.": "The local model returned an invalid response.",
    "Время поиска истекло. Повторите запрос.": "The search timed out. Please try again.",
    "Достигнут лимит обращений к локальной модели.": "The local model call limit was reached.",
    "Достигнут лимит шагов поиска.": "The search step limit was reached.",
    "Разбор запроса": "Interpret request",
    "Уточнение": "Clarify",
    "Сравнение": "Compare",
    "Подготовка запроса": "Prepare search query",
    "Поиск кандидатов": "Retrieve candidates",
    "Отбор кандидатов": "Select candidates",
    "Оценка выдачи": "Evaluate results",
    "Переформулировка": "Rewrite query",
    "Проверка и ответ": "Verify and respond",
    "Выдача достаточна": "Results are sufficient",
    "Повторный поиск выполнен": "Retry completed",
    "Нет подходящей новой формулировки": "No usable rewrite",
    "Каталог пуст по заданным фильтрам": "No catalog products match the filters",
    "Нужно уточнение": "Clarification needed",
    "Сравнение выполнено": "Comparison completed",
    "Выдача пуста": "No results",
    "Ошибка поиска": "Search failed",
    "Сбой локальной модели": "Local model unavailable",
    "Некорректный ответ модели": "Invalid model response",
    "Истекло время поиска": "Search timed out",
    "Лимит вызовов Ollama": "Ollama call limit",
    "Лимит шагов графа": "Graph step limit",
    "Подбор смартфона": "Smartphone shopping assistant",
    "📱 Подбор смартфона": "📱 Smartphone shopping assistant",
    "Локальная модель: {model}. Поиск по каталогу, до 5 товаров в ответе.": (
        "Local model: {model}. Catalog search, up to 5 products per response."
    ),
    "* USD — валюта проекта. Валюта исходного каталога не подтверждена; пересчёта нет.": (
        "* USD is a project setting. The source currency is unconfirmed; no conversion is applied."
    ),
    "Фильтры": "Filters",
    "Очистить фильтры": "Clear filters",
    "Бренды через запятую": "Brands, separated by commas",
    "Цена от": "Minimum price",
    "Цена до": "Maximum price",
    "ОЗУ от, ГБ": "Minimum RAM, GB",
    "ОЗУ до, ГБ": "Maximum RAM, GB",
    "Память от, ГБ": "Minimum storage, GB",
    "Память до, ГБ": "Maximum storage, GB",
    "Без блокировки оператора": "Carrier unlocked",
    "Любой": "Any",
    "Да": "Yes",
    "Нет": "No",
    "Применить фильтры": "Apply filters",
    "Применено": "Applied filters",
    "Показать дешевле": "Show cheaper",
    "Сравнить первые два": "Compare the first two",
    "Покажи дешевле первого": "Show cheaper than the first",
    "Сравни первые два телефона": "Compare the first two phones",
    "Например: Samsung с 8 ГБ ОЗУ": "For example: Samsung with 8 GB RAM",
    "История диалога — предыдущие ответы": "Conversation history — previous responses",
    "Подбираем смартфоны…": "Finding smartphones…",
    "Укажите целые ГБ и числовую цену.": "Enter whole GB values and a numeric price.",
    "Поиск недоступен.": "Search is unavailable.",
    "Уточните запрос.": "Please clarify your request.",
    "Ничего не найдено.": "No products found.",
    "ГБ": "GB",
    "ОЗУ": "RAM",
    "Память": "Storage",
    "Разблокирован": "Unlocked",
    "Привязан к оператору": "Carrier locked",
    "Характеристика": "Feature",
    "Модель": "Model",
    "Бренд": "Brand",
    "Бренды": "Brands",
    "Цвет": "Color",
    "Без блокировки": "Carrier unlocked",
    "Цена": "Price",
    "Цена*": "Price*",
    "от": "from",
    "до": "up to",
    "Фильтры не заданы": "No filters applied",
    "Уточните запрос (не более 500 символов).": "Please enter a request of up to 500 characters.",
    "Эта характеристика или конвертация валюты недоступна.": (
        "This specification or currency conversion is unavailable."
    ),
    "Введите запрос или измените фильтры.": "Enter a request or change the filters.",
    "Уточните условия поиска.": "Please clarify your search criteria.",
    "Укажите две позиции последней выдачи.": "Choose two positions from the last results.",
    "Один из товаров больше недоступен.": "One of the products is no longer available.",
    "Сравнение по текущим данным каталога.": "Comparison using current catalog records.",
    "Уточните условие по бренду.": "Please clarify the brand filter.",
    "Сначала выполните поиск для сравнения цены.": "Search first to establish a reference price.",
    "Предыдущий товар больше недоступен.": "The previous product is no longer available.",
    "Уточните условие цены.": "Please clarify the price filter.",
    "Нет товаров дешевле при текущем минимуме цены.": (
        "No cheaper products are available with the current minimum price."
    ),
    "По заданным условиям товары не найдены.": "No products match these filters.",
    "Подходящие товары не выбраны.": "No suitable products were selected.",
    "Товары из каталога; валюта источника не подтверждена, пересчёта нет.": (
        "Catalog products; source currency unconfirmed, no conversion applied."
    ),
    "Поиск сейчас недоступен. Повторите запрос позже.": (
        "Search is currently unavailable. Please try again later."
    ),
}


def translate(text: str, language: str) -> str:
    """Translate known presentation strings, preserving all other content verbatim."""
    if language != "en":
        return text
    prefix = "Уточните противоречивые условия: "
    if text.startswith(prefix):
        field = text.removeprefix(prefix).removesuffix(".")
        label = {
            "brands": "brand",
            "price": "price",
            "ram_gb": "RAM",
            "storage_gb": "storage",
            "carrier_unlocked": "carrier lock",
        }.get(field, field)
        return f"Please clarify conflicting {label} filters."
    return ENGLISH.get(text, text)
