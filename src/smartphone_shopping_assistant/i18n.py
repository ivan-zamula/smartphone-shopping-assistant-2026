"""English presentation text; catalog facts and user messages are never translated."""

ENGLISH = {
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
