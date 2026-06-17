from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/online_shoppers_intention.csv")
CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")
MODEL_DATA_PATH = Path("outputs/model_data.csv")
SUMMARY_PATH = Path("outputs/preprocessing_summary.txt")
BOOLEAN_COLUMNS = ["Weekend", "Revenue"]

CATEGORICAL_COLUMNS = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

def load_data(path: Path) -> pd.DataFrame:
    """Загрузка данных из CSV файла"""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Предварительная обработка данных: удаление дубликатов и преобразование типов"""
    cleaned_df = df.copy()

    duplicate_count = cleaned_df.duplicated().sum()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    for column in BOOLEAN_COLUMNS:
        cleaned_df[column] = cleaned_df[column].astype(int)
    
    return cleaned_df, duplicate_count
    

def create_model_dataset(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Создание набора данных для модели с использованием one-hot encoding 
    для категориальных признаков"""
    model_df = pd.get_dummies(
        cleaned_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )

    return model_df


def save_results(
        cleaned_df: pd.DataFrame,
        model_df: pd.DataFrame,
        duplicate_count: int,
) -> None:
    """Сохранение очищенного набора данных, набора данных для модели и отчета о предварительной обработке"""
    
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    model_df.to_csv(MODEL_DATA_PATH, index=False)

    summary_text = (
        "Результаты предварительной обработки данных:\n"
        f"Количество строк после обработки: {cleaned_df.shape[0]}\n"
        f"Количество столбцов в очищенном наборе данных: {cleaned_df.shape[1]}\n"
        f"Количество удаленных дубликатов: {duplicate_count}\n"
        f"Количество столбцов в датасете для модели после one-hot encoding: {model_df.shape[1]}\n"
        "\nЦелевая переменная Revenue преобразована в числовой формат:\n"
        "(0 - не совершил покупку, 1 - совершил покупку)\n" 
    )

    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def main() -> None:
    """Основная функция для предварительной обработки данных"""
    df = load_data(RAW_DATA_PATH)

    print("Исходный размер датасета:")
    print(df.shape)

    print("\nКоличество пропусков по столбцам до обработки:")
    print(df.isnull().sum())

    cleaned_df, duplicate_count = preprocess_data(df)
    model_df = create_model_dataset(cleaned_df)

    save_results(cleaned_df, model_df, duplicate_count)

    print("\nКоличество дубликатов:")
    print(duplicate_count)

    print("\nРазмер очищенного датасета:")
    print(cleaned_df.shape)

    print("\nТипы данных после обработки:")
    print(cleaned_df.dtypes)

    print("\nРазмер датасета для модели после one-hot encoding:")
    print(model_df.shape)

    print("\nФайлы сохранены в папке outputs:")
    print(CLEANED_DATA_PATH)
    print(MODEL_DATA_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()