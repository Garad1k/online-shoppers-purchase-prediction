from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/online_shoppers_intention.csv")

def load_data(path: Path) -> pd.DataFrame:
    """Загрузка данных из CSV файла"""
    return pd.read_csv(path)

def main() -> None:
    """Запуск первоначальной проверки набора данных"""
    df = load_data(DATA_PATH)
    print("Первые 5 строк датасета:")
    print(df.head())
    print("\nРАсмер датасета:")
    print(df.shape)
    print("\nИнформация о датасете:")
    print(df.info())
    print("\nКоличество пропусков по столбцам:")
    print(df.isna().sum())
    print("\nРаспределение целевой переменной Revenue:")
    print(df["Revenue"].value_counts())

if __name__ == "__main__":
    main()