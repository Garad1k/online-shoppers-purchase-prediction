from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")
FIGURES_DIR = Path("outputs/figures")
EDA_SUMMARY_PATH = Path("outputs/eda_summary.txt")
MONTHLY_CONVERSION_PATH = Path("outputs/monthly_conversion.csv")
VISITOR_TYPE_CONVERSION_PATH = Path("outputs/visitor_type_conversion.csv")
CORRELATION_PATH = Path("outputs/correlation_with_revenue.csv")

TARGET_COLUMN = "Revenue"


def load_data(path: Path) -> pd.DataFrame:
    """Загрузка очищенного датасета из CSV-файла."""
    return pd.read_csv(path)


def prepare_output_directory() -> None:
    """Создание папки для сохранения графиков."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_summary_statistics(df: pd.DataFrame) -> dict:
    """Расчет основных показателей набора данных."""
    total_sessions = len(df)
    purchase_sessions = int(df[TARGET_COLUMN].sum())
    no_purchase_sessions = total_sessions - purchase_sessions
    conversion_rate = purchase_sessions / total_sessions

    return {
        "total_sessions": total_sessions,
        "purchase_sessions": purchase_sessions,
        "no_purchase_sessions": no_purchase_sessions,
        "conversion_rate": conversion_rate,
        "columns_count": df.shape[1],
    }


def calculate_monthly_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Расчет конверсии по месяцам."""
    month_order = [
        "Feb",
        "Mar",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    monthly_conversion = (
        df.groupby("Month", observed=False)
        .agg(
            sessions=(TARGET_COLUMN, "count"),
            purchases=(TARGET_COLUMN, "sum"),
            conversion_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )

    monthly_conversion["Month"] = pd.Categorical(
        monthly_conversion["Month"],
        categories=month_order,
        ordered=True,
    )

    monthly_conversion = monthly_conversion.sort_values("Month")
    monthly_conversion["conversion_rate_percent"] = (
        monthly_conversion["conversion_rate"] * 100
    ).round(2)

    return monthly_conversion


def calculate_visitor_type_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Расчет конверсии по типу посетителя."""
    visitor_conversion = (
        df.groupby("VisitorType", observed=False)
        .agg(
            sessions=(TARGET_COLUMN, "count"),
            purchases=(TARGET_COLUMN, "sum"),
            conversion_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )

    visitor_conversion["conversion_rate_percent"] = (
        visitor_conversion["conversion_rate"] * 100
    ).round(2)

    return visitor_conversion.sort_values(
        "conversion_rate",
        ascending=False,
    )


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Расчет корреляции числовых признаков с целевой переменной."""
    numeric_df = df.select_dtypes(include=["number"])
    correlations = (
        numeric_df.corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=abs, ascending=False)
        .reset_index()
    )

    correlations.columns = ["feature", "correlation_with_revenue"]

    return correlations


def save_revenue_distribution_plot(df: pd.DataFrame) -> None:
    """Сохранение графика распределения целевой переменной Revenue."""
    revenue_counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Без покупки", "С покупкой"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, revenue_counts.values, label="Количество сессий")
    plt.title("Распределение пользовательских сессий по факту покупки")
    plt.xlabel("Результат пользовательской сессии")
    plt.ylabel("Количество сессий")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_revenue_distribution.png", dpi=300)
    plt.close()


def save_visitor_type_conversion_plot(
    visitor_conversion: pd.DataFrame,
) -> None:
    """Сохранение графика конверсии по типу посетителя."""
    plt.figure(figsize=(9, 5))
    plt.bar(
        visitor_conversion["VisitorType"],
        visitor_conversion["conversion_rate_percent"],
        label="Конверсия",
    )
    plt.title("Конверсия по типу посетителя")
    plt.xlabel("Тип посетителя")
    plt.ylabel("Конверсия, %")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_visitor_type_conversion.png", dpi=300)
    plt.close()


def save_monthly_conversion_plot(monthly_conversion: pd.DataFrame) -> None:
    """Сохранение графика конверсии по месяцам."""
    plt.figure(figsize=(10, 5))
    plt.plot(
        monthly_conversion["Month"].astype(str),
        monthly_conversion["conversion_rate_percent"],
        marker="o",
        label="Конверсия",
    )
    plt.title("Конверсия по месяцам")
    plt.xlabel("Месяц")
    plt.ylabel("Конверсия, %")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_monthly_conversion.png", dpi=300)
    plt.close()


def save_page_values_boxplot(df: pd.DataFrame) -> None:
    """Сохранение boxplot для признака PageValues по факту покупки."""
    plot_df = df.copy()
    plot_df["PurchaseStatus"] = plot_df[TARGET_COLUMN].map(
        {0: "Без покупки", 1: "С покупкой"}
    )

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=plot_df, x="PurchaseStatus", y="PageValues")
    plt.title("Распределение PageValues по факту покупки")
    plt.xlabel("Результат пользовательской сессии")
    plt.ylabel("PageValues")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_page_values_by_revenue.png", dpi=300)
    plt.close()


def save_product_duration_boxplot(df: pd.DataFrame) -> None:
    """Сохранение boxplot для ProductRelated_Duration по факту покупки."""
    plot_df = df.copy()
    plot_df["PurchaseStatus"] = plot_df[TARGET_COLUMN].map(
        {0: "Без покупки", 1: "С покупкой"}
    )

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=plot_df,
        x="PurchaseStatus",
        y="ProductRelated_Duration",
    )
    plt.title("Время просмотра товарных страниц по факту покупки")
    plt.xlabel("Результат пользовательской сессии")
    plt.ylabel("Длительность просмотра товарных страниц")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "05_product_related_duration_by_revenue.png",
        dpi=300,
    )
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Сохранение тепловой карты корреляций числовых признаков."""
    numeric_df = df.select_dtypes(include=["number"])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
    )
    plt.title("Корреляционная матрица числовых признаков")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_correlation_matrix.png", dpi=300)
    plt.close()


def save_top_correlations_plot(correlations: pd.DataFrame) -> None:
    """Сохранение графика топ-10 признаков по связи с Revenue."""
    top_correlations = correlations.head(10).sort_values(
        "correlation_with_revenue"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_correlations["feature"],
        top_correlations["correlation_with_revenue"],
        label="Корреляция с Revenue",
    )
    plt.title("Топ-10 признаков по связи с Revenue")
    plt.xlabel("Коэффициент корреляции с Revenue")
    plt.ylabel("Признак")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_top_correlations_with_revenue.png", dpi=300)
    plt.close()


def save_tables(
    monthly_conversion: pd.DataFrame,
    visitor_conversion: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    """Сохранение аналитических таблиц в CSV-файлы."""
    monthly_conversion.to_csv(MONTHLY_CONVERSION_PATH, index=False)
    visitor_conversion.to_csv(VISITOR_TYPE_CONVERSION_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)


def save_eda_summary(
    metrics: dict,
    monthly_conversion: pd.DataFrame,
    visitor_conversion: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    """Сохранение текстовой сводки по исследовательскому анализу."""
    best_month = monthly_conversion.sort_values(
        "conversion_rate",
        ascending=False,
    ).iloc[0]

    best_visitor_type = visitor_conversion.sort_values(
        "conversion_rate",
        ascending=False,
    ).iloc[0]

    top_feature = correlations.iloc[0]

    summary_text = (
        "Результаты исследовательского анализа данных\n"
        "============================================\n\n"
        f"Количество пользовательских сессий: "
        f"{metrics['total_sessions']}\n"
        f"Количество сессий без покупки: "
        f"{metrics['no_purchase_sessions']}\n"
        f"Количество сессий с покупкой: "
        f"{metrics['purchase_sessions']}\n"
        f"Общая конверсия: "
        f"{metrics['conversion_rate']:.2%}\n"
        f"Количество признаков в очищенном датасете: "
        f"{metrics['columns_count']}\n\n"
        f"Месяц с наибольшей конверсией: "
        f"{best_month['Month']} "
        f"({best_month['conversion_rate_percent']}%).\n"
        f"Тип посетителя с наибольшей конверсией: "
        f"{best_visitor_type['VisitorType']} "
        f"({best_visitor_type['conversion_rate_percent']}%).\n"
        f"Наиболее связанный с покупкой числовой признак: "
        f"{top_feature['feature']} "
        f"(корреляция = "
        f"{top_feature['correlation_with_revenue']:.4f}).\n\n"
        "Сохраненные графики:\n"
        "1. 01_revenue_distribution.png\n"
        "2. 02_visitor_type_conversion.png\n"
        "3. 03_monthly_conversion.png\n"
        "4. 04_page_values_by_revenue.png\n"
        "5. 05_product_related_duration_by_revenue.png\n"
        "6. 06_correlation_matrix.png\n"
        "7. 07_top_correlations_with_revenue.png\n"
    )

    EDA_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def save_product_duration_limited_boxplot(df: pd.DataFrame) -> None:
    """Сохранение boxplot для ProductRelated_Duration с ограничением выбросов."""
    plot_df = df.copy()
    plot_df["PurchaseStatus"] = plot_df[TARGET_COLUMN].map(
        {0: "Без покупки", 1: "С покупкой"}
    )

    upper_limit = plot_df["ProductRelated_Duration"].quantile(0.95)

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=plot_df,
        x="PurchaseStatus",
        y="ProductRelated_Duration",
    )
    plt.ylim(0, upper_limit)
    plt.title("Время просмотра товарных страниц по факту покупки")
    plt.xlabel("Результат пользовательской сессии")
    plt.ylabel("Длительность просмотра товарных страниц")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "05b_product_related_duration_limited.png",
        dpi=300,
    )
    plt.close()


def main() -> None:
    """Запуск исследовательского анализа данных."""
    sns.set_theme(style="whitegrid")

    prepare_output_directory()

    df = load_data(CLEANED_DATA_PATH)

    metrics = calculate_summary_statistics(df)
    monthly_conversion = calculate_monthly_conversion(df)
    visitor_conversion = calculate_visitor_type_conversion(df)
    correlations = calculate_correlations(df)

    save_revenue_distribution_plot(df)
    save_visitor_type_conversion_plot(visitor_conversion)
    save_monthly_conversion_plot(monthly_conversion)
    save_page_values_boxplot(df)
    save_product_duration_boxplot(df)
    save_product_duration_limited_boxplot(df)
    save_correlation_heatmap(df)
    save_top_correlations_plot(correlations)

    save_tables(monthly_conversion, visitor_conversion, correlations)
    save_eda_summary(
        metrics,
        monthly_conversion,
        visitor_conversion,
        correlations,
    )

    print("Исследовательский анализ выполнен.")
    print(f"Количество сессий: {metrics['total_sessions']}")
    print(f"Количество покупок: {metrics['purchase_sessions']}")
    print(f"Общая конверсия: {metrics['conversion_rate']:.2%}")
    print("\nФайлы с графиками сохранены в папке outputs/figures.")
    print("Сводка сохранена в файле outputs/eda_summary.txt.")


if __name__ == "__main__":
    main()