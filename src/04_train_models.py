from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MODEL_DATA_PATH = Path("outputs/model_data.csv")
FIGURES_DIR = Path("outputs/figures")
METRICS_PATH = Path("outputs/model_metrics.csv")
REPORT_PATH = Path("outputs/classification_report.txt")
BEST_MODEL_PATH = Path("outputs/best_model.pkl")
MODEL_SUMMARY_PATH = Path("outputs/model_summary.txt")
RF_IMPORTANCE_PATH = Path("outputs/random_forest_feature_importance.csv")
LR_COEFFICIENTS_PATH = Path("outputs/logistic_regression_coefficients.csv")

TARGET_COLUMN = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Загрузка подготовленного датасета для машинного обучения."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Разделение данных на признаки и целевую переменную."""
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделение данных на обучающую и тестовую выборки."""
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def create_models() -> dict:
    """Создание набора моделей классификации."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(
    model,
    model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Оценка качества модели классификации."""
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def train_and_evaluate_models(
    models: dict,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """Обучение моделей и расчет метрик качества."""
    trained_models = {}
    metrics = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[model_name] = model

        model_metrics = evaluate_model(
            model,
            model_name,
            x_test,
            y_test,
        )
        metrics.append(model_metrics)

    metrics_df = pd.DataFrame(metrics).sort_values(
        "f1_score",
        ascending=False,
    )

    return trained_models, metrics_df


def get_best_model_name(metrics_df: pd.DataFrame) -> str:
    """Определение лучшей модели по F1-score."""
    return metrics_df.iloc[0]["model"]


def save_metrics(metrics_df: pd.DataFrame) -> None:
    """Сохранение таблицы метрик качества моделей."""
    metrics_df.to_csv(METRICS_PATH, index=False)


def save_best_model(model) -> None:
    """Сохранение лучшей модели в файл."""
    joblib.dump(model, BEST_MODEL_PATH)


def save_classification_report(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
) -> None:
    """Сохранение classification report для лучшей модели."""
    y_pred = model.predict(x_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Без покупки", "С покупкой"],
        zero_division=0,
    )

    text = (
        f"Classification report для модели: {best_model_name}\n"
        "====================================================\n\n"
        f"{report}"
    )

    REPORT_PATH.write_text(text, encoding="utf-8")


def save_metrics_comparison_plot(metrics_df: pd.DataFrame) -> None:
    """Сохранение графика сравнения качества моделей."""
    plot_df = metrics_df.set_index("model")[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]

    plt.figure(figsize=(11, 6))
    plot_df.plot(kind="bar", figsize=(11, 6))
    plt.title("Сравнение качества моделей классификации")
    plt.xlabel("Модель")
    plt.ylabel("Значение метрики")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Метрика")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_model_metrics_comparison.png", dpi=300)
    plt.close()


def save_confusion_matrix_plot(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
) -> None:
    """Сохранение матрицы ошибок лучшей модели."""
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Без покупки", "С покупкой"],
        yticklabels=["Без покупки", "С покупкой"],
    )
    plt.title(f"Матрица ошибок: {best_model_name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_confusion_matrix_best_model.png", dpi=300)
    plt.close()


def save_roc_curve_plot(
    trained_models: dict,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Сохранение ROC-кривых для обученных моделей."""
    plt.figure(figsize=(8, 6))

    for model_name, model in trained_models.items():
        y_score = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_value = roc_auc_score(y_test, y_score)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (ROC-AUC = {auc_value:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Случайная модель")
    plt.title("ROC-кривые моделей")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_roc_curves.png", dpi=300)
    plt.close()


def save_random_forest_feature_importance(
    model,
    feature_names: list,
) -> None:
    """Сохранение важности признаков Random Forest."""
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(RF_IMPORTANCE_PATH, index=False)

    top_features = importance_df.head(15).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Топ-15 признаков по важности в Random Forest")
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "11_random_forest_feature_importance.png",
        dpi=300,
    )
    plt.close()


def save_logistic_regression_coefficients(
    model,
    feature_names: list,
) -> None:
    """Сохранение коэффициентов Logistic Regression."""
    logistic_model = model.named_steps["model"]

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": logistic_model.coef_[0],
        }
    )

    coefficients_df["abs_coefficient"] = coefficients_df[
        "coefficient"
    ].abs()

    coefficients_df = coefficients_df.sort_values(
        "abs_coefficient",
        ascending=False,
    )

    coefficients_df.to_csv(LR_COEFFICIENTS_PATH, index=False)

    top_coefficients = coefficients_df.head(15).sort_values("coefficient")

    plt.figure(figsize=(10, 7))
    plt.barh(top_coefficients["feature"], top_coefficients["coefficient"])
    plt.title("Топ-15 коэффициентов Logistic Regression")
    plt.xlabel("Значение коэффициента")
    plt.ylabel("Признак")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "12_logistic_regression_coefficients.png",
        dpi=300,
    )
    plt.close()


def save_model_summary(
    metrics_df: pd.DataFrame,
    best_model_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Сохранение текстовой сводки по моделированию."""
    best_metrics = metrics_df.iloc[0]

    summary_text = (
        "Результаты построения и оценки моделей\n"
        "======================================\n\n"
        f"Размер обучающей выборки: {x_train.shape[0]} строк\n"
        f"Размер тестовой выборки: {x_test.shape[0]} строк\n"
        f"Количество признаков: {x_train.shape[1]}\n"
        f"Доля покупок в обучающей выборке: {y_train.mean():.2%}\n"
        f"Доля покупок в тестовой выборке: {y_test.mean():.2%}\n\n"
        f"Лучшая модель по F1-score: {best_model_name}\n\n"
        "Метрики лучшей модели:\n"
        f"Accuracy: {best_metrics['accuracy']:.4f}\n"
        f"Precision: {best_metrics['precision']:.4f}\n"
        f"Recall: {best_metrics['recall']:.4f}\n"
        f"F1-score: {best_metrics['f1_score']:.4f}\n"
        f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n\n"
        "Все метрики сохранены в файле model_metrics.csv.\n"
        "Графики оценки модели сохранены в папке outputs/figures.\n"
    )

    MODEL_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def main() -> None:
    """Запуск этапа обучения и оценки моделей."""
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(MODEL_DATA_PATH)

    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    models = create_models()

    trained_models, metrics_df = train_and_evaluate_models(
        models,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    best_model_name = get_best_model_name(metrics_df)
    best_model = trained_models[best_model_name]

    save_metrics(metrics_df)
    save_best_model(best_model)
    save_classification_report(
        best_model,
        x_test,
        y_test,
        best_model_name,
    )

    save_metrics_comparison_plot(metrics_df)
    save_confusion_matrix_plot(
        best_model,
        x_test,
        y_test,
        best_model_name,
    )
    save_roc_curve_plot(trained_models, x_test, y_test)

    save_random_forest_feature_importance(
        trained_models["Random Forest"],
        x.columns.tolist(),
    )
    save_logistic_regression_coefficients(
        trained_models["Logistic Regression"],
        x.columns.tolist(),
    )

    save_model_summary(
        metrics_df,
        best_model_name,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    print("Модели обучены и оценены.")
    print("\nМетрики моделей:")
    print(metrics_df)
    print(f"\nЛучшая модель по F1-score: {best_model_name}")
    print("\nФайлы с результатами сохранены в папке outputs.")


if __name__ == "__main__":
    main()