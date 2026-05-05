from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from hybrid_transfer_metric import (  # noqa: E402
    CLMConfig,
    HybridMetricConfig,
    HybridTransferMetric,
    JMDSConfig,
)


BuiltinLoader = Callable[[], tuple[pd.DataFrame, np.ndarray]]


def print_title(text: str) -> None:
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def print_step(number: int, text: str) -> None:
    print("\n" + "-" * 88)
    print(f"PASO {number}: {text}")
    print("-" * 88)


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def parse_list_argument(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_builtin_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    if name not in loaders:
        raise ValueError(f"Unknown built-in dataset: {name}")

    bunch = loaders[name]()
    feature_names = getattr(bunch, "feature_names", None)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(bunch.data.shape[1])]
    X = pd.DataFrame(bunch.data, columns=[str(col) for col in feature_names])

    target_names = getattr(bunch, "target_names", None)
    if target_names is None:
        y = np.asarray([str(label) for label in bunch.target])
    else:
        y = np.asarray([str(target_names[label]) for label in bunch.target])
    return X, y


def load_csv_dataset(
    dataset: str,
    data_dir: Path,
    target_column: str | None,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(dataset)
    if not path.is_absolute():
        path = data_dir / dataset
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Put your CSV inside {data_dir}/ or pass an absolute path."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The CSV file is empty: {path}")

    if target_column is None:
        target_column = str(df.columns[-1])
        print(f"[info] No target column was provided. Using last column: {target_column}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in {path}.")

    missing_drops = [col for col in drop_columns if col not in df.columns]
    if missing_drops:
        raise ValueError(f"Columns requested in --drop-columns do not exist: {missing_drops}")

    df = df.dropna(subset=[target_column]).copy()
    y = df[target_column].astype(str).to_numpy()
    X = df.drop(columns=[target_column] + drop_columns)
    return X, y


def prepare_features(X_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    X = X_raw.copy()
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric = X[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X.index)
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        fill_value = numeric[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        numeric[col] = numeric[col].fillna(fill_value)

    categorical = pd.DataFrame(index=X.index)
    if categorical_cols:
        categorical = X[categorical_cols].copy().fillna("__missing__").astype(str)
        categorical = pd.get_dummies(categorical, prefix=categorical_cols, dtype=float)

    prepared = pd.concat([numeric.astype(float), categorical.astype(float)], axis=1)
    if prepared.shape[1] == 0:
        raise ValueError("No usable feature columns were found after preprocessing.")

    metadata = {
        "raw_features": int(X_raw.shape[1]),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(categorical_cols)),
        "final_features": int(prepared.shape[1]),
    }
    return prepared, metadata


def apply_class_merge(y: np.ndarray, merge_json: str | None) -> np.ndarray:
    if not merge_json:
        return y.astype(str)

    try:
        spec = json.loads(merge_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--merge-classes must be valid JSON.") from exc

    if not isinstance(spec, dict):
        raise ValueError("--merge-classes must be a JSON object.")

    mapping: dict[str, str] = {}
    for new_label, old_labels in spec.items():
        if not isinstance(old_labels, list):
            raise ValueError("Each value in --merge-classes must be a list.")
        for old_label in old_labels:
            mapping[str(old_label)] = str(new_label)

    return np.asarray([mapping.get(str(label), str(label)) for label in y], dtype=str)


def class_summary(y: np.ndarray) -> pd.DataFrame:
    labels, counts = np.unique(y.astype(str), return_counts=True)
    proportions = counts / counts.sum()
    return pd.DataFrame(
        {
            "label": labels,
            "count": counts,
            "proportion": proportions,
        }
    )


def resolve_target_proportions(
    mode: str,
    y: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    n_classes = len(labels)
    if mode == "natural":
        summary = class_summary(y)
        lookup = dict(zip(summary["label"], summary["proportion"]))
        return np.asarray([lookup[str(label)] for label in labels], dtype=float)

    if mode == "uniform":
        return np.ones(n_classes, dtype=float) / n_classes

    values = np.asarray([float(item) for item in parse_list_argument(mode)], dtype=float)
    if values.shape[0] != n_classes:
        raise ValueError(
            f"Target proportions/counts has {values.shape[0]} values, but the dataset has {n_classes} classes."
        )
    if np.any(values < 0):
        raise ValueError("Target proportions/counts cannot contain negative values.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Target proportions/counts must sum to a positive number.")
    return values / total


def own_reliability_by_original_label(result, y: np.ndarray) -> pd.DataFrame:
    rows = []
    own = result.jmds.own_reliability
    for label in result.labels:
        mask = y.astype(str) == str(label)
        rows.append(
            {
                "label": str(label),
                "count": int(mask.sum()),
                "mean": float(np.mean(own[mask])),
                "min": float(np.min(own[mask])),
                "max": float(np.max(own[mask])),
            }
        )
    return pd.DataFrame(rows)


def print_result_report(result, y: np.ndarray, show_samples: int) -> None:
    print_step(5, "Componente global tipo CLM")
    print(f"G global score: {result.global_score:.6f}")
    metric_rows = []
    for name, metric_result in result.clm.metrics.items():
        metric_rows.append(
            {
                "metric": name,
                "raw": metric_result.raw,
                "null_mean": metric_result.null_mean,
                "null_std": metric_result.null_std,
                "adjusted_0_1": metric_result.adjusted,
            }
        )
    print(pd.DataFrame(metric_rows).round(6).to_string(index=False))

    print_step(6, "Componente local tipo JMDS")
    print("R(i,k) = p_data(i,k) * p_model(i,k)")
    print("Resumen de confianza propia R(i, y_i) por clase original:")
    print(own_reliability_by_original_label(result, y).round(6).to_string(index=False))

    if show_samples > 0:
        sample_count = min(show_samples, result.jmds.reliability.shape[0])
        sample_df = pd.DataFrame(
            result.jmds.reliability[:sample_count],
            columns=[f"R_to_{label}" for label in result.labels],
        )
        sample_df.insert(0, "sample_index", np.arange(sample_count))
        sample_df.insert(1, "original_label", y[:sample_count])
        print("\nPrimeras filas de la matriz R:")
        print(sample_df.round(6).to_string(index=False))

    print_step(7, "Asignacion con restricciones de tamano")
    labels = [str(label) for label in result.labels]
    assignment_prop = pd.DataFrame(
        result.assignment_matrix,
        index=[f"from_{label}" for label in labels],
        columns=[f"to_{label}" for label in labels],
    )
    assignment_count = assignment_prop * len(y)
    print("Matriz de masa reasignada en proporciones del dataset:")
    print(assignment_prop.round(6).to_string())
    print("\nMatriz de masa reasignada aproximada en numero de muestras:")
    print(assignment_count.round(3).to_string())

    print_step(8, "Resultado final")
    print(f"T transfer score : {result.transfer_score:.6f}")
    print(f"G global score   : {result.global_score:.6f}")
    print(f"L local factor   : {result.local_factor:.6f}")
    print(f"C* local cost    : {result.local_cost:.6f}")
    print(f"Moved fraction   : {result.moved_fraction:.6f}")
    print(f"Observed pi      : {format_vector(result.observed_proportions)}")
    print(f"Target q         : {format_vector(result.target_proportions)}")

    print("\nLectura rapida:")
    print("- T cerca de 1: dataset muy transferible bajo esa restriccion.")
    print("- G alto: las etiquetas parecen grupos naturales.")
    print("- L alto: cumplir los tamanos pedidos no rompe mucho la estructura.")
    print("- C* alto: mover muestras cuesta mucho porque eran confiables en su clase original.")
    print("- Moved fraction: cuanta masa del dataset tuvo que cambiar de clase.")


def run_analysis(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    print_title("TESIS.PY - Analisis de transferencia clasificacion -> clustering")
    print(f"Directorio de datos esperado: {data_dir.resolve()}")

    print_step(1, "Carga del dataset")
    builtins = {"iris", "wine", "breast_cancer", "digits"}
    if args.dataset in builtins:
        X_raw, y_raw = load_builtin_dataset(args.dataset)
        print(f"Dataset integrado cargado: {args.dataset}")
    else:
        X_raw, y_raw = load_csv_dataset(
            dataset=args.dataset,
            data_dir=data_dir,
            target_column=args.target_column,
            drop_columns=parse_list_argument(args.drop_columns),
        )
        print(f"CSV cargado: {args.dataset}")

    y = apply_class_merge(y_raw, args.merge_classes)
    print(f"Filas originales: {len(y_raw)}")
    print(f"Columnas de entrada antes de preparar X: {X_raw.shape[1]}")
    print("\nClases despues de posible union de clases:")
    print(class_summary(y).round(6).to_string(index=False))

    print_step(2, "Preparacion de variables X e y")
    X_prepared, feature_metadata = prepare_features(X_raw)
    print(f"Variables originales: {feature_metadata['raw_features']}")
    print(f"Variables numericas: {feature_metadata['numeric_features']}")
    print(f"Variables categoricas codificadas: {feature_metadata['categorical_features']}")
    print(f"Variables finales usadas por la metrica: {feature_metadata['final_features']}")
    print(f"Matriz final X: {X_prepared.shape[0]} filas x {X_prepared.shape[1]} columnas")

    labels = np.unique(y.astype(str))
    target_q = resolve_target_proportions(args.target, y, labels)

    print_step(3, "Restriccion de tamanos objetivo")
    target_table = pd.DataFrame(
        {
            "label_order_used": labels,
            "target_q": target_q,
            "target_count_equivalent": target_q * len(y),
        }
    )
    print("Orden de clases usado para q:")
    print(target_table.round(6).to_string(index=False))
    print("\nNota: si pasas --target 50,100 se normaliza a proporciones.")

    print_step(4, "Ejecucion de la metrica hibrida")
    print(f"Permutaciones CLM: {args.clm_permutations}")
    print(f"Vecinos JMDS: {args.neighbors}")
    print(f"Folds CV modelo: {args.cv_folds}")
    print(f"Estandarizar variables: {args.standardize}")

    metric = HybridTransferMetric(
        HybridMetricConfig(
            clm=CLMConfig(
                n_permutations=args.clm_permutations,
                random_state=args.random_state,
                silhouette_weight=args.silhouette_weight,
                calinski_harabasz_weight=args.calinski_harabasz_weight,
            ),
            jmds=JMDSConfig(
                n_neighbors=args.neighbors,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                neighbor_smoothing=args.neighbor_smoothing,
            ),
            standardize=args.standardize,
        )
    )

    result = metric.evaluate(
        X=X_prepared.to_numpy(dtype=float),
        y=y,
        target_proportions=target_q,
    )
    print_result_report(result, y, show_samples=args.show_reliability_samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza un dataset de clasificacion con la metrica hibrida CLM-JMDS "
            "para estimar su transferencia a clustering con restricciones de tamano."
        )
    )
    parser.add_argument(
        "--dataset",
        default="iris",
        help=(
            "Dataset a usar. Opciones integradas: iris, wine, breast_cancer, digits. "
            "Tambien puedes pasar un CSV dentro de data/, por ejemplo mi_dataset.csv."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Carpeta donde se buscan los CSV. Default: data.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Nombre de la columna etiqueta para CSV. Si se omite, se usa la ultima columna.",
    )
    parser.add_argument(
        "--drop-columns",
        default="",
        help="Columnas a excluir del CSV, separadas por coma.",
    )
    parser.add_argument(
        "--target",
        default="natural",
        help=(
            "Proporciones objetivo. Usa natural, uniform, o valores separados por coma. "
            "Ejemplos: natural | uniform | 0.2,0.4,0.4 | 50,100."
        ),
    )
    parser.add_argument(
        "--merge-classes",
        default=None,
        help=(
            "JSON opcional para unir clases. Ejemplo: "
            "'{\"Setosa\": [\"setosa\"], \"Otra\": [\"versicolor\", \"virginica\"]}'"
        ),
    )
    parser.add_argument("--clm-permutations", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--neighbor-smoothing", type=float, default=0.0)
    parser.add_argument("--silhouette-weight", type=float, default=0.5)
    parser.add_argument("--calinski-harabasz-weight", type=float, default=0.5)
    parser.add_argument("--show-reliability-samples", type=int, default=5)
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estandariza variables antes de calcular distancias/modelos. Default: true.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
