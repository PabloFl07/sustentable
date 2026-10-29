
# tests/test_data_shuffler.py

import pandas as pd
import pytest
from data_shuffler import data_shuffler

# --------------------------------------------------
# FIXTURE: DataFrame de prueba
# --------------------------------------------------

@pytest.fixture
def sample_frame():
    """DataFrame de ejemplo para reutilizar en las pruebas."""
    return pd.DataFrame({
        "id": range(10),
        "valor": [x * 10 for x in range(10)]
    })


# --------------------------------------------------
# 1. PRUEBAS UNITARIAS FUNCIONALES
# --------------------------------------------------

def test_returns_two_dataframes(sample_frame):
    """Debe devolver dos objetos DataFrame."""
    df_test, df_train = data_shuffler(
        sample_frame, seed=42, test_percent=0.3
    )

    assert isinstance(df_test, pd.DataFrame)
    assert isinstance(df_train, pd.DataFrame)


def test_split_sizes(sample_frame):
    """El número de filas debe respetar round(n * porcentaje)."""
    df_test, df_train = data_shuffler(
        sample_frame, seed=42, test_percent=0.3
    )

    assert len(df_test) == round(10 * 0.3)
    assert len(df_train) == 10 - round(10 * 0.3)


def test_preserves_all_rows(sample_frame):
    """No debe perder ni duplicar filas durante la partición."""
    df_test, df_train = data_shuffler(
        sample_frame, seed=42, test_percent=0.3
    )

    combined = pd.concat([df_test, df_train], ignore_index=True)

    pd.testing.assert_frame_equal(
        combined.sort_values("id").reset_index(drop=True),
        sample_frame.sort_values("id").reset_index(drop=True)
    )


def test_preserves_columns(sample_frame):
    """Las columnas deben conservarse en ambos subconjuntos."""
    df_test, df_train = data_shuffler(
        sample_frame, seed=42, test_percent=0.3
    )

    assert list(df_test.columns) == list(sample_frame.columns)
    assert list(df_train.columns) == list(sample_frame.columns)


def test_reproducibility_with_same_seed(sample_frame):
    """La misma semilla debe producir la misma partición."""
    result1 = data_shuffler(sample_frame, 123, 0.3)
    result2 = data_shuffler(sample_frame, 123, 0.3)

    pd.testing.assert_frame_equal(result1[0], result2[0])
    pd.testing.assert_frame_equal(result1[1], result2[1])


def test_different_seed_changes_shuffle():
    """Semillas distintas deben cambiar el orden en este conjunto."""
    frame = pd.DataFrame({"id": range(100)})

    test1, train1 = data_shuffler(frame, 1, 0.5)
    test2, train2 = data_shuffler(frame, 2, 0.5)

    combined1 = pd.concat([test1, train1], ignore_index=True)
    combined2 = pd.concat([test2, train2], ignore_index=True)

    assert not combined1.equals(combined2)


def test_returns_test_before_train(sample_frame):
    """El primer resultado corresponde al test y el segundo al train."""
    df_test, df_train = data_shuffler(
        sample_frame, seed=42, test_percent=0.2
    )

    assert len(df_test) == 2
    assert len(df_train) == 8


# --------------------------------------------------
# 2. PRUEBAS DE CAJA BLANCA: RAMAS CONDICIONALES
# --------------------------------------------------

@pytest.mark.parametrize("percentage", [-0.1, 1.1, -1.0, 2.0])
def test_invalid_percentage_raises_value_error(
    sample_frame, percentage
):
    """Rama: porcentaje fuera del intervalo permitido."""
    with pytest.raises(
        ValueError,
        match="test_percent must be between 0 and 1"
    ):
        data_shuffler(sample_frame, 42, percentage)


def test_frame_type_validation():
    """Rama: frame no es exactamente un DataFrame."""
    with pytest.raises(
        TypeError,
        match="frame must be a Pandas Dataframe"
    ):
        data_shuffler([1, 2, 3], 42, 0.3)


def test_dataframe_subclass_is_rejected():
    """Comprueba el uso de type(frame) is pd.DataFrame."""
    class CustomDataFrame(pd.DataFrame):
        pass

    frame = CustomDataFrame({"id": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="frame must be a Pandas Dataframe"
    ):
        data_shuffler(frame, 42, 0.3)



@pytest.mark.parametrize("percentage", [0, 1])
def test_integer_percentage_is_rejected(sample_frame, percentage):
    """Los enteros 0 y 1 pasan el rango, pero no el tipo."""
    with pytest.raises(
        TypeError,
        match="test_percent must be a float"
    ):
        data_shuffler(sample_frame, 42, percentage)


@pytest.mark.parametrize("percentage", ["0.3", None])
def test_non_numeric_percentage_raises_type_error(
    sample_frame, percentage
):
    """La comparación del rango falla antes de validar el tipo."""
    with pytest.raises(TypeError):
        data_shuffler(sample_frame, 42, percentage)


@pytest.mark.parametrize("seed", ["42", 42.0, None, True])
def test_seed_must_be_int(sample_frame, seed):
    """Rama: la semilla no tiene el tipo int exacto."""
    with pytest.raises(
        TypeError,
        match="seed must be a int"
    ):
        data_shuffler(sample_frame, seed, 0.3)


# --------------------------------------------------
# 3. PRUEBAS DE LÍMITES
# --------------------------------------------------

def test_zero_test_percentage(sample_frame):
    """Con 0.0, todas las filas deben ir a train."""
    df_test, df_train = data_shuffler(
        sample_frame, 42, 0.0
    )

    assert df_test.empty
    assert len(df_train) == len(sample_frame)


def test_full_test_percentage(sample_frame):
    """Con 1.0, todas las filas deben ir a test."""
    df_test, df_train = data_shuffler(
        sample_frame, 42, 1.0
    )

    assert len(df_test) == len(sample_frame)
    assert df_train.empty


def test_empty_dataframe():
    """Un DataFrame vacío debe poder procesarse."""
    frame = pd.DataFrame(columns=["id", "valor"])

    df_test, df_train = data_shuffler(
        frame, 42, 0.5
    )

    assert df_test.empty
    assert df_train.empty
    assert list(df_test.columns) == ["id", "valor"]
    assert list(df_train.columns) == ["id", "valor"]


def test_single_row_dataframe():
    """Con una fila y porcentaje 0.5, round(0.5) da 0."""
    frame = pd.DataFrame({"id": [1]})

    df_test, df_train = data_shuffler(
        frame, 42, 0.5
    )

    assert len(df_test) == 0
    assert len(df_train) == 1


# --------------------------------------------------
# 4. PRUEBAS DE COPIA E ÍNDICES
# --------------------------------------------------

def test_indices_are_reset(sample_frame):
    """Los índices de ambos resultados deben comenzar en cero."""
    df_test, df_train = data_shuffler(
        sample_frame, 42, 0.3
    )

    assert list(df_test.index) == list(range(len(df_test)))
    assert list(df_train.index) == list(range(len(df_train)))


def test_input_dataframe_is_not_modified(sample_frame):
    """La función no debe modificar el DataFrame original."""
    original = sample_frame.copy(deep=True)

    data_shuffler(sample_frame, 42, 0.3)

    pd.testing.assert_frame_equal(sample_frame, original)


def test_results_are_independent_copies(sample_frame):
    """Modificar los resultados no debe alterar el DataFrame original."""
    df_test, df_train = data_shuffler(
        sample_frame, 42, 0.3
    )

    original = sample_frame.copy(deep=True)

    if not df_test.empty:
        df_test.iloc[0, df_test.columns.get_loc("valor")] = -999

    if not df_train.empty:
        df_train.iloc[0, df_train.columns.get_loc("valor")] = -999

    pd.testing.assert_frame_equal(sample_frame, original)
