import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from simple_repo.simple_sklearn.functions import (
    TrainTestDatasetSplit,
    TrainTestSampleTargetSplit,
)


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


class TestTrainTestDatasetSplit:
    def test_execution(self):
        iris = load_iris(as_frame=True).data

        stt = TrainTestDatasetSplit("stt", test_size=0.33, shuffle=False)
        stt.set_input_value("dataset", iris)

        stt.execute()

        expected_train, expected_test = train_test_split(
            iris, test_size=0.33, shuffle=False
        )

        assert stt.train_dataset.equals(expected_train) and stt.test_dataset.equals(
            expected_test
        )


class TestTrainTestSampleTargetSplit:
    def test_execution(self):
        iris_dt, iris_target = load_iris(return_X_y=True, as_frame=True)

        stt = TrainTestSampleTargetSplit("stt", test_size=0.33, shuffle=False)
        stt.set_input_value("sample_dataset", iris_dt)
        stt.set_input_value("target_dataset", iris_target)

        stt.execute()

        (
            expected_dt_train,
            expected_dt_test,
            expected_target_train,
            expected_target_test,
        ) = train_test_split(iris_dt, iris_target, test_size=0.33, shuffle=False)

        assert (
            stt.sample_train_dataset.equals(expected_dt_train)
            and stt.sample_test_dataset.equals(expected_dt_test)
            and stt.target_train_dataset.equals(expected_target_train)
            and stt.target_test_dataset.equals(expected_target_test)
        )
