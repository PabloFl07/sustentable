import pickle

import joblib


class TestObject:
    def __reduce__(self):
        return (print, ("Code executed during loading!",))


# Save the object
obj = TestObject()


def test_pickle():

    with open("test.pkl", "wb") as file:
        pickle.dump(obj, file)

    # Load the object
    with open("test.pkl", "rb") as file:
        _ = pickle.load(file)


def test_joblib():

    joblib.dump(obj, "test.joblib")

    # Load the object
    _ = joblib.load("test.joblib")
