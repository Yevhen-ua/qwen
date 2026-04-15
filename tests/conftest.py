import pytest

from llm_connector import LlmConnector


@pytest.fixture
def llmconnector():
    return LlmConnector()


@pytest.fixture
def jsonvalidator():
    def validate(mode, response):
        assert isinstance(response, dict)
        assert response["status"] in {"found", "not_found", "ambiguous"}
        assert isinstance(response["comment"], str)
        return True

    return validate
