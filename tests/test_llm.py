import pytest


IMAGE_PATH = "test_images/input/rozetka_50.png"


def test_yes_no_logo_smiley_found(llmconnector, jsonvalidator):
    mode = "yes_no"
    question = 'Logo with smiley face should be in top left corner of image'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_yes_no_logo_sad_not_found(llmconnector, jsonvalidator):
    mode = "yes_no"
    question = 'Logo with sad face should be in top left corner of image'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "not_found"


def test_yes_no_logo_rozetka_text_found(llmconnector, jsonvalidator):
    mode = "yes_no"
    question = 'Logo with text "ROZETKA" should be in top left corner of image'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_yes_no_logo_random_text_not_found(llmconnector, jsonvalidator):
    mode = "yes_no"
    question = 'Logo with text "fjgdfkj" should be in top left corner of image'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "not_found"


def test_yes_no_green_find_button_found(llmconnector, jsonvalidator):
    mode = "yes_no"
    question = 'Is there a green button with text "Find" visible on the screenshot?'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_yes_no_red_find_button_not_found(llmconnector, jsonvalidator):
    mode = "yes_no"
    question = 'Is there a red button with text "Find" visible on the screenshot?'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "not_found"


def test_point_green_find_button(llmconnector, jsonvalidator):
    mode = "point"
    question = 'Point to green button with text "Find"'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_point_rectangular_find_button(llmconnector, jsonvalidator):
    mode = "point"
    question = 'Point to rectangular button with text "Find"'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_point_red_find_button_not_found(llmconnector, jsonvalidator):
    mode = "point"
    question = 'Point to red button with text "Find"'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "not_found"


def test_point_round_find_button_not_found(llmconnector, jsonvalidator):
    mode = "point"
    question = 'Point to round button with text "Find"'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "not_found"


def test_point_upload_button(llmconnector, jsonvalidator):
    mode = "point"
    question = "Return the center point of the Upload button"
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_input_random_text_brand_field(llmconnector, jsonvalidator):
    mode = "input"
    question = 'input some random text with length of 12 symbols in input box below Brand label'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_drag_price_slider_left_handle(llmconnector, jsonvalidator):
    mode = "drag"
    question = 'Drag the left handle of the price slider to the middle of the slider track'
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_value_cart_item_count(llmconnector, jsonvalidator):
    mode = "value"
    question = "how many items shown in shopping cart in top right corner?"
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"


def test_multi_value_popular_brands(llmconnector, jsonvalidator):
    mode = "multi_value"
    question = "What items displayed in Popular brands list?"
    response = llmconnector.request(mode, question, IMAGE_PATH)
    assert jsonvalidator(mode, response)
    assert response["status"] == "found"
