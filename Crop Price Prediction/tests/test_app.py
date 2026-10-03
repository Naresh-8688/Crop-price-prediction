import json

import pytest

import app as crop_app


@pytest.fixture
def client():
    crop_app.app.config['TESTING'] = True
    with crop_app.app.test_client() as client:
        yield client


def test_home_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Crop Price Prediction' in response.data


def test_prediction_returns_json(client):
    response = client.post(
        '/predict',
        data=json.dumps({
            'year': 2023,
            'month': 'January',
            'crop_name': 'Groundnut',
        }),
        content_type='application/json',
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['crop_name'] == 'Groundnut'
    assert 'price_per_quintal' in payload
    assert 'price_per_kg' in payload


def test_invalid_crop_rejected(client):
    response = client.post(
        '/predict',
        data=json.dumps({'year': 2023, 'crop_name': 'Tomato'}),
        content_type='application/json',
    )
    assert response.status_code == 400
