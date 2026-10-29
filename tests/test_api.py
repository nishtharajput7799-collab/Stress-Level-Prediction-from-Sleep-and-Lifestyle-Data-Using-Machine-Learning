def test_health_when_model_loaded(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_health_when_model_missing(no_model_client):
    assert no_model_client.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_predict_returns_score_and_category(client, valid_payload):
    r = client.post("/predict/sleep-stress", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 1 <= body["predicted_stress_level"] <= 10
    assert body["category"] in {"Low", "Moderate", "High"}
    assert "diagnosis" in body["disclaimer"].lower()


def test_unknown_occupation_does_not_crash(client, valid_payload):
    valid_payload["occupation"] = "Astronaut"
    assert client.post("/predict/sleep-stress", json=valid_payload).status_code == 200


def test_503_when_model_missing(no_model_client, valid_payload):
    assert no_model_client.post("/predict/sleep-stress", json=valid_payload).status_code == 503


def test_rejects_invalid_gender(client, valid_payload):
    valid_payload["gender"] = "Robot"
    assert client.post("/predict/sleep-stress", json=valid_payload).status_code == 422


def test_rejects_out_of_range_values(client, valid_payload):
    valid_payload["sleep_duration"] = 30
    assert client.post("/predict/sleep-stress", json=valid_payload).status_code == 422


def test_rejects_diastolic_above_systolic(client, valid_payload):
    valid_payload["systolic"], valid_payload["diastolic"] = 80, 120
    assert client.post("/predict/sleep-stress", json=valid_payload).status_code == 422


def test_rejects_missing_field(client, valid_payload):
    del valid_payload["age"]
    assert client.post("/predict/sleep-stress", json=valid_payload).status_code == 422