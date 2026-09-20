from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# -----------------------------------------------------------------------------
# Cas nominaux : entrées valides et représentatives
# -----------------------------------------------------------------------------
def test_predict_success():
    response = client.post("/predict", json={
    "features": [3.5, 1.2, 4.9]
    })
    assert response.status_code == 200
    assert response.json() == {"predictions": [7.0, 2.4, 9.8]}

def test_predict_incorrect_expectation():
    response = client.post("/predict", json={
        "features": [3.5, 1.2, 4.9]
    })
    assert response.status_code == 200
    # On vérifie que la réponse de l'API N'EST PAS égale au résultat volontairement faux
    assert response.json() != {"predictions": [0.0, 0.0, 0.0]}

    # Nouveau test ajouté ici :
def test_predict_with_custom_values():
    response = client.post("/predict", json={
        "features": [1.0, 2.0, 3.0]
    })
    assert response.status_code == 200
    assert response.json() == {"predictions": [2.0, 4.0, 6.0]}  # Ajustez le résultat attendu selon votre modèle

# -----------------------------------------------------------------------------
# Cas invalides : données ne respectant pas les préconditions attendues
# -----------------------------------------------------------------------------
def test_predict_unprocessable_entity():
    response = client.post("/predict", json={
    "feature1": 3.5,
    "feature2": 1.2,
    "feature3": 4.9
    })
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Field required"


def test_predict_invalid_json_format():
    # Envoi d'une chaîne malformée pour simuler un JSON invalide (ex: {[3.5, 1.2, 4.9]})
    invalid_json_payload = "{[3.5, 1.2, 4.9]}"
    
    response = client.post(
        "/predict", 
        content=invalid_json_payload, 
        headers={"Content-Type": "application/json"}
    )
    # FastAPI/Pydantic renvoie automatiquement un statut 422 (Unprocessable Entity) 
    # ou 400 (Bad Request) lorsque le body ne respecte pas la syntaxe JSON.
    assert response.status_code == 422
# -----------------------------------------------------------------------------
# Cas smoke : valider que l'API est disponible
# -----------------------------------------------------------------------------
def test_predict_smoke():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "API is up and running!"
    
