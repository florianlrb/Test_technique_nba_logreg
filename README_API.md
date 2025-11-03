
# API : Question 2 (Intégration)

API REST **unitaire** pour servir le classifieur "5 ans NBA".
- **/health**: statut + info modèle
- **/schema**: liste des features attendues
- **/predict**: POST JSON avec stats d'un joueur → probabilité, prédiction, seuil utilisé

## Démarrage rapide

### 0) Pré-requis
Avoir généré le modèle et les seuils depuis le notebook :
- `model_final_explained.joblib`
- `thresholds_final_explained.json`

Par défaut, l'API les cherche dans `/mnt/data/`. Vous pouvez changer via variables d'env.

### 1) Installer les dépendances
```bash
pip install -r requirements_api.txt
```

### 2) Lancer l'API (port 8000)
```bash
uvicorn app_api:app --host 0.0.0.0 --port 8000
```

### 3) Tester

**Docs interactives** : http://127.0.0.1:8000/docs

**Health:**
```bash
curl -s http://127.0.0.1:8000/health | jq
```

**Prédiction (exemple):**
```bash
curl -s -X POST "http://127.0.0.1:8000/predict?mode=recall"   -H "Content-Type: application/json"   -d '{"GP": 82, "PTS": 10.5, "AST": 3.1, "REB": 4.2, "FG%": 0.45, "FT%": 0.79, "3P%": 0.36, "MIN": 24.0}'
```

Réponse (exemple) :
```json
{
  "model_path": "/mnt/data/model_final_explained.joblib",
  "mode": "recall",
  "threshold": 0.02,
  "probability": 0.41,
  "prediction": 1,
  "missing_filled": ["Has3PA","HasFTA","HasFGA","PTS/GP","PTS/MIN", "..."],
  "n_features": 37
}
```

## Variables d'environnement

- `MODEL_PATH` : chemin du bundle .joblib (par défaut `/mnt/data/model_final_explained.joblib`)
- `THRESHOLDS_PATH` : chemin du JSON des seuils (par défaut `/mnt/data/thresholds_final_explained.json`)
- `PRED_MODE` : `recall` | `balanced` | `precision` (par défaut `balanced`)
- `DEFAULT_THRESHOLD` : seuil par défaut si pas de JSON ou mode non applicable (par défaut `0.5`)

## Notes
- L'API recalcule `PTS/GP` et `PTS/MIN` si ces features figurent dans le schéma du modèle.
- Les features manquantes sont remplies à 0.0 et listées dans `missing_filled` pour la auditabilité.
